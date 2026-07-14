import math
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.encoder import MambaEncoder
from models.Metrics import acc,rand_index_score
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import fowlkes_mallows_score as fmi
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import f1_score as f1
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.optimize import linear_sum_assignment
import faiss
import matplotlib.pyplot as plt

from tools.tool import generate_pos_neg_index, MASK


class MultiViewEncoder(nn.Module):
    """Wrapper containing only the encoders and decoders (without training logic)"""
    def __init__(self, view_encoders, view_decoders, cross_view_decoders):
        super().__init__()
        self.view_encoders = view_encoders
        self.view_decoders = view_decoders
        self.cross_view_decoders = cross_view_decoders


class FMMVCC_Model(nn.Module):
    ''' Fuzzy Mamba-based Multi-View Contrastive Clustering for Time Series '''

    def __init__(
            self,
            data_loader,
            dataset_size,
            timesteps_len,
            batch_size,
            pretraining_epoch,
            n_cluster,
            dataset_name,
            input_dims,
            MaxIter=100,
            m=1.5,
            T1=2,
            output_dims=32,
            hidden_dims=64,
            n_layers=4,
            device='cuda',
            lr=0.001,
            max_train_length=4000,
            temporal_unit=0,
            mode='unidirectional',
            num_views = 4,
            separation_weight=0.5,
            balance_weight=0.2,
            ):

        super().__init__()
        self.device = device
        self.lr = lr
        self.num_cluster = n_cluster
        self.batch_size = batch_size
        self.T1 = T1
        self.m = m
        self.pretraining_epoch = pretraining_epoch
        self.MaxIter1 = MaxIter
        self.data_loader = data_loader
        self.dataset_size = dataset_size
        self.timesteps_len = timesteps_len
        self.input_dims = input_dims
        self.dataset_name = dataset_name
        self.latent_size = output_dims
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        self.n_layers = n_layers
        self.mode = mode
        self.num_views = num_views
        self.hard_w  = min(1.0, self.T1 / 20)
        self.mask_mode = 0
        self.dropout = 0.2
        self.separation_weight = separation_weight
        self.balance_weight = balance_weight
        # Training always uses masking.
        self.mask = True

        self.u_mean = torch.zeros([n_cluster, self.latent_size], device=self.device)
        
        # Encoders
        self.view_encoders = nn.ModuleDict()
        for i in range(self.num_views):
            self.view_encoders[f'view_{i}'] = MambaEncoder(input_dims=input_dims, 
                                                        output_dims=self.latent_size, 
                                                        hidden_dims=hidden_dims, 
                                                        n_layers=self.n_layers,
                                                        mask_mode=self.mask_mode,
                                                        dropout=self.dropout,
                                                        mode=self.mode,
                                                        ).to(self.device)

        # Cross view decoders
        self.cross_view_decoders = nn.ModuleDict()
        for i in range(self.num_views):
            for j in range(i + 1, self.num_views):
                self.cross_view_decoders[f'{i}_to_{j}'] = nn.Sequential(
                    nn.Linear(self.latent_size, hidden_dims * 2),
                    nn.LayerNorm(hidden_dims * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dims * 2, self.latent_size)
                )
                self.cross_view_decoders[f'{j}_to_{i}'] = nn.Sequential(
                    nn.Linear(self.latent_size, hidden_dims * 2),
                    nn.LayerNorm(hidden_dims * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dims * 2, self.latent_size)
                )
        self.cross_view_decoders.to(self.device)

        # Intra view decoders
        self.view_decoders = nn.ModuleDict()
        for i in range(self.num_views):
            self.view_decoders[f'view_{i}_decoder'] = nn.Sequential(
                nn.Linear(self.latent_size, hidden_dims * 2),
                nn.LayerNorm(hidden_dims * 2),
                nn.ReLU(),
                nn.Linear(hidden_dims * 2, hidden_dims),
                nn.LayerNorm(hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, input_dims)
            )
        self.view_decoders.to(self.device)
        
        # Weights for losses
        self.log_w_contrast = nn.Parameter(
            torch.zeros(1, device=self.device)
        )
        self.log_w_cross = nn.Parameter(
            torch.zeros(1, device=self.device)
        )
        self.log_w_rec = nn.Parameter(
            torch.zeros(1, device=self.device)
        )
        self.log_w_msc = nn.Parameter(
            torch.zeros(1, device=self.device)
        )
        
        # Wrap in MultiViewEncoder and AveragedModel for SWA
        self.encoder_module = MultiViewEncoder(
            self.view_encoders,
            self.view_decoders,
            self.cross_view_decoders
        )

        self.state_centers = nn.Parameter(
            F.normalize(
                torch.randn(n_cluster, self.latent_size, device=self.device),
                dim=1
            )
        )
        # Dynamic plot epochs: 0 (start), middle, last
        mid_pretrain = pretraining_epoch // 2
        mid_finetune = MaxIter // 2
        self.pretrain_plot_epochs = {0, mid_pretrain, pretraining_epoch - 1}
        self.finetune_plot_epochs = {0, mid_finetune, MaxIter - 1}
        self.__dict__['net'] = torch.optim.swa_utils.AveragedModel(self.encoder_module)

    def Pretraining(self, logger, use_view_losses: bool = True):
        """Args:
            use_view_losses: if False, the inter-view contrastive and
                cross-view reconstruction losses are skipped entirely (only
                the per-view reconstruction loss is optimized) -- used by
                the loss-component ablation (`Config.phase1_loss_ablation`,
                see client.py) to isolate FMMVCC's self-supervised loss
                terms while holding the multi-view Mamba architecture fixed.
        """
        logger.info('Pretraining...' if use_view_losses else 'Pretraining (reconstruction-only, view losses disabled)...')

        # Set all parameters to require gradients
        modules = [
            self.view_encoders,
            self.view_decoders,
            self.cross_view_decoders
        ]

        for module in modules:
            module.train()
            for param in module.parameters():
                param.requires_grad = True

        optimizer = optim.AdamW(
            list(self.view_encoders.parameters()) +
            list(self.view_decoders.parameters()) +
            list(self.cross_view_decoders.parameters()),
            lr=self.lr
        )
        if use_view_losses:
            optimizer.add_param_group({'params': [self.log_w_contrast,
                                                self.log_w_cross,
                                                self.log_w_rec,
                                                self.log_w_msc],
                                        "lr":1e-4})
        else:
            optimizer.add_param_group({'params': [self.log_w_rec,
                                                self.log_w_msc],
                                        "lr":1e-4})

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=12)

        # Logs
        loss_log = []
        acc_log = []
        nmi_log = []
        contr_log = []
        contr_par_log = []
        cross_log = []
        cross_par_log = []
        rec_log = []
        rec_par_log = []

        # Pretraining loop
        for T in range(0, self.pretraining_epoch):
            logger.info(f'Pretraining Epoch: {T + 1}')
            total_loss = 0
            total_contrastive_loss = 0
            total_contrastive_par = 0
            total_cross_loss = 0
            total_cross_par = 0
            total_rec_loss = 0
            total_rec_par = 0
            num_batches = 0
            for batch_idx, (x, target, _) in enumerate(self.data_loader):
                optimizer.zero_grad()

                x = x.to(self.device)
                sample_size = x.size(0)

                # Encode views
                z_views = self.encode_views(x)

                # Initialize losses
                contrastive_loss = torch.tensor(0.0, device=self.device)
                cross_loss = torch.tensor(0.0, device=self.device)
                reconstruction_loss = 0

                # Inter-view-Constrastive Loss (skipped entirely when
                # use_view_losses=False -- loss-component ablation)
                if use_view_losses:
                    for i in range(self.num_views):
                        for j in range(i + 1, self.num_views):
                            z_i = z_views[i]
                            z_j = z_views[j]

                            # Cross-view reconstruction
                            z_i_recon = self.cross_view_decoders[f'{i}_to_{j}'](z_i)
                            z_j_recon = self.cross_view_decoders[f'{j}_to_{i}'](z_j)
                            cross_loss += F.mse_loss(z_j_recon, z_j.detach())
                            cross_loss += F.mse_loss(z_i_recon, z_i.detach())

                            # Contrastive loss
                            view_contrastive_loss = self.contrastive_loss(z_i_recon, z_j_recon)
                            contrastive_loss += view_contrastive_loss
                    contrastive_loss = contrastive_loss / (self.num_views * (self.num_views - 1))
                    cross_loss = cross_loss / (self.num_views * (self.num_views - 1))

                # Reconstruction Loss
                for i in range(self.num_views):
                    z_i = z_views[i]
                    recon_i = self.view_decoders[f'view_{i}_decoder'](z_i)
                    reconstruction_loss += F.mse_loss(recon_i, x.detach())
                    if i == 0 and batch_idx == 0 and T in self.pretrain_plot_epochs:
                        data_train = []
                        data_target = []
                        for batch_x, batch_target, _ in self.data_loader:
                            data_train.append(batch_x)
                            data_target.append(batch_target)
                        data_train = torch.cat(data_train, dim=0).to(self.device)
                        data_target = torch.cat(data_target, dim=0).to(self.device)
                        u_train = self.encode_in_batches(data_train)
                        self.plot_analysis(
                            x,
                            recon_i,
                            u_train,
                            data_target,
                            T,
                            mode='pretraining',
                            config={'dataset_name': self.dataset_name}
                        )

                reconstruction_loss /= self.num_views

                # Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" (2018)
                w_rec = torch.exp(-self.log_w_rec)
                if use_view_losses:
                    w_contrast = torch.exp(-self.log_w_contrast)
                    w_cross = torch.exp(-self.log_w_cross)
                    loss = (
                        w_contrast * contrastive_loss + self.log_w_contrast +
                        w_cross * cross_loss + self.log_w_cross +
                        w_rec * reconstruction_loss + self.log_w_rec
                    )
                else:
                    # Reconstruction-only: contrastive/cross terms excluded
                    # from both the graph and the Kendall normalization (no
                    # log_w_contrast/log_w_cross gradient at all, since
                    # they're not in this optimizer's param group either).
                    w_contrast = torch.tensor(0.0, device=self.device)
                    w_cross = torch.tensor(0.0, device=self.device)
                    loss = w_rec * reconstruction_loss + self.log_w_rec
                loss.backward()

                # ---- Gradient Clipping ----
                torch.nn.utils.clip_grad_norm_(
                    list(self.view_encoders.parameters()) +
                    list(self.view_decoders.parameters()) +
                    list(self.cross_view_decoders.parameters()),
                    1.0
                )
                for p in [self.log_w_contrast, self.log_w_cross, self.log_w_rec]:
                    p.data.clamp_(-5, 5)
                optimizer.step()
                
                # Update SWA model
                self.__dict__['net'].update_parameters(self.encoder_module)

                # Accumulate losses for logging
                total_loss += loss.item()
                total_contrastive_loss += contrastive_loss.item()
                total_contrastive_par += w_contrast.item()
                total_cross_loss += cross_loss.item()
                total_cross_par += w_cross.item()
                total_rec_loss += reconstruction_loss.item()
                total_rec_par += w_rec.item()
                num_batches += 1

            average_loss = total_loss / num_batches
            average_contrastive_loss = total_contrastive_loss / num_batches
            average_contrastive_par = total_contrastive_par / num_batches
            average_cross_loss = total_cross_loss / num_batches
            average_cross_par = total_cross_par / num_batches
            average_rec_loss = total_rec_loss / num_batches
            average_rec_par = total_rec_par / num_batches
            scheduler.step(average_loss)

            ACC, NMI = self.Kmeans_model_evaluation(T, logger)
            acc_log.append(ACC)
            nmi_log.append(NMI)
            loss_log.append(average_loss)
            cross_log.append(average_cross_loss)
            cross_par_log.append(average_cross_par)
            rec_log.append(average_rec_loss)
            rec_par_log.append(average_rec_par)
            contr_log.append(average_contrastive_loss)
            contr_par_log.append(average_contrastive_par)
            logger.info(f"Epoch #{T + 1}: "
                f"loss={average_loss:.4f}, "
                f"contrastive_loss={average_contrastive_loss:.4f}, "
                f"w_contrast={average_contrastive_par:.4f}, "
                f"cross_loss={average_cross_loss:.4f}, "
                f"w_cross={average_cross_par:.4f}, "
                f"rec_loss={average_rec_loss:.4f}, "
                f"w_rec={average_rec_par:.4f}, "
                f"ACC={ACC}, "
                f"NMI={NMI}")

        file = os.getcwd() + f'/launches_{self.mode}/' + self.dataset_name + f'/pretraining_NViews{self.num_views}_Sep{self.separation_weight}_Bal{self.balance_weight}.csv' if self.mode != 'unidirectional' else os.getcwd() + '/launches/' + self.dataset_name + f'/pretraining_NViews{self.num_views}_Sep{self.separation_weight}_Bal{self.balance_weight}.csv'
        if not os.path.exists(os.getcwd() + '/launches/' + self.dataset_name) and self.mode == 'unidirectional':
            os.makedirs(os.getcwd() + '/launches/' + self.dataset_name)
        if not os.path.exists(os.getcwd() + f'/launches_{self.mode}/' + self.dataset_name) and self.mode != 'unidirectional':
            os.makedirs(os.getcwd() + f'/launches_{self.mode}/' + self.dataset_name)
        data = pd.DataFrame.from_dict({'pretraining': loss_log, 
                                    'contrastive_loss': contr_log, 
                                    'rec_loss': rec_log, 
                                    'ACC': acc_log, 
                                    'NMI': nmi_log,
                                    'w_contrast': contr_par_log,
                                    'w_cross': cross_par_log,
                                    'w_rec': rec_par_log}, orient='index')
        data.to_csv(file, index=True)
        
        # Plot loss curves
        self.plot_loss_curves(
            phase='pretraining',
            loss_log=loss_log,
            contr_log=contr_log,
            contr_par_log=contr_par_log,
            cross_log=cross_log,
            cross_par_log=cross_par_log,
            rec_log=rec_log,
            rec_par_log=rec_par_log
        )
        
        if T == self.pretraining_epoch-1:
            save_path = f'launches_{self.mode}/' + self.dataset_name + f'/Pretraining_phase_NViews{self.num_views}_Sep{self.separation_weight}_Bal{self.balance_weight}.pt' \
                if self.mode != 'unidirectional' else 'launches/' + self.dataset_name + f'/Pretraining_phase_NViews{self.num_views}_Sep{self.separation_weight}_Bal{self.balance_weight}.pt'
            torch.save({
                'view_encoders': self.view_encoders.state_dict(),
                'view_decoders': self.view_decoders.state_dict(),
                'cross_view_decoders': self.cross_view_decoders.state_dict(),
            }, save_path)

        return self.net

    def Finetuning(self, logger):
        # Initialiazation
        net_obj, self.u_mean = self.initialization(logger)
        self.__dict__['net'] = net_obj
        
        # Set all parameters to require gradients
        self.view_encoders.train()
        self.view_decoders.train()
        self.cross_view_decoders.train()
        
        for param in self.view_encoders.parameters():
            param.requires_grad = True
        for param in self.view_decoders.parameters():
            param.requires_grad = True
        for param in self.cross_view_decoders.parameters():
            param.requires_grad = True
            
        optimizer = optim.AdamW(
            list(self.view_encoders.parameters()) +
            list(self.view_decoders.parameters()) +
            list(self.cross_view_decoders.parameters()),
            lr=0.0001
        )

        optimizer.add_param_group({
            'params': [self.log_w_rec, 
                    self.log_w_msc],
            "lr":1e-4
        })
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=12)

        # Logs
        loss_log = []
        acc_log = []
        nmi_log = []
        rec_log = []
        cluster_log = []
        entropy_log = []
        balance_log = []
        separation_log = []
        rec_par_log = []
        cluster_par_log = []
        
        # ---- Finetuning loop ----
        for T in range(0, self.MaxIter1):
            logger.info(f'Finetuning Epoch: {T + 1}')
            total_loss = 0
            total_rec_loss = 0
            total_cluster_loss = 0
            total_entropy = 0
            total_balance = 0
            total_separation = 0
            total_rec_par = 0
            total_cluster_par = 0
            num_batches = 0

            # Update cluster centers every T1 epochs
            # if T % self.T1 == 1:
            new_centers = self.update_cluster_centers()
            alpha = 0.9
            self.u_mean = alpha * self.u_mean + (1 - alpha) * new_centers
            self.u_mean = F.normalize(self.u_mean, dim=1)
            
            for batch_idx, (x, target, index) in enumerate(self.data_loader):
                x = x.to(self.device)
                sample_size = x.size(0)

                optimizer.zero_grad()

                # Encoding
                z_views = self.encode_views(x)
                u = self.encode_with_pooling(x)  # [B, D]

                # Reconstruction Loss
                reconstruction_loss = 0
                for i in range(self.num_views):
                    recon = self.view_decoders[f'view_{i}_decoder'](z_views[i])
                    reconstruction_loss += F.mse_loss(recon, x.detach())
                    if i == 0 and batch_idx == 0 and T in self.finetune_plot_epochs:
                        data_train = []
                        data_target = []
                        for batch_x, batch_target, _ in self.data_loader:
                            data_train.append(batch_x)
                            data_target.append(batch_target)
                        data_train = torch.cat(data_train, dim=0).to(self.device)
                        data_target = torch.cat(data_target, dim=0).to(self.device)
                        u_train = self.encode_in_batches(data_train)
                        self.plot_analysis(
                            x,
                            recon,
                            u_train,
                            data_target,
                            T,
                            mode='finetuning',
                            config={'dataset_name': self.dataset_name}
                        )

                reconstruction_loss /= self.num_views

                # Clustering Loss
                loss_c, entropy, balance, separation = self.series_clustering_loss(u)
                loss_c /= self.num_views

                # Learnable weights for losses
                w_rec = torch.exp(-self.log_w_rec)
                w_msc = torch.exp(-self.log_w_msc)
                loss = (
                    w_rec * reconstruction_loss + self.log_w_rec +
                    w_msc * loss_c + self.log_w_msc
                )
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(self.view_encoders.parameters()) +
                    list(self.view_decoders.parameters()) +
                    list(self.cross_view_decoders.parameters()),
                    1.0
                )
                for p in [self.log_w_rec, self.log_w_msc]:
                    p.data.clamp_(-5, 5)

                optimizer.step()
                
                # Update SWA model
                self.__dict__['net'].update_parameters(self.encoder_module)

                # Accumulate losses for logging
                total_loss += loss.item()
                total_rec_loss += reconstruction_loss.item()
                total_cluster_loss += loss_c.item()
                total_entropy += entropy.item()
                total_balance += balance.item()
                total_separation += separation.item()
                total_rec_par += w_rec.item()
                total_cluster_par += w_msc.item()
                num_batches += 1
            
            # Average losses over batches
            average_loss = total_loss / num_batches
            average_rec_loss = total_rec_loss / num_batches
            average_cluster_loss = total_cluster_loss / num_batches
            average_entropy = total_entropy / num_batches
            average_balance = total_balance / num_batches
            average_separation = total_separation / num_batches
            average_rec_par = total_rec_par / num_batches
            average_cluster_par = total_cluster_par / num_batches
            scheduler.step(average_loss)

            ACC, NMI = self.model_evaluation(T, logger)

            acc_log.append(ACC)
            nmi_log.append(NMI)
            loss_log.append(average_loss)
            rec_log.append(average_rec_loss)
            cluster_log.append(average_cluster_loss)
            entropy_log.append(average_entropy)
            balance_log.append(average_balance)
            separation_log.append(average_separation)
            rec_par_log.append(average_rec_par)
            cluster_par_log.append(average_cluster_par)
            if T == self.MaxIter1 - 1:
                torch.save(self.state_dict(),
                        f'launches_{self.mode}/' + self.dataset_name + f'/Finetuning_phase_NViews{self.num_views}_Sep{self.separation_weight}_Bal{self.balance_weight}.pt' if self.mode != 'unidirectional' else 'launches/' + self.dataset_name + f'/Finetuning_phase_NViews{self.num_views}_Sep{self.separation_weight}_Bal{self.balance_weight}.pt'
                        )

                torch.save(self.u_mean,
                        f'launches_{self.mode}/' + self.dataset_name + f'/Centers_NViews{self.num_views}_Sep{self.separation_weight}_Bal{self.balance_weight}.pt' if self.mode != 'unidirectional' else 'launches/' + self.dataset_name + f'/Centers_NViews{self.num_views}_Sep{self.separation_weight}_Bal{self.balance_weight}.pt'
                        )

            logger.info(
                f"Finetuning Epoch: {T + 1}: "
                f"loss={average_loss:.4f}, "
                f"rec_loss={average_rec_loss:.4f},"
                f"w_rec={average_rec_par:.4f}, "
                f"cluster_loss={average_cluster_loss:.4f}, "
                f"w_cluster={average_cluster_par:.4f}, "
                f"ACC={ACC}, "
                f"NMI={NMI}"
            )

        file = os.getcwd() + f'/launches_{self.mode}/' + self.dataset_name + f'/finetuning_NViews{self.num_views}_Sep{self.separation_weight}_Bal{self.balance_weight}.csv' if self.mode != 'unidirectional' else os.getcwd() + '/launches/' + self.dataset_name + f'/finetuning_NViews{self.num_views}_Sep{self.separation_weight}_Bal{self.balance_weight}.csv'
        if not os.path.exists(os.getcwd() + '/launches/' + self.dataset_name) and self.mode == 'unidirectional':
            os.makedirs(os.getcwd() + '/launches/' + self.dataset_name)
        if not os.path.exists(os.getcwd() + f'/launches_{self.mode}/' + self.dataset_name) and self.mode != 'unidirectional':
            os.makedirs(os.getcwd() + f'/launches_{self.mode}/' + self.dataset_name)
        data = pd.DataFrame.from_dict({'finetuning': loss_log, 
                                    'rec_loss': rec_log, 
                                    'cluster_loss': cluster_log, 
                                    'entropy': entropy_log,
                                    'balance': balance_log,
                                    'separation': separation_log,
                                    'ACC': acc_log, 
                                    'NMI': nmi_log, 
                                    'w_rec': rec_par_log, 
                                    'w_cluster': cluster_par_log}, orient='index')
        data.to_csv(file, index=True)
        
        # Plot loss curves
        self.plot_loss_curves(
            phase='finetuning',
            loss_log=loss_log,
            rec_log=rec_log,
            rec_par_log=rec_par_log,
            cluster_log=cluster_log,
            entropy_log=entropy_log,
            balance_log=balance_log,
            separation_log=separation_log,
            cluster_par_log=cluster_par_log
        )

    def encode_in_batches(self, X, batch_size=64):
        embeddings = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size].clone().detach().float().to(X.device)
            with torch.no_grad():
                z = self.encode_with_pooling(batch).detach().cpu()
            embeddings.append(z)
        return torch.cat(embeddings).numpy()
    
    def plot_loss_curves(
        self,
        phase,
        loss_log,
        rec_log=None,
        rec_par_log=None,
        contr_log=None,
        contr_par_log=None,
        cross_log=None,
        cross_par_log=None,
        cluster_log=None,
        entropy_log=None,
        balance_log=None,
        separation_log=None,
        cluster_par_log=None,
    ):
        """Plot and save loss curves for pretraining or finetuning phase."""
        name_dataset = self.dataset_name
        folder = f'reconstruction_plots_views/{name_dataset}'
        os.makedirs(folder, exist_ok=True)

        epochs = np.arange(1, len(loss_log) + 1)
        series = [('total_loss', loss_log)]
        
        if phase == 'pretraining':
            if rec_log is not None:
                series.append(('rec_loss', rec_log))
            if contr_log is not None:
                series.append(('contrastive_loss', contr_log))
            if cross_log is not None:
                series.append(('cross_loss', cross_log))
            if rec_par_log is not None:
                series.append(('w_rec', rec_par_log))
            if contr_par_log is not None:
                series.append(('w_contrast', contr_par_log))
            if cross_par_log is not None:
                series.append(('w_cross', cross_par_log))
        elif phase == 'finetuning':
            if rec_log is not None:
                series.append(('rec_loss', rec_log))
            if cluster_log is not None:
                series.append(('cluster_loss', cluster_log))
            if entropy_log is not None:
                series.append(('entropy', entropy_log))
            if balance_log is not None:
                series.append(('balance', balance_log))
            if separation_log is not None:
                series.append(('separation', separation_log))
            if rec_par_log is not None:
                series.append(('w_rec', rec_par_log))
            if cluster_par_log is not None:
                series.append(('w_cluster', cluster_par_log))

        fig, axes = plt.subplots(len(series), 1, figsize=(10, 3 * len(series)), sharex=True)
        if len(series) == 1:
            axes = [axes]

        for axis, (title, values) in zip(axes, series):
            axis.plot(epochs, values, linewidth=2)
            axis.set_title(title)
            axis.set_ylabel(title)
            axis.grid(True, alpha=0.3)

        axes[-1].set_xlabel('epoch')
        fig.suptitle(f'{phase.capitalize()} loss curves', y=1.01)
        fig.tight_layout()
        fig.savefig(f'{folder}/{phase}_loss_curves.png', bbox_inches='tight')
        plt.close(fig)
    
    def plot_analysis(self, x_real, x_recon, z, target, T, mode, config):
        recon_0 = x_recon[0].detach().cpu()
        x_0 = x_real[0].detach().cpu()
        name_dataset = config['dataset_name']
        folder = f'reconstruction_plots_views/{name_dataset}/{mode}_epoch_{T}'
        os.makedirs(folder, exist_ok=True)

        # Plot at most 10 channels (not necessarily the first 10) to avoid overloading the plots
        numbers = min(10, x_0.shape[-1])
        list_channel = np.random.choice(x_0.shape[-1], numbers, replace=False)
        i = -1
        for channels in list_channel:
            i += 1
            plt.figure()
            plt.plot(x_0[:, channels].cpu().numpy(), label='Original')
            plt.plot(recon_0[:, channels].cpu().detach().numpy(), label='Reconstruction')
            plt.legend()
            plt.title(f'{mode} Epoch {T} - Channel {i}')
            # plt.show()
            plt.savefig(f'reconstruction_plots_views/{name_dataset}/{mode}_epoch_{T}/channel_{channels}.png')
        
        # Plottiamo anche lo spazio latente
        from sklearn.manifold import TSNE
        perplexity = min(30, len(z) - 1)  # t-SNE perplexity < number of samples
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=200,
            max_iter=1000,
            random_state=42
        )
        z_tsne = tsne.fit_transform(z)

        plt.figure()
        scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=target.cpu().numpy(), cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, ticks=range(len(torch.unique(target))))
        plt.title(f'{mode} Epoch {T} - Latent Space t-SNE')
        plt.savefig(f'reconstruction_plots_views/{name_dataset}/{mode}_epoch_{T}/latent_space_tsne.png')

    def initialization(self, logger):
        logger.info("-----initialization mode--------")

        # Load pretraining weights
        pretrain_path = f'launches_{self.mode}/' + self.dataset_name + f'/Pretraining_phase_NViews{self.num_views}_Sep{self.separation_weight}_Bal{self.balance_weight}.pt' \
            if self.mode != 'unidirectional' else 'launches/' + self.dataset_name + f'/Pretraining_phase_NViews{self.num_views}_Sep{self.separation_weight}_Bal{self.balance_weight}.pt'
        
        # Support both the old checkpoint format and the newer component-wise format.
        loaded = torch.load(pretrain_path, map_location=self.device)
        if isinstance(loaded, dict) and any(k in loaded for k in ('view_encoders', 'view_decoders', 'cross_view_decoders')):
            if 'view_encoders' in loaded:
                try:
                    self.view_encoders.load_state_dict(loaded['view_encoders'])
                    logger.info('Loaded view_encoders from pretraining checkpoint')
                except Exception as e:
                    logger.warning(f'Failed to load view_encoders: {e}')
            if 'view_decoders' in loaded:
                try:
                    self.view_decoders.load_state_dict(loaded['view_decoders'])
                    logger.info('Loaded view_decoders from pretraining checkpoint')
                except Exception as e:
                    logger.warning(f'Failed to load view_decoders: {e}')
            if 'cross_view_decoders' in loaded:
                try:
                    self.cross_view_decoders.load_state_dict(loaded['cross_view_decoders'])
                    logger.info('Loaded cross_view_decoders from pretraining checkpoint')
                except Exception as e:
                    logger.warning(f'Failed to load cross_view_decoders: {e}')
        else:
            try:
                self.load_state_dict(loaded, strict=False)
                logger.info('Loaded full model state dict from pretraining checkpoint')
            except Exception as e:
                logger.warning(f'Failed to load checkpoint with load_state_dict: {e}')

        self.encoder_module = MultiViewEncoder(
            self.view_encoders,
            self.view_decoders,
            self.cross_view_decoders
        )
        self.__dict__['net'] = torch.optim.swa_utils.AveragedModel(self.encoder_module)
        self.__dict__['net'].update_parameters(self.encoder_module)
        
        # Code data for KMeans
        datas = torch.zeros(
            self.dataset_size,
            self.latent_size,
            device=self.device
        )
        ii = 0
        for x, _, _ in self.data_loader:
            x = x.to(self.device)
            with torch.no_grad():
                u = self.encode_with_pooling(x)
            real_batch_size = u.size(0)
            datas[ii * self.batch_size:(ii * self.batch_size) + real_batch_size, :] = u
            ii += 1

        datas_np = datas.cpu().numpy().astype(np.float32)
        kmeans = faiss.Kmeans(
            d=self.latent_size,
            k=self.num_cluster,
            niter=30,
            gpu=True
        )

        kmeans.train(datas_np)
        self.u_mean = torch.from_numpy(kmeans.centroids).to(self.device)
        
        return self.__dict__['net'], self.u_mean
    
    def series_clustering_loss(self, u):

        K = self.num_cluster

        u_norm = F.normalize(u, dim=1)
        c_norm = F.normalize(self.u_mean, dim=1)

        sim = torch.matmul(u_norm, c_norm.T)

        q = torch.softmax(sim / 0.5, dim=1)

        # entropy
        entropy = -(q * torch.log(q + 1e-8)).sum(dim=1).mean()
        entropy = entropy / math.log(K)

        # balance
        p = q.mean(dim=0)
        balance = torch.sum(p * torch.log(p + 1e-8))
        balance = balance / math.log(K)

        # center separation
        center_sim = torch.matmul(c_norm, c_norm.T)

        mask = torch.eye(K, device=u.device).bool()
        center_sim = center_sim.masked_fill(mask, 0)

        separation = torch.mean(center_sim**2)

        loss = entropy + self.balance_weight * balance + self.separation_weight * separation

        return loss, entropy, balance, separation


    def encode_views(self, x):
        if self.mask:
            views = MASK(
                x,
                missing_rate=0.3,
                num_view=self.num_views
            )
        else:
            views = [x for _ in range(self.num_views)]

        latents = []
        for i, v in enumerate(views):
            encoder = self.view_encoders[f'view_{i}']
            z = encoder(v.to(self.device))
            latents.append(z)

        return latents
    
    def Kmeans_model_evaluation(self, T, logger):
        self.view_encoders.eval()

        datas_list = []
        label_true_list = []
        for x, target, _ in self.data_loader:
            x = x.to(self.device)
            with torch.no_grad():
                u = self.encode_with_pooling(x)
            if u is None:
                raise ValueError("encode_with_pooling(x) returned None")
            datas_list.append(u)
            label_true_list.append(target.numpy())

        datas = torch.cat(datas_list, dim=0)
        if datas.numel() == 0:
            raise ValueError("Empty datas tensor")
        label_true = np.concatenate(label_true_list, axis=0)
        datas_np = datas.detach().cpu().numpy().astype(np.float32)

        # FAISS KMeans on GPU
        kmeans = faiss.Kmeans(
            d=self.latent_size,
            k=self.num_cluster,
            niter=30,
            gpu=True
        )

        kmeans.train(datas_np)
        # self.u_mean = torch.from_numpy(kmeans.centroids).to(self.device) # Update centers with KMeans (the centers are updated during finetuning)

        # Assign clusters
        _, labels_pred = kmeans.index.search(datas_np, 1)
        labels_pred = labels_pred.ravel().astype(int)

        # Check length
        assert labels_pred.size == label_true.size, f"labels_pred ({labels_pred.size}) != label_true ({label_true.size})"

        ACC = acc(label_true, labels_pred, self.num_cluster)
        NMI = nmi(label_true, labels_pred)
        logger.info(f'ACC: {ACC}')
        logger.info(f'NMI: {NMI}')

        name = 'results' if self.mode == 'unidirectional' else f'results_{self.mode}'
        feature_dir = f'./{name}/{self.dataset_name}/features/NViews{self.num_views}_Sep{self.separation_weight}_Bal{self.balance_weight}'
        os.makedirs(feature_dir, exist_ok=True) 
        if T == 0:
            np.save(f'{feature_dir}/Start_Pretraining_R.npy', datas_np)
            np.save(f'{feature_dir}/Start_Pretraining_y_true.npy', label_true)
        if T == self.pretraining_epoch-1:
            np.save(f'{feature_dir}/End_Pretraining_R.npy', datas_np)
            np.save(f'{feature_dir}/End_Pretraining_y_true.npy', label_true)

        self.view_encoders.train()
        return ACC, NMI

    def update_cluster_centers(self):
        self.view_encoders.eval()
        for param in self.view_encoders.parameters():
            param.requires_grad = False
        den = torch.zeros([self.num_cluster]).to(self.device)
        num = torch.zeros([self.num_cluster, self.latent_size]).to(self.device)

        for x, _, _ in self.data_loader:
            x = x.to(self.device)
            with torch.no_grad():
                u = self.encode_with_pooling(x)

            p = self.cmp(u.unsqueeze(0).repeat(self.num_cluster, 1, 1), self.u_mean)
            p = torch.pow(p, self.m)
            for kk in range(0, self.num_cluster):
                den[kk] = den[kk] + torch.sum(p[:, kk])
                num[kk, :] = num[kk, :] + torch.matmul(p[:, kk], u)
                # num[kk, :] = num[kk, :] + torch.matmul(p[:, kk].mT, u)
        for kk in range(0, self.num_cluster):
            self.u_mean[kk, :] = torch.div(num[kk, :], den[kk].clamp(min=1e-8))

        self.u_mean = F.normalize(self.u_mean, dim=1)
        self.view_encoders.train()
        # Unfreeze parameters
        for param in self.view_encoders.parameters():
            param.requires_grad = True
        return self.u_mean

    def recompute_u_mean(self, logger=None):
        """Recompute `u_mean` (fuzzy c-means cluster centroids) from scratch via
        k-means over the current (frozen) encoder's embeddings of the local
        dataset. Used after loading a new global encoder, since the previous
        `u_mean` was computed w.r.t. the client's own (now-discarded) encoder.
        """
        self.view_encoders.eval()
        datas = torch.zeros(self.dataset_size, self.latent_size, device=self.device)
        ii = 0
        with torch.no_grad():
            for x, _, _ in self.data_loader:
                x = x.to(self.device)
                u = self.encode_with_pooling(x)
                real_batch_size = u.size(0)
                datas[ii * self.batch_size:(ii * self.batch_size) + real_batch_size, :] = u
                ii += 1

        datas_np = datas.cpu().numpy().astype(np.float32)
        kmeans = faiss.Kmeans(
            d=self.latent_size,
            k=self.num_cluster,
            niter=30,
            gpu=True
        )
        kmeans.train(datas_np)
        self.u_mean = torch.from_numpy(kmeans.centroids).to(self.device)
        if logger is not None:
            logger.info("Recomputed u_mean via k-means on frozen global encoder embeddings")
        return self.u_mean

    def cmp(self, u, u_mean):
        real_batch_size = u.size(1)
        p = torch.zeros([real_batch_size, self.num_cluster]).to(self.device)
        for j in range(0, self.num_cluster):
            p[:, j] = torch.sum(torch.pow(u[j, :, :] - u_mean[j, :].unsqueeze(0).repeat(real_batch_size, 1), 2), dim=1)
        # a point landing exactly on a cluster center gives distance 0, and
        # raising 0 to a negative power (-1/(m-1)) blows up to inf -> nan
        p = torch.clamp(p, min=1e-8)
        p = torch.pow(p, -1 / (self.m - 1))
        sum1 = torch.sum(p, dim=1)
        p = torch.div(p, sum1.unsqueeze(1).repeat(1, self.num_cluster))
        return p

    def model_evaluation(self, T, logger):
        datas = np.zeros([self.dataset_size, self.latent_size])
        pred_labels = np.zeros(self.dataset_size)
        true_labels = np.zeros(self.dataset_size)
        ii = 0
        for x, target, _ in self.data_loader:
            x = x.to(self.device)
            u = self.encode_with_pooling(x)
            real_batch_size = u.size(0)
            datas[ii * self.batch_size:(ii * self.batch_size) + real_batch_size, :] = u.data.cpu().numpy()

            u = u.unsqueeze(0).repeat(self.num_cluster, 1, 1)
            p = self.cmp(u, self.u_mean)
            y = torch.argmax(p, dim=1)
            y = y.cpu()
            y = y.numpy()
            pred_labels[(ii) * self.batch_size:(ii * self.batch_size) + real_batch_size] = y
            true_labels[(ii) * self.batch_size:(ii * self.batch_size) + real_batch_size] = target.numpy()
            ii = ii + 1

        ACC = acc(true_labels, pred_labels, self.num_cluster)
        NMI = nmi(true_labels, pred_labels)
        logger.info(f'ACC: {ACC}')
        logger.info(f'NMI: {NMI}')
        name = 'results' if self.mode == 'unidirectional' else f'results_{self.mode}'
        feature_dir = f'./{name}/{self.dataset_name}/features/NViews{self.num_views}_Sep{self.separation_weight}_Bal{self.balance_weight}'
        os.makedirs(feature_dir, exist_ok=True)
        if T == 0:
            np.save(f'{feature_dir}/Start_Finetuning_R.npy', datas)
            np.save(f'{feature_dir}/Start_Finetuning_y_pred.npy', pred_labels)
            np.save(f'{feature_dir}/Start_Finetuning_y_true.npy', true_labels)
        if T == self.MaxIter1-1:
            np.save(f'{feature_dir}/End_Finetuning_End_Finetuning_R.npy', datas)
            np.save(f'{feature_dir}/End_Finetuning_y_pred.npy', pred_labels)
            np.save(f'{feature_dir}/End_Finetuning_y_true.npy', true_labels)
        self.view_encoders.train()
        for param in self.view_encoders.parameters():
            param.requires_grad = True

        return ACC, NMI

    def encode_with_pooling(self, x, return_cluster_weights: bool = False):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        else:
            x = x.to(self.device)
        assert x.ndim == 3

        was_training_encoders = self.view_encoders.training
        self.view_encoders.eval()
        z_views = []
        for i in range(self.num_views):
            encoder = self.view_encoders[f'view_{i}']
            z_views.append(encoder(x))

        pooled_views = []
        cluster_weights_views = []
        for z in z_views:
            B, T, D = z.shape

            centers = F.normalize(self.state_centers, dim=1)
            z_norm = F.normalize(z, dim=2)

            sim = torch.matmul(z_norm, centers.T)   # [B,T,K]
            weights = torch.softmax(sim.mean(dim=2), dim=1).unsqueeze(-1)
            pooled = torch.sum(z * weights, dim=1)

            pooled_views.append(pooled)
            if return_cluster_weights:
                # Softmax over time for each cluster (dim=1), BEFORE the
                # averaging over clusters done above for `weights`. Exposes the
                # per-cluster temporal contribution instead of a single average
                # weight shared by all clusters.
                cluster_weights_views.append(torch.softmax(sim, dim=1))

        fused = torch.stack(pooled_views, dim=0).mean(dim=0)
        fused = F.normalize(fused, dim=1)

        if was_training_encoders:
            self.view_encoders.train()

        if return_cluster_weights:
            return fused, cluster_weights_views, pooled_views

        return fused

    def prototype_path(self, cluster_weights):
        """Temporal sequence of the most relevant prototypes (per view).

        Given w[b,t,c] (see `encode_with_pooling(..., return_cluster_weights=True)`),
        computes argmax_c w[b,t,:] for each timestep: the index of the most
        relevant prototype (cluster) at every time instant.

        Args:
            cluster_weights: tensor [B,T,num_cluster], or a list/tuple of such
                tensors (one per view) as returned by
                encode_with_pooling(..., return_cluster_weights=True).

        Returns:
            torch.Tensor [B,T] if a single tensor was passed, or a list of
            tensors [B,T] (one per view) if a list was passed.
        """
        if isinstance(cluster_weights, (list, tuple)):
            return [w.argmax(dim=-1) for w in cluster_weights]
        return cluster_weights.argmax(dim=-1)

    def view_ablation_importance(self, x):
        """Importance of each view on the final membership (view-ablation).

        Computes the baseline membership with predict_membership(x) (all
        views), then for each view i reconstructs the fusion excluding
        pooled_views[i] from the average (same fusion formula used in
        encode_with_pooling: average + normalize), recomputes the membership
        with cmp() and measures the L2 norm against the baseline.

        Args:
            x: tensor (or array-like) [B, T, D] of multivariate time series.

        Returns:
            torch.Tensor [num_views, B]: importance score (L2 norm of the
            membership difference) for each view and sample.
        """
        with torch.no_grad():
            baseline = self.predict_membership(x)

            _, _, pooled_views = self.encode_with_pooling(x, return_cluster_weights=True)

            scores = []
            for i in range(self.num_views):
                remaining = [pv for j, pv in enumerate(pooled_views) if j != i]
                fused_ablated = torch.stack(remaining, dim=0).mean(dim=0)
                fused_ablated = F.normalize(fused_ablated, dim=1)

                u = fused_ablated.unsqueeze(0).repeat(self.num_cluster, 1, 1)
                ablated_membership = self.cmp(u, self.u_mean)

                scores.append(torch.norm(baseline - ablated_membership, p=2, dim=1))

            return torch.stack(scores, dim=0)

    def nearest_medoid_prototypes(self, raw_series, chunk_size=None):
        """"Nearest-medoid" prototype for each cluster (zero cost, no decoder).

        For each cluster c, finds the real sample in `raw_series` whose pooled
        embedding is closest to self.u_mean[c], using the same metric as
        cmp(): squared Euclidean distance between normalized embedding and
        normalized center embedding (encode_with_pooling already normalizes
        its own output with F.normalize). Returns the corresponding raw series
        as an interpretable prototype of the cluster, without any generative
        decoder.

        Args:
            raw_series: tensor or array [N,T,D] of raw time series (e.g. the
                training set, or a representative subset of it).
            chunk_size: if given, encode_with_pooling(raw_series) is run in
                chunks of this many samples instead of a single unbatched
                forward pass over all N samples. Does not change the result,
                only the peak memory needed to compute it (a single forward
                pass over a large N, e.g. a whole train set, can OOM since
                encode_with_pooling has no batching of its own).

        Returns:
            tuple (prototypes, indices):
                prototypes: tensor [num_cluster,T,D], the raw series closest
                    to each center.
                indices: tensor [num_cluster] with the index in raw_series of
                    the sample chosen for each cluster.
        """
        if isinstance(raw_series, np.ndarray):
            raw_series_t = torch.from_numpy(raw_series).float().to(self.device)
        else:
            raw_series_t = raw_series.to(self.device)

        with torch.no_grad():
            if chunk_size is None:
                embeddings = self.encode_with_pooling(raw_series_t)  # already L2-normalized [N, latent]
            else:
                embeddings = torch.cat([
                    self.encode_with_pooling(raw_series_t[start:start + chunk_size])
                    for start in range(0, raw_series_t.shape[0], chunk_size)
                ], dim=0)
            centers = F.normalize(self.u_mean, dim=1)

            diff = embeddings.unsqueeze(0) - centers.unsqueeze(1)  # [num_cluster, N, latent]
            dist_sq = torch.sum(diff ** 2, dim=2)  # [num_cluster, N]
            indices = torch.argmin(dist_sq, dim=1)  # [num_cluster]

            prototypes = raw_series_t[indices]
        return prototypes, indices

    def cluster_center_similarity(self):
        """Cosine similarity between cluster centers (self.u_mean).

        Useful in fuzzy clustering (unlike classification) because clusters
        are not necessarily well separated: two centers with high similarity
        indicate regions of the latent space that are very close or
        overlapping, even though they remain formally distinct clusters. Same
        metric (cosine on normalized centers) as the `center_sim` term
        computed inside `series_clustering_loss()` during training, here
        exposed as the full [C,C] matrix instead of as a loss scalar.

        Returns:
            torch.Tensor [num_cluster, num_cluster]: cosine similarity between
            the centers, with values in [-1, 1] and diagonal equal to 1.
        """
        centers_norm = F.normalize(self.u_mean, dim=1)
        return torch.matmul(centers_norm, centers_norm.T)

    def mask_instance_loss_with_mixup(self, z1, z2, pseudo_label=None):
        # Evaluation of the loss function with mixup
        B, T = z1.size(0), z1.size(1)
        temp = 1.0

        # If no pseudo label is provided, set it to -1
        if pseudo_label == None:
            pseudo_label = torch.full((B,), -1, dtype=torch.int64).to(self.device)

        # If batch size is 1, return 0 (needs at least 2 samples)
        if B == 1:
            return z1.new_tensor(0.)

        pseudo_label = pseudo_label.to(z1.device)

        # Hard weight
        hard_w = self.hard_w

        # Generate hard positive and negative samples and evaluate h1
        pos_indices, neg_indices = generate_pos_neg_index(pseudo_label)
        uni_z1 = hard_w * z1[pos_indices, :, :] + (1 - hard_w) * z1[neg_indices, :, :].view(z1.size())

        # Generate hard positive and negative samples and evaluate h2
        pos_indices, neg_indices = generate_pos_neg_index(pseudo_label)
        uni_z2 = hard_w * z2[pos_indices, :, :] + (1 - hard_w) * z2[neg_indices, :, :].view(z2.size())

        # Concatenate the original and hard samples
        z = torch.cat([z1, z2, uni_z1, uni_z2], dim=0)

        # Transpose the matrix (loss evaluated per timestep)
        z = z.transpose(0, 1) 
        
        # Similarity matrix (dot product) --> preparing the denominator of the loss function
        sim = torch.matmul(z[:, : 2 * B, :], z.transpose(1, 2))

        # Invalid index
        invalid_index = pseudo_label == -1

        # Mask cluster-aware
        mask = torch.eq(
            pseudo_label.view(-1, 1),
            pseudo_label.view(1, -1)
        ).to(z1.device)

        # Invalid index
        mask[invalid_index, :] = False
        mask[:, invalid_index] = False

        # Mask out self-contrast
        mask_eye = torch.eye(B).float().to(z1.device)
        mask &= ~(mask_eye.bool())
        mask = mask.float()

        # Adapting to sim shape
        mask = mask.repeat(2, 4)
        mask_eye = mask_eye.repeat(2, 4)

        # Initializing logits mask
        logits_mask = torch.ones(2 * B, 4 * B).to(z1.device)

        # Deleting self-contrast
        rows = torch.arange(2 * B).view(-1, 1).to(z1.device)
        logits_mask = logits_mask.scatter(1, rows, 0)

        # Deleting positive samples (for denominator)
        logits_mask *= 1 - mask

        # Building self-positive samples (for numerator)
        mask_eye = mask_eye * logits_mask

        # Numeric Stabilization (avoiding overflow for exp)
        logits = sim
        logits_max = torch.max(logits, dim=-1, keepdim=True)[0]
        logits = logits - logits_max

        # It's the full denominator of the loss function
        neg_exp_logits = torch.exp(logits) * logits_mask
        neg_exp_log_sum = neg_exp_logits.sum(-1, keepdim=True)

        # It's the full numerator of the loss function
        pos_exp_log = torch.exp(logits)

        # Total loss
        prob = pos_exp_log / (neg_exp_log_sum + 1e-10)

        # Selection of the positive samples (instance-level positive base)
        prob = prob[:, 0:B, B:2 * B]

        # Building final positive mask
        mask = mask[:B, : B]
        self_mask = mask_eye[:B, B:2 * B]
        diffaug_cluster_mask = mask
        pos_mask = (self_mask + diffaug_cluster_mask)

        # Final positive probability sum
        pos_prob_sum = (prob * pos_mask.unsqueeze(0)).sum(-1)

        # Log probability
        log_prob = torch.log(pos_prob_sum + 1e-10)

        # Final log probability (mean over timesteps)
        log_prob = log_prob.sum(dim=0) / T

        # Final loss
        instance_loss = -log_prob
        instance_loss = instance_loss.mean()

        return instance_loss

    def contrastive_loss(
        self,
        z1, 
        z2, 
        mask=False,
        pseudo_label=None,
        alpha=0.8,
    ):
        """
        Instance CL
        """

        instance_loss = torch.tensor(0., device=z1.device)

        # -------- Instance-level CL --------
        if alpha > 0:
            if not mask:
                instance_loss += self.mask_instance_loss_with_mixup(z1, z2)
            else:
                instance_loss += self.mask_instance_loss_with_mixup(z1, z2, pseudo_label)

        return instance_loss

    def pooling(self, x_whole_list):

        pooled_list = []

        for x_view in x_whole_list:

            x_view = x_view.to(self.device)
            num_samples, seq_len, feature_dims = x_view.shape

            var_per_sample = torch.var(x_view, dim=2).sum(dim=1)  # (num_samples,)
            var_min = var_per_sample.min()
            var_max = var_per_sample.max()
            time_steps = torch.arange(seq_len, device=self.device).float()

            pooled_samples = []
            for i in range(num_samples):

                alpha = (var_per_sample[i] - var_min) / (var_max - var_min + 1e-8)


                logits = alpha * time_steps
                weights = torch.softmax(logits, dim=0)


                pooled = torch.sum(x_view[i] * weights.view(-1, 1), dim=0)
                pooled_samples.append(pooled)

            pooled_list.append(torch.stack(pooled_samples).cpu().numpy())

        return pooled_list
    
    # Define the function to evaluate real test data
    def eval_with_test_data(self, dataset_name, logger, data_loader, save=False):
        self.view_encoders.eval()

        data = np.zeros([self.dataset_size, self.timesteps_len, self.input_dims])
        reps = np.zeros([self.dataset_size, self.latent_size])
        label_true = np.zeros(self.dataset_size)
        label_pred = np.zeros(self.dataset_size)

        ii = 0
        for x, target, _ in data_loader:
            x = x.to(self.device)
            with torch.no_grad():
                u = self.encode_with_pooling(x)
            real_batch_size = u.size(0)

            reps[ii * self.batch_size: ii * self.batch_size + real_batch_size, :] = u.data.cpu().numpy()
            data[ii * self.batch_size: ii * self.batch_size + real_batch_size, :, :] = x.cpu().numpy()

            # Get predicted labels
            u = u.unsqueeze(0).repeat(self.num_cluster, 1, 1)
            p = self.cmp(u, self.u_mean)
            y = torch.argmax(p, dim=1)
            y = y.cpu().numpy()

            label_true[ii * self.batch_size: ii * self.batch_size + real_batch_size] = target.numpy()
            label_pred[ii * self.batch_size: ii * self.batch_size + real_batch_size] = y

            ii = ii + 1

        # Evaluate performance
        logger.info("-------testdata_Evaluate---------")
        
        # --- Label alignment ---
        label_pred = label_pred.astype(int)
        label_true = label_true.astype(int)
        w = np.zeros((self.num_cluster, self.num_cluster))
        for i in range(label_pred.size):
            w[label_pred[i], label_true[i]] += 1
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        # mapping cluster -> class
        mapping = dict(zip(row_ind, col_ind))
        # apply mapping
        label_pred_aligned = np.array([
            mapping.get(label, label) for label in label_pred
        ])
        # label_pred_aligned = np.array([mapping.get(label, label) for label in label_pred])

        name = 'results' if self.mode == 'unidirectional' else f'results_{self.mode}'
        save_dir = f'./{name}/{self.dataset_name}/label/'
        os.makedirs(save_dir, exist_ok=True)

        df_true = pd.DataFrame(label_true, columns=['label_true'])
        df_true.to_csv(f'{save_dir}/{dataset_name}_label_true.csv', index=False)

        df_pred = pd.DataFrame(label_pred, columns=['label_pred'])
        df_pred.to_csv(f'{save_dir}/{dataset_name}_label_pred.csv', index=False)
        
        accuracy = acc(label_true, label_pred, self.num_cluster)
        nmi_score = nmi(label_true, label_pred)
        ari_score = ari(label_true, label_pred)
        fmi_score = fmi(label_true, label_pred)
        test_ri = rand_index_score(label_pred, label_true)
        f1_score = f1(label_true, label_pred_aligned, average='macro')

        self.view_encoders.train()
        return accuracy, nmi_score, ari_score, test_ri, fmi_score, f1_score

    def predict_membership(self, x):
        """Final fuzzy membership (Fuzzy C-Means on pooled embedding, via cmp()).

        This is the ONLY membership function to use for any XAI explanation
        (intrinsic or post-hoc): it composes encode_with_pooling() and cmp()
        with self.u_mean, exactly as model_evaluation() and
        eval_with_test_data() do to produce the final predictions (ACC/NMI/ARI).
        Do NOT use the softmax pseudo-membership `q` from series_clustering_loss()
        (a gradient surrogate, never used at inference) nor the internal
        temporal pooling weights of encode_with_pooling() (which are not a
        cluster membership) as an explanation target.

        Args:
            x: tensor (or array-like) [B, T, D] of multivariate time series.

        Returns:
            torch.Tensor [B, num_cluster]: fuzzy membership, rows summing to 1.
        """
        u = self.encode_with_pooling(x)
        u = u.unsqueeze(0).repeat(self.num_cluster, 1, 1)
        p = self.cmp(u, self.u_mean)
        return p

    def load_for_xai(self, checkpoint_path, centers_path, map_location=None):
        """Hardened (safety-checked) loading of the model for the XAI pipeline.

        Loads SEPARATELY the model checkpoint (Finetuning_phase_*.pt, i.e.
        self.state_dict()) and the centers file (Centers_*.pt), since
        self.u_mean is neither an nn.Parameter nor a registered buffer and is
        therefore not saved inside state_dict(). Never proceeds silently: it
        fails loudly if either file is missing, if u_mean has an unexpected
        shape, or if u_mean is still the default zero tensor defined in
        __init__.
        """
        if map_location is None:
            map_location = self.device

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}"
            )
        if not os.path.isfile(centers_path):
            raise FileNotFoundError(
                f"Centers file (u_mean) not found: {centers_path}. "
                "u_mean is not included in the model checkpoint (it is not an "
                "nn.Parameter nor a registered buffer) and must be loaded "
                "separately from this file."
            )

        state_dict = torch.load(checkpoint_path, map_location=map_location)
        self.load_state_dict(state_dict, strict=True)

        u_mean = torch.load(centers_path, map_location=map_location)
        if not isinstance(u_mean, torch.Tensor):
            raise TypeError(
                f"The centers file {centers_path} does not contain a torch.Tensor "
                f"(found {type(u_mean)})"
            )

        expected_shape = (self.num_cluster, self.latent_size)
        if tuple(u_mean.shape) != expected_shape:
            raise ValueError(
                f"u_mean loaded from {centers_path} has shape {tuple(u_mean.shape)}, "
                f"expected {expected_shape}"
            )

        if torch.all(u_mean == 0):
            raise ValueError(
                f"u_mean loaded from {centers_path} is still the default zero "
                "tensor (torch.zeros in __init__): the centers were not "
                "trained, or the file is corrupted/empty. Cannot proceed."
            )

        self.u_mean = u_mean.to(self.device)
        return self