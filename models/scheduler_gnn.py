import os
import numpy as np
import matplotlib.pyplot as plt
# import torch
from torch.utils.data import random_split
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
# import torch_geometric as pyg
# import torch_geometric.nn as pyg_nn    
from torch_geometric.loader import DataLoader
from torch_geometric.nn.inits import reset, uniform
# import torch.nn.functional as F
from joblib import dump, load
import wandb
from dataset.MatDataset import Sub_JHTDB
# from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from models.model import *
from utils import *
import datetime


class GNNPartitionScheduler():
    def __init__(self, exp_name, num_partitons, dataset, model=None, train=True, encoder=None, classifier=None):
        super(GNNPartitionScheduler, self).__init__()
        self.name = exp_name
        self.num_partitions = num_partitons
        if num_partitons != 1:
            self.encoder = encoder
            self.classifier = classifier
            raise NotImplementedError('Currently only support num_partitions=1')
            
        self.model = model
        self.dataset = dataset

        self.subsets = self._train_partitions(num_partitons, train)
        if not train:
            self.models = self._load_models()

    def get_sub_dataset(self):
        return self.subsets
    
    def _initialize_model(self):
        return self.model
    
    def _load_models(self):
        models = []
        for i in range(self.num_partitions):
            model = self._initialize_model()
            model.load_state_dict(torch.load('logs/models/collection_{}/partition_{}.pth'.format(self.name, i), map_location=torch.device('cpu')))
            models.append(model)
        return models
    
    def _train_partitions(self, num_partitions, train):
        # if num_partitions == 1: skip the clustering and directly train the model
        if num_partitions == 1:
            return [self.dataset]
        if train:
            os.makedirs('logs/models/collection_{}'.format(self.name), exist_ok=True)
            # train the encoder on the dataset
            self.encoder.train(self.dataset, save_model=True, path='logs/models/collection_{}'.format(self.name))
            # dump(self.encoder.model, 'logs/models/collection_{}/encoder.joblib'.format(self.name))
            latent_space = self.encoder.get_latent_space(self.dataset)
            print('Latent space shape:', latent_space.shape)
            # cluster the latent space into different groups
            self.classifier.train(latent_space, save_model=True, path='logs/models/collection_{}'.format(self.name))
            # dump(self.classifier.model, 'logs/models/collection_{}/classifier.joblib'.format(self.name))
            labels = self.classifier.cluster(latent_space)
            # print('Labels:', labels)
        else:
            # load the pre-trained encoder and classifier
            self.encoder.load_model('logs/models/collection_{}'.format(self.name))
            self.classifier.load_model('logs/models/collection_{}'.format(self.name))
            latent_space = self.encoder.get_latent_space(self.dataset)
            labels = self.classifier.cluster(latent_space)

        # partition the dataset into different subsets
        subsets = []
        for i in range(num_partitions):
            idx = np.where(labels == i)[0]
            # print(f'Partition {i}: {len(idx)} samples')
            subsets.append(Sub_JHTDB(self.dataset.root, idx))

        return subsets
    
    def _train_sub_models(self, model, train_config, device, subset_idx=None, is_parallel=False):
        models = []
        if subset_idx is not None:
            subsets = [self.subsets[idx] for idx in subset_idx]
        else:
            subsets = self.subsets

        if is_parallel:
            subset = subsets[0]
            train_dataset, val_dataset = random_split(
                subset, 
                [int(0.8 * len(subset)), len(subset) - int(0.8 * len(subset))]
            )
            world_size = torch.cuda.device_count()
            mp.spawn(
                self._train_sub_models_parallel,
                args=(self.model, self.name, world_size, train_dataset, val_dataset, train_config),
                nprocs=world_size,
                join=True,
            )

        else:
            for i, subset in enumerate(subsets):
                # print(len(subset))
                wandb.init(project='domain_partition_scheduler', group='partition_training', config=train_config)
                train_dataset, val_dataset = random_split(subset, [int(0.8 * len(subset)), len(subset) - int(0.8 * len(subset))])
                # print(train_config['batch_size'])
                train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)
                # model = self._initialize_model(self.model, 8, 8, width=64)

                model = model.to(device)
                # if is_parallel:
                #     model = nn.DistributedDataParallel(model, device_ids=[device])
                #     model = model.to(device)
                # else:
                #     model = model.to(device)

                criterion = torch.nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
                epochs = train_config['epochs']
                log_interval = train_config['log_interval']
                val_interval = train_config['val_interval']
                best_loss = np.inf
                for epoch in range(epochs):
                    model.train()
                    train_loss = 0
                    for batch in train_loader:
                        optimizer.zero_grad()
                        batch = batch.to(device)
                        out = model(batch.x, batch.edge_index, batch.edge_attr)
                        loss = criterion(out, batch.y)
                        loss_l_infty = torch.max(torch.abs(out - batch.y))
                        loss += 0.1 * loss_l_infty
                        loss.backward()
                        # monitor gradient during training
                        # for name, param in model.named_parameters():
                        #     if param.grad is not None:
                        #         print({f'{name}_grad': param.grad.norm()})
                        optimizer.step()
                        # print(optimizer.param_groups[0]['lr'])
                        train_loss += loss.item()
                    train_loss /= len(train_loader)
                    wandb.log({'train_loss': train_loss})
                    if epoch % log_interval == 0:
                        print(f'Epoch {epoch}: Train loss: {train_loss}')
                    if epoch % val_interval == 0:
                        model.eval()
                        val_loss = 0
                        with torch.no_grad():
                            for batch in val_loader:
                                batch = batch.to(device)
                                out = model(batch.x, batch.edge_index, batch.edge_attr)
                                loss = criterion(out, batch.y)
                                loss_l_infty = torch.max(torch.abs(out - batch.y))
                                loss += 0.1 * loss_l_infty
                                val_loss += loss.item()
                            val_loss /= len(val_loader)
                            wandb.log({'val_loss': val_loss})
                            # plot_3d_prediction(batch[0], out, save_mode='wandb', path='logs/figures/{}'.format(self.name))
                            if val_loss < best_loss:
                                best_loss = val_loss
                                os.makedirs('logs/models/collection_{}'.format(self.name), exist_ok=True)
                                torch.save(model.state_dict(), 'logs/models/collection_{}/partition_{}.pth'.format(self.name, i))
                                print(f'Epoch {epoch}: Validation loss: {val_loss}')
                    scheduler.step()

                models.append(model)
                wandb.finish()
        return models
    
    def train(self, train_config, subset_idx=None):
        # for parallel training on multiple gpus
        if torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs')
            self._train_sub_models(self.model, train_config, torch.device('cuda'), subset_idx, is_parallel=True)
        elif torch.cuda.device_count() == 1:
            print('Using single GPU')
            self._train_sub_models(self.model, train_config, torch.device('cuda'), subset_idx, is_parallel=False)
        else:
            print('Using CPU')
            self._train_sub_models(self.model, train_config, torch.device('cpu'), subset_idx, is_parallel=False)

    def predict(self, x):
        pred_y_list = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                if torch.cuda.device_count() > 1:
                    try:
                        # Check if start method is already set
                        current_method = mp.get_start_method(allow_none=True)
                        if current_method is None:
                            # Only set if not already set
                            mp.set_start_method('spawn', force=False)
                    except RuntimeError:
                        # If we can't set it, just use whatever is already set
                        pass
                    
                    # split the list of x into chunks and predict in parallel
                    world_size = torch.cuda.device_count()
                    print(f'Using {world_size} GPUs')
                    model.share_memory()
                    num_samples_per_gpu = len(x) // world_size
                    x_input = [x[i*num_samples_per_gpu:(i+1)*num_samples_per_gpu] for i in range(world_size-1)]
                    x_input.append(x[(world_size-1)*num_samples_per_gpu:])
                    manager = mp.Manager()
                    result_queue = manager.dict()
                    processes = []
                    for rank in range(world_size):
                        p = mp.Process(
                            target=self._predict_sub_models_parallel,
                            args=(rank, model, x_input[rank], result_queue, world_size)
                        )
                        p.start()
                        processes.append(p)

                    for p in processes:
                        p.join()

                    for i in range(world_size):
                        pred_y_list.extend(result_queue[i])

                else:
                    model = model.to('cuda')
                    for i in range(len(x)):
                        x_data = x[i]
                        x_data = x_data.to('cuda')
                        pred_y = model(x_data.x, x_data.edge_index, x_data.edge_attr)
                        pred_y_list.append(pred_y)
        return pred_y_list
    
    @staticmethod
    def _predict_sub_models_parallel(rank, model, x_local, results, world_size):
        # Setup distributed process group
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=1000))

        local_device = f"cuda:{rank}"

        # Initialize model and wrap it with DistributedDataParallel
        model = model.to(local_device)

        pred_y_list = []
        for i in range(len(x_local)):
            local_data = x_local[i]
            local_data = local_data.to(local_device)
            pred_y = model(local_data.x, local_data.edge_index, local_data.edge_attr)
            pred_y_list.append(pred_y.detach().cpu())
            # print location of pred_y_list[i]
            # print(f'Rank {rank}: {pred_y_list[i].device}')

            del local_data, pred_y

        # Cleanup distributed process group
        dist.destroy_process_group()
        # pred_y_list = torch.cat(pred_y_list, dim=0)
        results[rank] = pred_y_list
                 
    @staticmethod
    def _train_sub_models_parallel(rank, model, name, world_size, train_dataset, val_dataset, train_config):
        # Setup distributed process group
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=1000))

        local_device = f"cuda:{rank}"
        models = []

        if rank == 0:
            wandb.init(project='domain_partition_scheduler', group='partition_training', config=train_config)

        # Iterate through subsets
        # for i, subset in enumerate(subsets):
        # # Split dataset
        # train_dataset, val_dataset = random_split(
        #     subset, 
        #     [int(0.8 * len(subset)), len(subset) - int(0.8 * len(subset))]
        # )
        # split train dataset based on rank
        # train_dataset_actual = train_dataset[rank::world_size]
        train_loader = DataLoader(
            train_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=train_config['batch_size'], shuffle=False, num_workers=2
        )

        # Initialize model and wrap it with DistributedDataParallel
        # model = self._initialize_model(self.model, 8, 8, width=64)
        model = model.to(local_device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

        # Set up optimizer, criterion, and scheduler
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=train_config['step_size'], gamma=train_config['gamma']
        )

        # Training loop
        best_loss = np.inf
        for epoch in range(train_config['epochs']):
            model.train()
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                batch = batch.to(local_device)
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                loss = criterion(out, batch.y) 
                # wandb.log({'train_loss': loss.item()})
                loss.backward()
                # log gradient during training

                optimizer.step()
                # if rank == 0:
                    # for name, param in model.named_parameters():
                    #     if param.grad is not None:
                    #         wandb.log({f'{name}_grad': param.grad.norm()})
                    # wandb.log({'lr': optimizer.param_groups[0]['lr']})
                train_loss += loss.item()
            train_loss /= len(train_loader)

            if rank == 0:
                print(f'Epoch {epoch}: Train loss: {train_loss}')
                wandb.log({'train_loss': train_loss})
                wandb.log({'lr': optimizer.param_groups[0]['lr']})

            # Validation loop
            if epoch % train_config['val_interval'] == 0:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(local_device)
                        out = model(batch.x, batch.edge_index, batch.edge_attr)
                        # print(out.shape)
                        loss = criterion(out, batch.y)
                        val_loss += loss.item()
                    val_loss /= len(val_loader)
                    if rank == 0:
                        wandb.log({'val_loss': val_loss})
                        plot_data = batch[0]
                        plot_data.pred = out[:plot_data.x.shape[0]]
                        plot_3d_prediction(plot_data, save_mode='wandb')
                        print(f'Epoch {epoch}: Validation loss: {val_loss}')
                    # Save the best model
                    if val_loss < best_loss:
                        best_loss = val_loss
                        os.makedirs(f'logs/models/collection_{name}', exist_ok=True)
                        torch.save(
                            model.module.state_dict(), 
                            f'logs/models/collection_{name}/partition_0.pth'
                        )
                    
                    # Reduce the loss across all processes
                    # if epoch == 0:
                    #     continue
                    
                    dist.all_reduce(torch.tensor(train_loss, device=local_device), op=dist.ReduceOp.AVG)
                    dist.all_reduce(torch.tensor(val_loss, device=local_device), op=dist.ReduceOp.AVG)
                    scheduler.step()

            models.append(model)

        # Cleanup distributed process group
        dist.destroy_process_group()

        # save the models
        if rank == 0:
            torch.save(models[0].module.state_dict(), 'logs/models/collection_{}/partition_{}.pth'.format(name, 0))
        return models
