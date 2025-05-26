import os
import argparse
import time
from models.model import *
# from models.scheduler import *
# from deepxde.nn.pytorch import DeepONet
from dataset.GraphDataset import *
# from torch_geometric.data import Data
from torch_geometric.nn import GraphSAGE
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import wandb
import vtk


def load_yaml(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_cur_time():
    return time.strftime('%m-%d-%H-%M', time.localtime())


def init_model(type, in_channels, out_channels, **kwargs):
    if type == 'fno':
        return FNO2d(in_channels, out_channels, **kwargs)
    elif type == 'teecnet':
        return TEECNet(in_channels, out_channels=out_channels, **kwargs)
    elif type == 'deeponet':
        # return DeepONet(in_channels, kwargs['trunk_size'], activation=kwargs['activation'], \
        #                 kernel_initializer=kwargs['kernel_initializer'], num_outputs=out_channels)
        return DeepONet(in_channels, kwargs['trunk_size'], hidden_dim=kwargs['width'], output_dim=out_channels)
    elif type == 'graphsage':
        return GraphSAGE(in_channels, out_channels, num_layers=5)
    elif type == 'neuralop':
        return KernelNN(width=kwargs['width'], ker_width=kwargs['width'], depth=kwargs['num_layers'], in_width=in_channels, out_width=out_channels)
    else:
        raise ValueError(f'Invalid model type: {type}')
    

def init_dataset(name, **kwargs):
    if name == 'duct':
        return DuctAnalysisDataset(**kwargs)
    elif name == 'ansys':
        return AnsysDataset(**kwargs)
    else:
        raise ValueError(f'Invalid dataset name: {name}')


def parse_args():
    parser = argparse.ArgumentParser(description='Run ALDS experiment')
    parser.add_argument('--dataset', type=str, default='ansys', help='Name of the dataset')
    parser.add_argument('--encoder', type=str, default='pca', help='Name of the encoder')
    parser.add_argument('--classifier', type=str, default='kmeans', help='Name of the classifier')
    parser.add_argument('--model', type=str, default='teecnet', help='Name of the model')
    parser.add_argument('--exp_name', type=str, default='ansys_teecnet', help='Name of the experiment')
    parser.add_argument('--mode', type=str, default='train', help='Mode of the experiment')
    parser.add_argument('--exp_config', type=str, default='configs/exp_config/teecnet_ansys.yaml', help='Path to the experiment configuration file')
    parser.add_argument('--train_config', type=str, default='configs/train_config/teecnet.yaml', help='Path to the training configuration file')
    args = parser.parse_args()
    return args


def save_pyg_to_vtk(data, mesh_path, save_path):
    # save the prediction data to vtk file
    # data: pytorch geometric data object
    # mesh_path: path to the original mesh data
    # save_path: path to save the vtk file
    reader = vtk.vtkFLUENTReader()
    reader.SetFileName(mesh_path)
    reader.Update()
    mesh = reader.GetOutput().GetBlock(0)

    # create a new vtk unstructured grid
    grid = vtk.vtkUnstructuredGrid()
    grid.DeepCopy(mesh)

    # add the prediction data to the grid
    pred = data.pred.cpu().detach().numpy()
    pred = np.expand_dims(pred, axis=1)
    pred = np.concatenate([pred, pred, pred], axis=1)
    pred = pred.flatten()
    pred = np.ascontiguousarray(pred, dtype=np.float64)
    vtk_pred = vtk.vtkDoubleArray()
    vtk_pred.SetNumberOfComponents(3)
    vtk_pred.SetNumberOfTuples(len(pred) // 3)
    vtk_pred.SetArray(pred, len(pred), 1)
    vtk_pred.SetName('prediction')
    grid.GetPointData().AddArray(vtk_pred)

    # write the grid to vtk file
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(save_path)
    writer.SetInputData(grid)
    writer.Write()



def plot_3d_prediction(y_pred, save_mode='wandb', **kwargs):
    position = y_pred.pos.cpu().detach().numpy()
    # projection 3d
    fig = plt.figure(figsize=(20, 5))
    ax0 = fig.add_subplot(131, projection='3d')
    ax0.scatter(position[:, 0], position[:, 1], position[:, 2], c=torch.norm(y_pred.x[:, :1], dim=1).cpu().detach().numpy(), cmap='plasma')
    # ax0.quiver(position[:, 0], position[:, 1], position[:, 2], y_pred.x[:, 0].cpu().detach().numpy(), y_pred.x[:, 1].cpu().detach().numpy(), y_pred.x[:, 2].cpu().detach().numpy(), length=torch.norm(y_pred.x[:, :3], dim=1).cpu().detach().numpy(), normalize=True)
    ax0.set_title('Input')
    ax0.axis('off')
    plt.colorbar(ax0.collections[0], ax=ax0, orientation='vertical')

    ax1 = fig.add_subplot(132, projection='3d')
    ax1.scatter(position[:, 0], position[:, 1], position[:, 2], c=torch.norm(y_pred.y[:, :1], dim=1).cpu().detach().numpy(), cmap='plasma')
    # ax1.quiver(position[:, 0], position[:, 1], position[:, 2], y_pred.y[:, 0].cpu().detach().numpy(), y_pred.y[:, 1].cpu().detach().numpy(), y_pred.y[:, 2].cpu().detach().numpy(), length=torch.norm(y_pred.y[:, :3], dim=1).cpu().detach().numpy(), normalize=True)
    ax1.set_title('Ground truth')
    ax1.axis('off')
    plt.colorbar(ax1.collections[0], ax=ax1, orientation='vertical')

    ax2 = fig.add_subplot(133, projection='3d')
    ax2.scatter(position[:, 0], position[:, 1], position[:, 2], c=torch.norm(y_pred.pred[:, :1], dim=1).cpu().detach().numpy(), cmap='plasma')
    # ax2.quiver(position[:, 0], position[:, 1], position[:, 2], y_pred.pred[:, 0].cpu().detach().numpy(), y_pred.pred[:, 1].cpu().detach().numpy(), y_pred.pred[:, 2].cpu().detach().numpy(), length=torch.norm(y_pred.pred[:, :3], dim=1).cpu().detach().numpy(), normalize=True)
    ax2.set_title('Prediction')
    ax2.axis('off')
    plt.colorbar(ax2.collections[0], ax=ax2, orientation='vertical')

    # ax2 = fig.add_subplot(133, projection='3d')
    # ax2.scatter(position[:, 0], position[:, 1], position[:, 2], c=np.abs(torch.norm(y_pred.x, dim=1).cpu().detach().numpy() - torch.norm(y_pred.y, dim=1).cpu().detach().numpy()), cmap='plasma')
    # ax2.quiver(position[:, 0], position[:, 1], position[:, 2], y_pred.x[:, 0].cpu().detach().numpy() - y_pred.y[:, 0].cpu().detach().numpy(), y_pred.x[:, 1].cpu().detach().numpy() - y_pred.y[:, 1].cpu().detach().numpy(), y_pred.x[:, 2].cpu().detach().numpy() - y_pred.y[:, 2].cpu().detach().numpy(), length=0.1, normalize=True)
    # ax2.set_title('Absolute difference')

    if save_mode == 'wandb':
        wandb.log({'prediction': wandb.Image(plt)})
    elif save_mode == 'plt':
        plt.show()
    elif save_mode == 'save':
        os.makedirs(os.path.dirname(kwargs['path']), exist_ok=True)
        plt.savefig(kwargs['path'] +'.pdf', format='pdf', dpi=300)
    elif save_mode == 'save_png':
        os.makedirs(os.path.dirname(kwargs['path']), exist_ok=True)
        plt.savefig(kwargs['path'] +'.png', format='png', dpi=300)

