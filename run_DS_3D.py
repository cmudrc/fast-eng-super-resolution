from models.scheduler_gnn import GNNPartitionScheduler
from utils import *

import os
from sklearn.metrics import r2_score

import time


def train_graph_ALDD(exp_name,  model, dataset, num_partitions, train_config, **kwargs):
    scheduler = GNNPartitionScheduler(exp_name, num_partitions, dataset, model, train=True)
    scheduler.train(train_config, **kwargs)

def pred_graph_ALDD(idxs, exp_name, model, dataset, num_partitions, save_mode, **kwargs):
    scheduler = GNNPartitionScheduler(exp_name, num_partitions, dataset, model, train=False)
    for idx in idxs:
        x = dataset.get_one_full_sample(idx)

        time_start = time.time()
        pred_y_list, ref_y_list = scheduler.predict(x)
        time_end = time.time()

        print(f'Prediction time: {time_end - time_start}')
        
        time_start = time.time()
        pred_y = dataset.reconstruct_from_partition(pred_y_list, ref_y_list, idx)
        time_end = time.time()

        print(f'Reconstruction time: {time_end - time_start}')
        # pred_y.pred = pred_y.x

        # save the prediction
        os.makedirs(f'logs/vtk/{exp_name}', exist_ok=True)
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(f'logs/vtk/{exp_name}/pred_{idx}.vtu')
        writer.SetInputData(pred_y)
        writer.Update()
        writer.Write()

        # torch.save(sub_y, f'logs/raw_data/{exp_name}/gt_timestep_{idx}.pth')
        print(f'Prediction done!')


if __name__ == '__main__':
    # dataset = CoronaryArteryDataset(root='data/coronary', partition=True, sub_size=5)
    # dataset = DuctAnalysisDataset(root='data/Duct', partition=True, sub_size=0.03)
    # MPI.Init()
    args = parse_args()
    run_mode = args.mode
    encoder_name = args.encoder
    classifier_name = args.classifier
    model_name = args.model
    exp_name = args.exp_name
    dataset_name = args.dataset
    exp_config = args.exp_config
    train_config = args.train_config

    exp_config = load_yaml(exp_config)
    train_config = load_yaml(train_config)

    n_clusters = exp_config['n_clusters']

    model = init_model(model_name, **exp_config)
    dataset = init_dataset(dataset_name, **exp_config)
    print('Dataset loaded!')

    if run_mode == 'train':
        train_graph_ALDD(exp_name, model, dataset, n_clusters, train_config)

    elif run_mode == 'pred':
        pred_graph_ALDD(exp_config['idxs'], exp_name, model, dataset, n_clusters, 'save_png', sub_size=0.03)
        # pred_graph_ALDD([0, 1, 2, 3], exp_name, encoder, classifier, model, dataset, n_clusters, 'local', sub_size=0.03)