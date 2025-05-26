import os
import psutil
# os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2' # Only add this for TRACE to work, comment out for other cases! 
import time

from numba import prange, njit
import torch
import numpy as np
import scipy
from scipy.spatial import KDTree
import tqdm
import vtk
from vtk import vtkFLUENTReader, vtkFLUENTCFFReader
from vtkmodules.util import numpy_support  
from vtkmodules.numpy_interface import dataset_adapter as dsa
import pyvista as pv
# import pyamg
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, bicgstab, gmres, cg
import pandas as pd
# import multiprocessing as mp
import torch_geometric as pyg
from torch_geometric.data import Data, InMemoryDataset
# from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
import h5py
import matplotlib.pyplot as plt

from tqdm import tqdm


class GenericGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, partition=False, **kwargs):
        super(GenericGraphDataset, self).__init__(root, transform, pre_transform)
        # self.raw_dir = os.path.join(root, 'raw')
        # self.processed_dir = os.path.join(root, 'processed')
        # check if the raw data & processed data directories are empty
        if len(os.listdir(self.raw_dir)) == 0:
            raise RuntimeError('Raw data directory is empty. Please download the dataset first.')
        if not os.path.exists(self.processed_dir) or len(os.listdir(self.processed_dir)) == 0:
            print('Processing data...')
            self.process()

        data = torch.load(self.processed_paths[0], map_location=torch.device('cpu'), weights_only=False)
        if partition:
            self.sub_size = kwargs['sub_size']
            self.data = self.get_partition_domain(data, 'train')
        else:
            self.data = data

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    # def process(self):
    #     data_list = []
    #     for i in range(1, 3):
    #         data = torch.load('data/processed/data_{}.pt'.format(i))
    #         data_list.append(data)
    #     data, slices = self.collate(data_list)
    #     torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def _download(self):
        pass

    def _process(self):
        pass

    def get_partition_domain(self, data, mode):
        """
        returns a full partitioned collection of subdomains of the original domain
        
        :param data: the original domain stored in a torch_geometric.data.Data object. geometry is stored in data.pos
        """
        # os.makedirs(os.path.join(self.root, 'partition'), exist_ok=True)
        # if os.path.exists(os.path.join(self.root, 'partition', 'data.pt')):
        #     subdomains = torch.load(os.path.join(self.root, 'partition', 'data.pt'))
        # else:
        #     if mode == 'train':
        #         num_processes = mp.cpu_count()
        #         len_single_process = max(len(data) // (num_processes - 1), 1)
        #         data_list = [(data[i * len_single_process:(i + 1) * len_single_process], self.sub_size) for i in range(0, len(data), len_single_process)]
        #         with mp.Pool(num_processes) as pool:
        #             # self.data_test = self._get_partiton_domain(data_list[0])
        #             subdomains = pool.map(self._get_partiton_domain, data_list)
        #         torch.save(subdomains, os.path.join(self.root, 'partition', 'data.pt'))
        # return subdomains
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    @staticmethod
    def _get_partiton_domain(data):
        """
        returns a full partitioned collection of subdomains of the original domain
        
        :param data: the original domain stored in a torch_geometric.data.Data
        :param sub_size: the size of the subdomains
        """
        data_batch, sub_size = data
        subdomains = []
        for data in data_batch:
            data = data[0]
        # get domain geometry bounds
            x_min, x_max = data.pos[:, 0].min(), data.pos[:, 0].max()
            y_min, y_max = data.pos[:, 1].min(), data.pos[:, 1].max()
            z_min, z_max = data.pos[:, 2].min(), data.pos[:, 2].max()
            # temporary fix to the device issue
            # data.edge_index = torch.Tensor(data.edge_index)

            # divide the domain into subdomains according to self.sub_size
            for x in np.arange(x_min, x_max, sub_size):
                for y in np.arange(y_min, y_max, sub_size):
                    for z in np.arange(z_min, z_max, sub_size):
                        # find nodes within the subdomain
                        mask = (data.pos[:, 0] >= x) & (data.pos[:, 0] < x + sub_size) & \
                            (data.pos[:, 1] >= y) & (data.pos[:, 1] < y + sub_size) & \
                            (data.pos[:, 2] >= z) & (data.pos[:, 2] < z + sub_size)
                        subdomain, _ = subgraph(mask, data.edge_index)

                        ########################## TBD: fix boundary information ##########################
                        '''
                        # add boundary information to the subdomain. boundary information is applied as vector on the boundary nodes
                        # indentify boundary nodes
                        boundary_mask = GenericGraphDataset.get_graph_boundary_edges(subdomain)
                        boundary_nodes = subdomain.edge_index[0][boundary_mask].unique()
                        boundary_nodes = torch.cat([boundary_nodes, subdomain.edge_index[1][boundary_mask].unique()])
                        boundary_nodes = boundary_nodes.unique()
                        boundary_nodes = boundary_nodes[boundary_nodes != -1]

                        # add boundary information to the subdomain
                        boundary_info = torch.zeros((boundary_nodes.size(0), 3))
                        # compute boundary vector
                        # get all edges connected to the boundary nodes
                        boundary_edges = subdomain.edge_index[:, boundary_mask]
                        # for every node on the boundary, compute Neumann boundary condition by averaging the 'x' property of the connected nodes
                        for i, node in enumerate(boundary_nodes):
                            connected_nodes = boundary_edges[1][boundary_edges[0] == node]

                            # compute magnitude & direction of the boundary vector
                            boundary_vector = data.pos[node] - data.pos[connected_nodes]
                            boundary_magnitude = data.x[node] - data.x[connected_nodes]
                            # compute Neumann boundary condition
                            boundary_info[i] = boundary_magnitude / boundary_vector.norm()

                        # add boundary information to the subdomain
                        subdomain.bc = boundary_info
                        '''
                        ####################################################################################

                        subdomain = Data(x=data.x[mask], pos=data.pos[mask], edge_index=subdomain)
                        subdomains.append(subdomain)

        return subdomains
    
    def get_graph_boundary_edges(data, dimension=3):
        """
        returns the boundary edges of a graph
        
        :param data: the graph stored in a torch_geometric.data.Data
        :param dimension: the defined geometrical dimension of the graph
        """
        # get adjacency matrix
        adj = pyg.utils.to_dense_adj(data).squeeze()
        # get boundary edges as edgws with only one cell assignments
        boundary_edges = []
        boundary_edges = torch.where(adj.sum(dim=0) == 1)[0]

        return boundary_edges

    def reconstruct_from_partition(self, subdomains):
        """
        reconstructs the original domain from a partitioned collection of subdomains
        
        :param subdomains: a list of subdomains, each stored in a torch_geometric.data.Data object
        """
        # concatenate all subdomains
        data = Data()
        data.x = torch.cat([subdomain.x for subdomain in subdomains], dim=0)
        data.edge_index = torch.cat([subdomain.edge_index for subdomain in subdomains], dim=1)
        data.edge_attr = torch.cat([subdomain.edge_attr for subdomain in subdomains], dim=0)
        data.pos = torch.cat([subdomain.pos for subdomain in subdomains], dim=0)
        data.bc = torch.cat([subdomain.bc for subdomain in subdomains], dim=0)
        return data
    

class DuctAnalysisDataset(GenericGraphDataset):
    def __init__(self, root, transform=None, pre_transform=None, partition=False, **kwargs):
        super(DuctAnalysisDataset, self).__init__(root, transform, pre_transform, partition, **kwargs)
        self.partition = partition
        # self.raw_file_names = [os.path.join(self.raw_dir, f) for f in os.listdir(self.raw_dir) if f.endswith('.mat')]

    def download(self):
        pass

    def len(self):
        # return the number of samples in the dataset
        if not self.partition:
            return len(self._data)
        else:
            # return the number of subdomains in the h5 file
            with h5py.File(os.path.join(self.root, 'partition', 'data.h5'), 'r') as f:
                return len(f.keys())
    
    def get(self, idx):
        if not self.partition:
            return self._data[idx]
        else:
            with h5py.File(os.path.join(self.root, 'partition', 'data.h5'), 'r') as f:
                group = f[f'subdomain_{idx}']
                x = torch.tensor(np.array(group['x']), dtype=torch.float)
                y = torch.tensor(np.array(group['y']), dtype=torch.float)
                pos = torch.tensor(np.array(group['pos']), dtype=torch.float)
                edge_index = torch.tensor(np.array(group['edge_index']), dtype=torch.long)
                edge_attr = torch.tensor(np.array(group['edge_attr']), dtype=torch.float)

                data = Data(x=x, y=y, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
                return data

    @property
    def raw_file_names(self):
        return ["Mesh_Output_High.msh", "Mesh_Output_Med.msh", "Mesh_Output_Low.msh", "Output_Summary_High_100", "Output_Summary_Med_100", "Output_Summary_Low_100", "Output_Summary_High_25", "Output_Summary_Med_25", "Output_Summary_Low_25"]

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        # num_processes = mp.cpu_count()
        # len_single_process = max(len(self.raw_file_names) // (num_processes - 1), 1)
        raw_data_list = [os.path.join(self.raw_dir, f) for f in self.raw_file_names]
        # raw_data_list = [raw_data_list[i:i + len_single_process] for i in range(0, len(raw_data_list), len_single_process)]
        # with mp.Pool(num_processes) as pool:
        #     # data_list_test = CoronaryArteryDataset._process_file(raw_data_list[0])
        #     data_list = pool.map(DuctAnalysisDataset._process_file, raw_data_list)
        # data, slices = self.collate(data_list)
        data_list = self._process_file(raw_data_list)
        torch.save(data_list, self.processed_paths[0])

    @staticmethod
    def extract_unstructured_grid(multi_block):
        """
        Extracts the first vtkUnstructuredGrid from a vtkMultiBlockDataSet.
        """
        for i in range(multi_block.GetNumberOfBlocks()):
            block = multi_block.GetBlock(i)
            # if the block metadata contains interior, return the body mesh block
            if isinstance(block, vtk.vtkUnstructuredGrid):
                if multi_block.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME()) == "interior:interior-fluid":
                    return block
        raise ValueError("No vtkUnstructuredGrid found in the .msh file.")
    
    @staticmethod
    def extract_wall_block(multi_block, target_name="wall:walls"):
        """
        Extracts the 'wall:walls' block from the vtkMultiBlockDataSet.
        Returns the unstructured grid corresponding to wall surfaces.
        """
        for i in range(multi_block.GetNumberOfBlocks()):
            block = multi_block.GetBlock(i)
            name = multi_block.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME()) if multi_block.GetMetaData(i) else None
            
            if isinstance(block, vtk.vtkUnstructuredGrid) and name and target_name in name:
                return block
        raise ValueError(f"No block named '{target_name}' found in the .msh file.")
    
    @staticmethod
    def vtk_to_pyg(data):
        """
        Converts a vtkUnstructuredGrid to a torch_geometric.data.Data object.
        """
        # Step 1: Extract vertex positions (nodes)
        num_points = data.GetNumberOfPoints()
        pos = np.array([data.GetPoint(i) for i in range(num_points)], dtype=np.float32)
        pos = torch.tensor(pos, dtype=torch.float)

        # Step 2: Extract edges from cell connectivity
        edge_set = set()
        # print warning if no cell is found
        if data.GetNumberOfCells() == 0:
            raise ValueError("No valid mesh data found in the given mesh.")
        for i in range(data.GetNumberOfCells()):
            # print warning if no cell is found
            cell = data.GetCell(i)
            num_cell_points = cell.GetNumberOfPoints()

            for j in range(num_cell_points):
                for k in range(j + 1, num_cell_points):
                    edge = (cell.GetPointId(j), cell.GetPointId(k))
                    edge_set.add(edge)
                    edge = (cell.GetPointId(k), cell.GetPointId(j))
                    edge_set.add(edge)

        edge_index = torch.tensor(list(edge_set), dtype=torch.long).t()

        return Data(pos=pos, edge_index=edge_index)

    def _map_physics_data_to_mesh(self, mesh, physics_points):
        """
        Map the physics data to the mesh nodes based on the coordinates using parallel processing.
        """
        num_points = mesh.GetNumberOfPoints()
        mesh_points = np.array([mesh.GetPoint(i) for i in range(num_points)])  # Convert mesh points to NumPy array
        
        tree = KDTree(physics_points)  # Build KDTree for fast lookup
        print("Building KDTree for fast lookup...")
        # Parallelized lookup
        _, nearest_indices = tree.query(mesh_points, workers=16)
        print("Parallelized KDTree lookup complete.")
        return nearest_indices.astype(np.int64)

    def _process_file(self, path_list):
        data_list = []
        # mesh_idx = ['High', 'Med', 'Low']
        # process mesh files
        for idx, path in enumerate(path_list[:2]):
            reader = vtkFLUENTReader()
            reader.SetFileName(path)
            reader.Update()

            # Extract mesh from VTK output
            dataset = reader.GetOutput()

            mesh = self.extract_unstructured_grid(dataset)
            num_points = mesh.GetNumberOfPoints()
            num_cells = mesh.GetNumberOfCells()

            if num_points == 0 or num_cells == 0:
                raise ValueError("No valid mesh data found in the .msh file.")

            try:
                wall_mesh = self.extract_wall_block(dataset)
                wall_indices = set()

                for i in range(wall_mesh.GetNumberOfCells()):
                    cell = wall_mesh.GetCell(i)
                    for j in range(cell.GetNumberOfPoints()):
                        wall_indices.add(cell.GetPointId(j))

                wall_index_tensor = torch.tensor(list(wall_indices), dtype=torch.long)
                print(f"Extracted {len(wall_indices)} wall node indices.")
            except ValueError:
                print("Wall block not found.")
                wall_index_tensor = torch.tensor([], dtype=torch.long)
            
            # process physics files
            # print(path_list[idx+3])
            physics = pd.read_csv(path_list[idx+3], sep=',')
            physics_points = np.vstack((physics['    x-coordinate'], 
                                        physics['    y-coordinate'], 
                                        physics['    z-coordinate'])).T

            # print(physics)
            velocity_x = torch.tensor(physics['      x-velocity'], dtype=torch.float).unsqueeze(1)
            velocity_y = torch.tensor(physics['      y-velocity'], dtype=torch.float).unsqueeze(1)
            velocity_z = torch.tensor(physics['      z-velocity'], dtype=torch.float).unsqueeze(1)
            # velocity = torch.cat([velocity_x, velocity_y, velocity_z], dim=1)
            # normalize the velocity to be in the range of [0, 1]

            pressure = torch.tensor(physics['        pressure'], dtype=torch.float).unsqueeze(1)
            # normalize the pressure
            pressure = pressure / torch.max(pressure)

            physics_map = self._map_physics_data_to_mesh(mesh, physics_points)

            # reorganize the physics data according to the mesh node id
            velocity_x = velocity_x[physics_map]
            velocity_y = velocity_y[physics_map]
            velocity_z = velocity_z[physics_map]
            velocity = torch.cat([velocity_x, velocity_y, velocity_z], dim=1)
            pressure = pressure[physics_map]

            del physics_points, physics_map

            velocity = velocity / torch.max(torch.abs(velocity))
            # create a torch_geometric.data.Data object if the mesh is of high resolution
            if idx == 0:
                mesh_high = mesh
                data = self.vtk_to_pyg(mesh)
                data.y = torch.cat([velocity, pressure], dim=1)
                data.wall_idx = wall_index_tensor
                data_list.append(data)
            else:
                # call lagrangian interpolation to interpolate the physics data to the high resolution mesh
                velocity_x_high = self._lagrangian_interpolation(mesh, velocity_x, mesh_high)
                velocity_y_high = self._lagrangian_interpolation(mesh, velocity_y, mesh_high)
                velocity_z_high = self._lagrangian_interpolation(mesh, velocity_z, mesh_high)
                pressure_high = self._lagrangian_interpolation(mesh, pressure, mesh_high)

                velocity_high = torch.cat([velocity_x_high, velocity_y_high, velocity_z_high], dim=1)
                # normalize the velocity
                velocity_high = velocity_high / torch.max(torch.abs(velocity_high))
                # normalize the pressure
                pressure_high = pressure_high / torch.max(pressure_high)
                # check if nan exists in the interpolated physics data
                if torch.isnan(velocity_high).sum() > 0 or torch.isnan(pressure_high).sum() > 0:
                    print('nan exists in interpolated physics data')

                data_list[0].x = torch.cat([velocity_high, pressure_high], dim=1)

        return data_list
    
    @staticmethod
    def _lagrangian_interpolation(mesh, physics, new_mesh):
        """
        Perform 1st-order Lagrangian interpolation of physics properties 
        at a new set of points based on provided 3D points and physics information.

        Args:
            mesh (vtkUnstructuredGrid): Unstructured mesh loaded from an Ansys .msh file via vtkFLUENTReader.
            physics (np.ndarray): Array of shape (num_points, 1) representing the physics information at the points.
            new_mesh (vtkUnstructuredGrid): Unstructured mesh loaded from an Ansys .msh file for interpolation.

        Returns:
            np.ndarray: Interpolated physics values at the new points of shape (num_new_points, 1).
        """

        # Ensure physics array shape
        physics = np.asarray(physics).flatten()
        num_original_points = mesh.GetNumberOfPoints()
        
        if physics.shape[0] != num_original_points:
            raise ValueError("Mismatch: physics array length must match the number of points in the original mesh.")

        num_new_points = new_mesh.GetNumberOfPoints()
        if num_new_points == 0:
            raise ValueError("New mesh has no points to interpolate.")

        # Step 1: Attach physics data to the original mesh
        physics_array = vtk.vtkFloatArray()
        physics_array.SetName("PhysicsData")
        physics_array.SetNumberOfComponents(1)
        physics_array.SetNumberOfTuples(num_original_points)

        for i in range(num_original_points):
            physics_array.SetValue(i, physics[i])

        mesh.GetPointData().AddArray(physics_array)

        # Step 2: Use vtkProbeFilter for interpolation
        probe_filter = vtk.vtkProbeFilter()
        probe_filter.SetSourceData(mesh)  # Set original mesh as the source
        probe_filter.SetInputData(new_mesh)  # Set new mesh as the target for interpolation
        probe_filter.Update()

        # Step 3: Extract interpolated values
        interpolated_array = probe_filter.GetOutput().GetPointData().GetArray("PhysicsData")

        if interpolated_array is None:
            raise RuntimeError("Interpolation failed: No physics data found in the output.")

        # Convert to NumPy array
        interpolated_values = np.array([interpolated_array.GetValue(i) for i in range(num_new_points)], dtype=np.float32)

        return torch.tensor(interpolated_values.reshape(-1, 1), dtype=torch.float)
    
    def get_partition_domain(self, data, mode):
        """
        returns a full partitioned collection of subdomains of the original domain
        
        :param data: the original domain stored in a torch_geometric.data.Data object. 
        """
        if os.path.exists(os.path.join(self.root, 'partition', 'data.h5')):
            pass
        else:
            os.makedirs(os.path.join(self.root, 'partition'), exist_ok=True)
            reader = vtkFLUENTReader()
            reader.SetFileName(os.path.join(self.raw_dir, self.raw_file_names[0]))
            reader.Update()
            mesh = reader.GetOutput().GetBlock(0)
            data = data[0]
            x, y, pos = data.x, data.y, data.pos
            num_subdomains = self.sub_size
            self._get_partition_domain(mesh, x, y, pos, num_subdomains)
    
    def visualize_partitioned_dataset(self, partitioned_dataset):
        """
        Visualizes a `vtkPartitionedDataSet`, with different colors for each partition.

        Args:
            partitioned_dataset (vtk.vtkPartitionedDataSet): The partitioned dataset from vtkRedistributeDataSetFilter.
        """
        num_partitions = partitioned_dataset.GetNumberOfPartitions()

        # Set up the renderer
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.1, 0.1, 0.1)  # Dark background

        for i in range(num_partitions):
            partition = partitioned_dataset.GetPartition(i)
            if not partition:
                continue

            # Convert partition to PolyData for visualization
            geometry_filter = vtk.vtkGeometryFilter()
            geometry_filter.SetInputData(partition)
            geometry_filter.Update()

            # Assign a unique color
            r, g, b = vtk.vtkMath.Random(0, 1), vtk.vtkMath.Random(0, 1), vtk.vtkMath.Random(0, 1)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(geometry_filter.GetOutput())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(r, g, b)

            renderer.AddActor(actor)

        # Set up the render window
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(800, 600)

        # Set up the interactive window
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)

        # Render and start interaction
        render_window.Render()
        interactor.Start()
    
    def _get_partition_domain(self, mesh, x, y, pos, num_subdomains):
        """
        Perform domain decomposition on a VTK unstructured mesh and associated physics data.
        Uses METIS-based partitioning without explicit graph conversion.

        Args:
            mesh (vtkUnstructuredGrid): The unstructured mesh to be partitioned.
            physics (np.ndarray): Physics properties at mesh nodes, shape (num_points, physics_dim).
            num_subdomains (int): The number of decomposed subdomains.

        Returns:
            subdomain_meshes (list of vtkUnstructuredGrid): Decomposed sub-meshes.
            subdomain_physics (list of np.ndarray): Physics properties for each subdomain.
        """
        # Initialize VTK MPI controller (Required for distributed processing)
        controller = vtk.vtkMultiProcessController.GetGlobalController()
        # Check if MPI is available
        if not controller or controller.GetNumberOfProcesses() <= 1:
            print("Warning: Running in serial mode. No MPI parallelism available.")
            controller = vtk.vtkDummyController()

        # generate global node IDs
        mesh = self.assign_global_node_id(mesh)

        # Set up the distributed data filter
        distributed_filter = vtk.vtkRedistributeDataSetFilter()
        distributed_filter.SetController(controller)

        distributed_filter.SetInputData(mesh)
        distributed_filter.SetPreservePartitionsInOutput(True)

        # set up the number of partitions
        distributed_filter.SetNumberOfPartitions(num_subdomains)
        # progress_observer = vtk.vtkProgressObserver()
        distributed_filter.AddObserver("ProgressEvent", ProgressObserver())

        distributed_filter.SetBoundaryModeToAssignToOneRegion()

        # Ensure the input data is correctly assigned
        distributed_filter.SetInputData(mesh)

        # Execute the partitioning
        distributed_filter.UpdateInformation()
        distributed_filter.Modified()
        distributed_filter.Update()

        # Retrieve partitioned mesh
        partitioned_mesh = distributed_filter.GetOutput()

        # save the partitioned mesh
        writer = vtk.vtkXMLPartitionedDataSetWriter()
        writer.SetFileName(os.path.join(self.root, 'partition', 'partitioned_mesh.vtpd'))
        writer.SetInputData(partitioned_mesh)
        writer.Write()

        # Visualize the partitioned mesh
        # self.visualize_partitioned_dataset(partitioned_mesh)

        controller.Finalize()

        num_partitions = partitioned_mesh.GetNumberOfPartitions()
        print(f"Partitioned mesh into {num_partitions} subdomains.")
        # update self.sub_size
        self.sub_size = num_partitions

        with h5py.File(os.path.join(self.root, 'partition', 'data.h5'), 'w') as f:
            for i in tqdm.tqdm(range(num_partitions), desc="Processing Subdomains"):
                partition = partitioned_mesh.GetPartition(i)
                if not partition:
                    print(f"Warning: Partition {i} is empty. Skipping.")
                    continue

                node_id_array = partition.GetPointData().GetArray("GlobalPointIds")
                if node_id_array is None:
                    print(f"Error: Partition {i} missing GlobalNodeID. Skipping.")
                    continue

                global_node_ids = np.array([node_id_array.GetValue(j) for j in range(node_id_array.GetNumberOfValues())])
                sub_x = x[global_node_ids]
                sub_y = y[global_node_ids]
                sub_pos = pos[global_node_ids]

                subdomain_data = self.vtk_to_pyg(partition)
                # set edge_attr as edge length
                edge_attr = np.linalg.norm(sub_pos[subdomain_data.edge_index[0]] - sub_pos[subdomain_data.edge_index[1]], axis=1)

                group = f.create_group(f'subdomain_{i}')
                group.create_dataset('x', data=sub_x)
                group.create_dataset('y', data=sub_y)
                group.create_dataset('pos', data=sub_pos)
                group.create_dataset('edge_index', data=subdomain_data.edge_index.numpy())
                group.create_dataset('edge_attr', data=edge_attr)

        print("Partitioning complete.")

    # been overwritten by the filter. will get deprecated
    def assign_global_node_id(self, mesh):
        """
        Assigns a global node ID to each point in a VTK unstructured grid.
        """
        num_points = mesh.GetNumberOfPoints()
        global_node_ids = np.arange(num_points)
        global_node_id_array = vtk.vtkIntArray()
        global_node_id_array.SetName("GlobalPointIds")
        global_node_id_array.SetNumberOfComponents(1)
        global_node_id_array.SetNumberOfValues(num_points)

        for i in range(num_points):
            global_node_id_array.SetValue(i, global_node_ids[i])

        mesh.GetPointData().AddArray(global_node_id_array)
        mesh.GetPointData().SetActiveScalars("GlobalPointIds")

        return mesh
    
    def reconstruct_from_partition(self, subdomain_data_list):
        """
        reconstructs the original domain from a partitioned collection of subdomains
        
        :param subdomains: a list of subdomains, each stored in a torch_geometric.data.Data object
        """
        # load the partitioned data
        reader = vtk.vtkXMLPartitionedDataSetReader()
        reader.SetFileName(os.path.join(self.root, 'partition', 'partitioned_mesh.vtpd'))
        reader.Update()

        partitioned_mesh = reader.GetOutput()

        append_filter = vtk.vtkAppendDataSets()
        # append_filter.SetMergePoints(True)
        num_partitions = partitioned_mesh.GetNumberOfPartitions()

        for i in range(num_partitions):
            partition = partitioned_mesh.GetPartition(i)
            velocity_array = vtk.vtkFloatArray()
            pressure_array = vtk.vtkFloatArray()
            velocity_array.SetName("velocity")
            pressure_array.SetName("pressure")
            velocity_array.SetNumberOfComponents(3)
            pressure_array.SetNumberOfComponents(1)

            for row in subdomain_data_list[i][:, :3].numpy():
                velocity_array.InsertNextTuple3(row[0], row[1], row[2])
            for value in subdomain_data_list[i][:, 3].numpy():
                pressure_array.InsertNextTuple1(value)

            partition.GetPointData().AddArray(velocity_array)
            partition.GetPointData().AddArray(pressure_array)

            if not partition:
                print(f"Warning: Partition {i} is empty. Skipping.")
                continue

            # Create a VTK unstructured grid from the subdomain data
            append_filter.AddInputData(partition)

        append_filter.Update()
        
        reconstructed_mesh = append_filter.GetOutput()

        # smooth the velocity field
        # reconstructed_mesh = self.smooth_vtu_with_continuity(reconstructed_mesh)

        return reconstructed_mesh
    
    
    def smooth_vtu_with_continuity(self, vtk_grid, 
                             num_iterations=10, 
                             smoothing_weight=0.02,
                             divergence_weight=0.05,
                             max_correction=0.005):
        """
        Main function to smooth VTU data with continuity constraint using
        parallel direct divergence minimization
        
        Parameters:
        -----------
        vtk_grid : vtkUnstructuredGrid
            Input VTK grid with velocity data
        num_iterations : int
            Number of smoothing iterations
        smoothing_weight : float
            Weight for Laplacian smoothing (0-1)
        divergence_weight : float
            Weight for divergence minimization (0-1)
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        
        Returns:
        --------
        vtkUnstructuredGrid
            Updated VTK grid with smoothed velocity
        """
        try:
            projector = DivergenceFreeProjection(vtk_grid)
            initial_divergence = projector.calculate_divergence()

            print(f"Initial divergence: {initial_divergence}")

            corrected_velocity, corrected_pressure, final_div_norm, iterations = projector.apply_divergence_free_projection(
                max_iterations=20,
                tolerance=1e-2
            )
            
            print(f"Final divergence: {final_div_norm} in {iterations} iterations")

            # Step 3: Update the VTK grid with smoothed velocity data
            updated_vtk_grid = projector.update_vtk_grid(corrected_velocity, corrected_pressure)
            
            print("Successfully smoothed velocity field with parallel direct divergence minimization")
            return updated_vtk_grid
            
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return vtk_grid  # Return original grid on error
    
    def get_one_full_sample(self, idx):
        return self
    

class AnsysDataset(GenericGraphDataset):
    def __init__(self, root, transform=None, pre_transform=None, partition=False, **kwargs):
        super(AnsysDataset, self).__init__(root, transform, pre_transform, partition, **kwargs)
        self.partition = partition
        # self.raw_file_names = [os.path.join(self.raw_dir, f) for f in os.listdir(self.raw_dir) if f.endswith('.mat')]

    def download(self):
        pass

    def len(self):
        # return the number of samples in the dataset
        if not self.partition:
            return len(self._data)
        else:
            total_idx = 0
            # return the number of subdomains in the h5 file
            with h5py.File(os.path.join(self.root, 'partition', 'data.h5'), 'r') as f:
                for key in f.keys():
                    total_idx += len(f[key].keys())
            return total_idx
    
    def get(self, idx):
        if not self.partition:
            return self._data[idx]
        else:
            mesh_idx = idx // (self.len() // 4)
            if mesh_idx >= 4:
                raise IndexError(f"Mesh index {mesh_idx} out of range. Maximum index is 3 for 4 meshes.")
            sub_idx = idx % 4
            with h5py.File(os.path.join(self.root, 'partition', 'data.h5'), 'r') as f:
                # get the group for the mesh index
                mesh_group = f[f'mesh_{mesh_idx}']
                # get the subdomain group for the subdomain index
                if f'subdomain_{sub_idx}' not in mesh_group:
                    raise IndexError(f"Subdomain {sub_idx} does not exist in mesh {mesh_idx}.")
                group = mesh_group[f'subdomain_{sub_idx}']
                x = torch.tensor(np.array(group['x']), dtype=torch.float)
                y = torch.tensor(np.array(group['y']), dtype=torch.float)
                pos = torch.tensor(np.array(group['pos']), dtype=torch.float)
                edge_index = torch.tensor(np.array(group['edge_index']), dtype=torch.long)
                edge_attr = torch.tensor(np.array(group['edge_attr']), dtype=torch.float)

                data = Data(x=x, y=y, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
                return data

    @property
    def raw_file_names(self):
        return ['0degree', '20degree', '40degree', '60degree']

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        # num_processes = mp.cpu_count()
        # len_single_process = max(len(self.raw_file_names) // (num_processes - 1), 1)
        raw_data_list = [os.path.join(self.raw_dir, f) for f in self.raw_file_names]

        data_list = self._process_file(raw_data_list)
        torch.save(data_list, self.processed_paths[0])

    @staticmethod
    def extract_unstructured_grid(multi_block):
        """
        Extracts the first vtkUnstructuredGrid from a vtkMultiBlockDataSet.
        """
        for i in range(multi_block.GetNumberOfBlocks()):
            block = multi_block.GetBlock(i)
            # if the block metadata contains interior, return the body mesh block
            if isinstance(block, vtk.vtkUnstructuredGrid):
                return block
        raise ValueError("No vtkUnstructuredGrid found in the .msh file.")
    
    @staticmethod
    def extract_wall_block(multi_block, target_name="wall:walls"):
        """
        Extracts the 'wall:walls' block from the vtkMultiBlockDataSet.
        Returns the unstructured grid corresponding to wall surfaces.
        """
        for i in range(multi_block.GetNumberOfBlocks()):
            block = multi_block.GetBlock(i)
            name = multi_block.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME()) if multi_block.GetMetaData(i) else None
            
            if isinstance(block, vtk.vtkUnstructuredGrid) and name and target_name in name:
                return block
        raise ValueError(f"No block named '{target_name}' found in the .msh file.")
    
    @staticmethod
    def vtk_to_pyg(data):
        """
        Converts a vtkUnstructuredGrid to a torch_geometric.data.Data object.
        """
        # Step 1: Extract vertex positions (nodes)
        num_points = data.GetNumberOfPoints()
        pos = np.array([data.GetPoint(i) for i in range(num_points)], dtype=np.float32)
        pos = torch.tensor(pos, dtype=torch.float)

        # Step 2: Extract edges from cell connectivity
        edge_set = set()
        # print warning if no cell is found
        if data.GetNumberOfCells() == 0:
            raise ValueError("No valid mesh data found in the given mesh.")
        for i in range(data.GetNumberOfCells()):
            # print warning if no cell is found
            cell = data.GetCell(i)
            num_cell_points = cell.GetNumberOfPoints()

            for j in range(num_cell_points):
                for k in range(j + 1, num_cell_points):
                    edge = (cell.GetPointId(j), cell.GetPointId(k))
                    edge_set.add(edge)
                    edge = (cell.GetPointId(k), cell.GetPointId(j))
                    edge_set.add(edge)

        edge_index = torch.tensor(list(edge_set), dtype=torch.long).t()

        return Data(pos=pos, edge_index=edge_index)

    def _map_physics_data_to_mesh(self, mesh, physics_points):
        """
        Map the physics data to the mesh nodes based on the coordinates using parallel processing.
        """
        num_points = mesh.GetNumberOfPoints()
        mesh_points = np.array([mesh.GetPoint(i) for i in range(num_points)])  # Convert mesh points to NumPy array
        
        tree = KDTree(physics_points)  # Build KDTree for fast lookup
        print("Building KDTree for fast lookup...")
        # Parallelized lookup
        _, nearest_indices = tree.query(mesh_points, workers=16)
        print("Parallelized KDTree lookup complete.")
        return nearest_indices.astype(np.int64)

    def _process_file(self, path_list):
        data_list = []
        # mesh_idx = ['High', 'Med', 'Low']
        # process mesh files
        for idx, path in enumerate(path_list):
            # Check available memory before reading
            print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
            print(f"File size: {os.path.getsize('data/ansys/raw/0degree.cas.h5') / (1024**3):.1f} GB")
            high_res_mesh_path = path + '_high.cas.h5'
            low_res_mesh_path = path + '.cas.h5'
            high_res_physics_path = path + '_high'
            low_res_physics_path = path
            for res_type in ['high', 'low']:
                if not os.path.exists(high_res_mesh_path) or not os.path.exists(low_res_physics_path):
                    print(f"File {high_res_mesh_path} or {high_res_physics_path} does not exist.")
                    continue
                if res_type == 'high':
                    mesh_path = high_res_mesh_path
                    physics_path = high_res_physics_path
                else:
                    mesh_path = low_res_mesh_path
                    physics_path = low_res_physics_path
                reader = vtkFLUENTCFFReader()
                reader.SetFileName(mesh_path)
                reader.Update()
                # Monitor memory usage during read
                process = psutil.Process()
                print(f"Memory usage before read: {process.memory_info().rss / (1024**2):.1f} MB")

                # Extract mesh from VTK output
                dataset = reader.GetOutput()

                mesh = self.extract_unstructured_grid(dataset)
                num_points = mesh.GetNumberOfPoints()
                num_cells = mesh.GetNumberOfCells()

                if num_points == 0 or num_cells == 0:
                    raise ValueError("No valid mesh data found in the .msh file.")

                try:
                    wall_mesh = self.extract_wall_block(dataset)
                    wall_indices = set()

                    for i in range(wall_mesh.GetNumberOfCells()):
                        cell = wall_mesh.GetCell(i)
                        for j in range(cell.GetNumberOfPoints()):
                            wall_indices.add(cell.GetPointId(j))

                    wall_index_tensor = torch.tensor(list(wall_indices), dtype=torch.long)
                    print(f"Extracted {len(wall_indices)} wall node indices.")
                except ValueError:
                    print("Wall block not found.")
                    wall_index_tensor = torch.tensor([], dtype=torch.long)
                
                # process physics files
                # print(path_list[idx+3])
                physics = pd.read_csv(physics_path, sep=',')
                physics_points = np.vstack((physics['    x-coordinate'], 
                                            physics['    y-coordinate'], 
                                            physics['    z-coordinate'])).T

                # print(physics)
                velocity_x = torch.tensor(physics['      x-velocity'], dtype=torch.float).unsqueeze(1)
                velocity_y = torch.tensor(physics['      y-velocity'], dtype=torch.float).unsqueeze(1)
                velocity_z = torch.tensor(physics['      z-velocity'], dtype=torch.float).unsqueeze(1)
                # velocity = torch.cat([velocity_x, velocity_y, velocity_z], dim=1)
                # normalize the velocity to be in the range of [0, 1]

                pressure = torch.tensor(physics['absolute-pressure'], dtype=torch.float).unsqueeze(1)
                # normalize the pressure
                pressure = pressure / torch.max(pressure)

                physics_map = self._map_physics_data_to_mesh(mesh, physics_points)

                # reorganize the physics data according to the mesh node id
                velocity_x = velocity_x[physics_map]
                velocity_y = velocity_y[physics_map]
                velocity_z = velocity_z[physics_map]
                velocity = torch.cat([velocity_x, velocity_y, velocity_z], dim=1)
                pressure = pressure[physics_map]

                del physics_points, physics_map

                velocity = velocity / torch.max(torch.abs(velocity))
                # create a torch_geometric.data.Data object if the mesh is of high resolution
                if res_type == 'high':
                    mesh_high = mesh
                    data = self.vtk_to_pyg(mesh)
                    data.y = torch.cat([velocity, pressure], dim=1)
                    data.wall_idx = wall_index_tensor
                    data_list.append(data)
                else:
                    # call lagrangian interpolation to interpolate the physics data to the high resolution mesh
                    velocity_x_high = self._lagrangian_interpolation(mesh, velocity_x, mesh_high)
                    velocity_y_high = self._lagrangian_interpolation(mesh, velocity_y, mesh_high)
                    velocity_z_high = self._lagrangian_interpolation(mesh, velocity_z, mesh_high)
                    pressure_high = self._lagrangian_interpolation(mesh, pressure, mesh_high)

                    velocity_high = torch.cat([velocity_x_high, velocity_y_high, velocity_z_high], dim=1)
                    # normalize the velocity
                    velocity_high = velocity_high / torch.max(torch.abs(velocity_high))
                    # normalize the pressure
                    pressure_high = pressure_high / torch.max(pressure_high)
                    # check if nan exists in the interpolated physics data
                    if torch.isnan(velocity_high).sum() > 0 or torch.isnan(pressure_high).sum() > 0:
                        print('nan exists in interpolated physics data')

                    data_list[idx].x = torch.cat([velocity_high, pressure_high], dim=1)

        return data_list
    
    @staticmethod
    def _lagrangian_interpolation(mesh, physics, new_mesh):
        """
        Perform 1st-order Lagrangian interpolation of physics properties 
        at a new set of points based on provided 3D points and physics information.

        Args:
            mesh (vtkUnstructuredGrid): Unstructured mesh loaded from an Ansys .msh file via vtkFLUENTReader.
            physics (np.ndarray): Array of shape (num_points, 1) representing the physics information at the points.
            new_mesh (vtkUnstructuredGrid): Unstructured mesh loaded from an Ansys .msh file for interpolation.

        Returns:
            np.ndarray: Interpolated physics values at the new points of shape (num_new_points, 1).
        """

        # Ensure physics array shape
        physics = np.asarray(physics).flatten()
        num_original_points = mesh.GetNumberOfPoints()
        
        if physics.shape[0] != num_original_points:
            raise ValueError("Mismatch: physics array length must match the number of points in the original mesh.")

        num_new_points = new_mesh.GetNumberOfPoints()
        if num_new_points == 0:
            raise ValueError("New mesh has no points to interpolate.")

        # Step 1: Attach physics data to the original mesh
        physics_array = vtk.vtkFloatArray()
        physics_array.SetName("PhysicsData")
        physics_array.SetNumberOfComponents(1)
        physics_array.SetNumberOfTuples(num_original_points)

        for i in range(num_original_points):
            physics_array.SetValue(i, physics[i])

        mesh.GetPointData().AddArray(physics_array)

        # Step 2: Use vtkProbeFilter for interpolation
        probe_filter = vtk.vtkProbeFilter()
        probe_filter.SetSourceData(mesh)  # Set original mesh as the source
        probe_filter.SetInputData(new_mesh)  # Set new mesh as the target for interpolation
        probe_filter.Update()

        # Step 3: Extract interpolated values
        interpolated_array = probe_filter.GetOutput().GetPointData().GetArray("PhysicsData")

        if interpolated_array is None:
            raise RuntimeError("Interpolation failed: No physics data found in the output.")

        # Convert to NumPy array
        interpolated_values = np.array([interpolated_array.GetValue(i) for i in range(num_new_points)], dtype=np.float32)

        return torch.tensor(interpolated_values.reshape(-1, 1), dtype=torch.float)
    
    def get_partition_domain(self, data, mode):
        """
        returns a full partitioned collection of subdomains of the original domain
        
        :param data: the original domain stored in a torch_geometric.data.Data object. 
        """
        if os.path.exists(os.path.join(self.root, 'partition', 'data.h5')):
            pass
        else:
            os.makedirs(os.path.join(self.root, 'partition'), exist_ok=True)
            for i in range(len(self.raw_file_names)):
                reader = vtkFLUENTCFFReader()
                high_res_mesh_path = self.raw_file_names[i] + '_high.cas.h5'
                reader.SetFileName(os.path.join(self.raw_dir, high_res_mesh_path))
                reader.Update()
                mesh = self.extract_unstructured_grid(reader.GetOutput())
                cur_data = data[i]
                x, y, pos = cur_data.x, cur_data.y, cur_data.pos
                num_subdomains = self.sub_size
                self._get_partition_domain(mesh, i, x, y, pos, num_subdomains)

            # concatenate all subdomains from saved separate h5 files into one h5 file
            with h5py.File(os.path.join(self.root, 'partition', 'data.h5'), 'w') as f:
                for i in range(len(self.raw_file_names)):
                    group_current = f.create_group(f'mesh_{i}')
                    with h5py.File(os.path.join(self.root, 'partition', 'data_{}.h5'.format(i)), 'r') as f_sub:
                        for j in f_sub.keys():
                            f_sub.copy(j, group_current)

    
    def visualize_partitioned_dataset(self, partitioned_dataset):
        """
        Visualizes a `vtkPartitionedDataSet`, with different colors for each partition.

        Args:
            partitioned_dataset (vtk.vtkPartitionedDataSet): The partitioned dataset from vtkRedistributeDataSetFilter.
        """
        num_partitions = partitioned_dataset.GetNumberOfPartitions()

        # Set up the renderer
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.1, 0.1, 0.1)  # Dark background

        for i in range(num_partitions):
            partition = partitioned_dataset.GetPartition(i)
            if not partition:
                continue

            # Convert partition to PolyData for visualization
            geometry_filter = vtk.vtkGeometryFilter()
            geometry_filter.SetInputData(partition)
            geometry_filter.Update()

            # Assign a unique color
            r, g, b = vtk.vtkMath.Random(0, 1), vtk.vtkMath.Random(0, 1), vtk.vtkMath.Random(0, 1)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(geometry_filter.GetOutput())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(r, g, b)

            renderer.AddActor(actor)

        # Set up the render window
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(800, 600)

        # Set up the interactive window
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)

        # Render and start interaction
        render_window.Render()
        interactor.Start()
    
    def _get_partition_domain(self, mesh, i, x, y, pos, num_subdomains):
        """
        Perform domain decomposition on a VTK unstructured mesh and associated physics data.
        Uses METIS-based partitioning without explicit graph conversion.

        Args:
            mesh (vtkUnstructuredGrid): The unstructured mesh to be partitioned.
            physics (np.ndarray): Physics properties at mesh nodes, shape (num_points, physics_dim).
            num_subdomains (int): The number of decomposed subdomains.

        Returns:
            subdomain_meshes (list of vtkUnstructuredGrid): Decomposed sub-meshes.
            subdomain_physics (list of np.ndarray): Physics properties for each subdomain.
        """
        # Initialize VTK MPI controller (Required for distributed processing)
        controller = vtk.vtkMultiProcessController.GetGlobalController()
        # Check if MPI is available
        if not controller or controller.GetNumberOfProcesses() <= 1:
            print("Warning: Running in serial mode. No MPI parallelism available.")
            controller = vtk.vtkDummyController()

        # generate global node IDs
        mesh = self.assign_global_node_id(mesh)

        # Set up the distributed data filter
        distributed_filter = vtk.vtkRedistributeDataSetFilter()
        distributed_filter.SetController(controller)

        distributed_filter.SetInputData(mesh)
        distributed_filter.SetPreservePartitionsInOutput(True)

        # set up the number of partitions
        distributed_filter.SetNumberOfPartitions(num_subdomains)
        # progress_observer = vtk.vtkProgressObserver()
        distributed_filter.AddObserver("ProgressEvent", ProgressObserver())

        distributed_filter.SetBoundaryModeToAssignToOneRegion()

        # Ensure the input data is correctly assigned
        distributed_filter.SetInputData(mesh)

        # Execute the partitioning
        distributed_filter.UpdateInformation()
        distributed_filter.Modified()
        distributed_filter.Update()

        # Retrieve partitioned mesh
        partitioned_mesh = distributed_filter.GetOutput()

        # save the partitioned mesh
        writer = vtk.vtkXMLPartitionedDataSetWriter()
        writer.SetFileName(os.path.join(self.root, 'partition', 'partitioned_mesh_{}.vtpd'.format(i)))
        writer.SetInputData(partitioned_mesh)
        writer.Write()

        # Visualize the partitioned mesh
        # self.visualize_partitioned_dataset(partitioned_mesh)

        controller.Finalize()

        num_partitions = partitioned_mesh.GetNumberOfPartitions()
        print(f"Partitioned mesh into {num_partitions} subdomains.")
        # update self.sub_size
        self.sub_size = num_partitions

        with h5py.File(os.path.join(self.root, 'partition', 'data_{}.h5'.format(i)), 'w') as f:
            # create a group for the current mesh index
            for i in tqdm(range(num_partitions), desc="Processing Subdomains"):
                partition = partitioned_mesh.GetPartition(i)
                if not partition:
                    print(f"Warning: Partition {i} is empty. Skipping.")
                    continue

                node_id_array = partition.GetPointData().GetArray("GlobalPointIds")
                if node_id_array is None:
                    print(f"Error: Partition {i} missing GlobalNodeID. Skipping.")
                    continue

                global_node_ids = np.array([node_id_array.GetValue(j) for j in range(node_id_array.GetNumberOfValues())])
                sub_x = x[global_node_ids]
                sub_y = y[global_node_ids]
                sub_pos = pos[global_node_ids]

                subdomain_data = self.vtk_to_pyg(partition)
                # set edge_attr as edge length
                edge_attr = np.linalg.norm(sub_pos[subdomain_data.edge_index[0]] - sub_pos[subdomain_data.edge_index[1]], axis=1)

                group = f.create_group(f'subdomain_{i}')
                group.create_dataset('x', data=sub_x)
                group.create_dataset('y', data=sub_y)
                group.create_dataset('pos', data=sub_pos)
                group.create_dataset('edge_index', data=subdomain_data.edge_index.numpy())
                group.create_dataset('edge_attr', data=edge_attr)

        print("Partitioning complete.")

    # been overwritten by the filter. will get deprecated
    def assign_global_node_id(self, mesh):
        """
        Assigns a global node ID to each point in a VTK unstructured grid.
        """
        num_points = mesh.GetNumberOfPoints()
        global_node_ids = np.arange(num_points)
        global_node_id_array = vtk.vtkIntArray()
        global_node_id_array.SetName("GlobalPointIds")
        global_node_id_array.SetNumberOfComponents(1)
        global_node_id_array.SetNumberOfValues(num_points)

        for i in range(num_points):
            global_node_id_array.SetValue(i, global_node_ids[i])

        mesh.GetPointData().AddArray(global_node_id_array)
        mesh.GetPointData().SetActiveScalars("GlobalPointIds")

        return mesh
    
    def reconstruct_from_partition(self, subdomain_data_list):
        """
        reconstructs the original domain from a partitioned collection of subdomains
        
        :param subdomains: a list of subdomains, each stored in a torch_geometric.data.Data object
        """
        # load the partitioned data
        reader = vtk.vtkXMLPartitionedDataSetReader()
        reader.SetFileName(os.path.join(self.root, 'partition', 'partitioned_mesh.vtpd'))
        reader.Update()

        partitioned_mesh = reader.GetOutput()

        append_filter = vtk.vtkAppendDataSets()
        # append_filter.SetMergePoints(True)
        num_partitions = partitioned_mesh.GetNumberOfPartitions()

        for i in range(num_partitions):
            partition = partitioned_mesh.GetPartition(i)
            velocity_array = vtk.vtkFloatArray()
            pressure_array = vtk.vtkFloatArray()
            velocity_array.SetName("velocity")
            pressure_array.SetName("pressure")
            velocity_array.SetNumberOfComponents(3)
            pressure_array.SetNumberOfComponents(1)

            for row in subdomain_data_list[i][:, :3].numpy():
                velocity_array.InsertNextTuple3(row[0], row[1], row[2])
            for value in subdomain_data_list[i][:, 3].numpy():
                pressure_array.InsertNextTuple1(value)

            partition.GetPointData().AddArray(velocity_array)
            partition.GetPointData().AddArray(pressure_array)

            if not partition:
                print(f"Warning: Partition {i} is empty. Skipping.")
                continue

            # Create a VTK unstructured grid from the subdomain data
            append_filter.AddInputData(partition)

        append_filter.Update()
        
        reconstructed_mesh = append_filter.GetOutput()

        # smooth the velocity field
        # reconstructed_mesh = self.smooth_vtu_with_continuity(reconstructed_mesh)

        return reconstructed_mesh
    
    
    def smooth_vtu_with_continuity(self, vtk_grid, 
                             num_iterations=10, 
                             smoothing_weight=0.02,
                             divergence_weight=0.05,
                             max_correction=0.005):
        """
        Main function to smooth VTU data with continuity constraint using
        parallel direct divergence minimization
        
        Parameters:
        -----------
        vtk_grid : vtkUnstructuredGrid
            Input VTK grid with velocity data
        num_iterations : int
            Number of smoothing iterations
        smoothing_weight : float
            Weight for Laplacian smoothing (0-1)
        divergence_weight : float
            Weight for divergence minimization (0-1)
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        
        Returns:
        --------
        vtkUnstructuredGrid
            Updated VTK grid with smoothed velocity
        """
        try:
            projector = DivergenceFreeProjection(vtk_grid)
            initial_divergence = projector.calculate_divergence()

            print(f"Initial divergence: {initial_divergence}")

            corrected_velocity, corrected_pressure, final_div_norm, iterations = projector.apply_divergence_free_projection(
                max_iterations=20,
                tolerance=1e-2
            )
            
            print(f"Final divergence: {final_div_norm} in {iterations} iterations")

            # Step 3: Update the VTK grid with smoothed velocity data
            updated_vtk_grid = projector.update_vtk_grid(corrected_velocity, corrected_pressure)
            
            print("Successfully smoothed velocity field with parallel direct divergence minimization")
            return updated_vtk_grid
            
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return vtk_grid  # Return original grid on error
    
    def get_one_full_sample(self, idx):
        return self
    

class SubGraphDataset(InMemoryDataset):
    # similar to Sub_jhtdb, creates a subset of the original dataset given the indices
    def __init__(self, root, indices):
        super(SubGraphDataset, self).__init__(None)
        self.indices = indices

        self.data = torch.load(os.path.join(self.root, 'processed', 'data.pt'))
        self.data = [self.data[i] for i in self.indices]


class ProgressObserver:
    """Observer class to track VTK filter progress."""
    def __init__(self):
        self.progress = 0

    def __call__(self, caller, event):
        """Handles progress update events."""
        if isinstance(caller, vtk.vtkRedistributeDataSetFilter):
            self.progress = caller.GetProgress() * 100
            print(f"\rPartitioning Progress: {self.progress:.2f}%", end="", flush=True)


@njit(parallel=True)
def compute_weights(n_points, points, neighbors_flat, offsets):
    max_neighbors = 0
    for i in range(n_points):
        max_neighbors = max(max_neighbors, offsets[i+1] - offsets[i])
    
    weights_matrix = np.zeros((n_points, 3, max_neighbors))
    
    for i in prange(n_points):
        start, end = offsets[i], offsets[i + 1]
        n_neighbors = end - start
        
        if n_neighbors == 0:
            continue
            
        if n_neighbors == 1:
            # For single neighbor, use unit direction
            j = neighbors_flat[start]
            v = points[j] - points[i]
            norm = np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
            if norm > 1e-10:
                weights_matrix[i, :, 0] = v / norm
            continue
            
        # Construct coordinate differences matrix
        A = np.zeros((n_neighbors, 3))
        for idx in range(n_neighbors):
            j = neighbors_flat[start + idx]
            v = points[j] - points[i]
            norm = np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
            if norm > 1e-10:
                A[idx, :] = v / norm  # Normalize vectors
        
        # If we have enough neighbors, try to create a proper basis
        if n_neighbors >= 3:
            # SVD approach for more stability
            try:
                U, S, Vh = np.linalg.svd(A, full_matrices=False)
                
                # Check condition number and handle ill-conditioning
                max_S = S[0]
                min_S = S[-1] if S[-1] > 0 else S[-2] if len(S) > 1 and S[-2] > 0 else 1e-10
                cond_number = max_S / min_S
                
                if cond_number < 1e8:  # Reasonable condition number
                    # Use truncated SVD with threshold
                    S_inv = np.zeros_like(S)
                    threshold = max_S * 1e-6
                    for s_idx in range(len(S)):
                        if S[s_idx] > threshold:
                            S_inv[s_idx] = 1.0 / S[s_idx]
                    
                    # Compute pseudoinverse
                    pinv = Vh.T @ np.diag(S_inv) @ U.T
                    
                    # Extract weights
                    for idx in range(n_neighbors):
                        weights_matrix[i, :, idx] = pinv[:, idx]
                        
                    # Check if weights are too large (indication of instability)
                    w_norm = 0.0
                    for idx in range(n_neighbors):
                        for d in range(3):
                            w_norm += weights_matrix[i, d, idx]**2
                    
                    # If weights are too large, revert to simpler method
                    if w_norm > 100.0:
                        for idx in range(n_neighbors):
                            weights_matrix[i, :, idx] = 1.0 / n_neighbors * A[idx, :]
                else:
                    # Fall back to simple averaging for ill-conditioned systems
                    for idx in range(n_neighbors):
                        weights_matrix[i, :, idx] = 1.0 / n_neighbors * A[idx, :]
            except:
                # Any numerical errors, fall back to simple weights
                for idx in range(n_neighbors):
                    weights_matrix[i, :, idx] = 1.0 / n_neighbors * A[idx, :]
        else:
            # For 1 or 2 neighbors, use normalized vectors directly
            for idx in range(n_neighbors):
                weights_matrix[i, :, idx] = 1.0 / n_neighbors * A[idx, :]
    
    return weights_matrix


@njit(parallel=True)
def compute_divergence(n_points, velocity, neighbors_flat, offsets, weights_matrix):
    divergence = np.zeros(n_points)
    for i in prange(n_points):
        start, end = offsets[i], offsets[i + 1]
        n_neighbors = end - start
        vel_diffs = np.zeros((n_neighbors, 3))
        for idx in range(n_neighbors):
            j = neighbors_flat[start + idx]
            vel_diffs[idx, :] = velocity[j, :] - velocity[i, :]
        # vel_diffs_T = np.ascontiguousarray(vel_diffs.T)
        # print(f"vel_diffs shape: {vel_diffs.shape}")
        # print(f"weights_matrix shape: {weights_matrix[i, :, :n_neighbors].shape}")
        divergence[i] = np.sum(weights_matrix[i, :, :n_neighbors] @ vel_diffs)
    return divergence

# @njit(parallel=True)
def solve_pressure_poisson_adaptive(laplacian_matrix, divergence, max_iterations=1000, initial_omega=0.05):
    n = divergence.shape[0]
    pressure = np.zeros(n, dtype=np.float64)
    residual = -divergence.copy()  # Initial residual
    
    # Extract diagonal and off-diagonal parts
    diag = np.zeros(n)
    for i in range(n):
        diag[i] = laplacian_matrix[i, i]
    
    # Normalize divergence for better convergence check
    div_norm = np.linalg.norm(divergence)
    if div_norm < 1e-5:
        return pressure  # No divergence to correct
    
    # Adaptive relaxation
    omega = initial_omega
    prev_res_norm = np.linalg.norm(residual)
    
    for iter in range(max_iterations):
        # Compute new pressure based on current residual
        delta_p = np.zeros(n)
        for i in prange(n):
            if diag[i] > 1e-10:
                delta_p[i] = omega * residual[i] / diag[i]
        
        # Update pressure
        pressure += delta_p
        
        # Recompute residual: r = b - Ax
        residual = -divergence - laplacian_matrix @ pressure
        
        # Check convergence
        res_norm = np.linalg.norm(residual)
        
        # Adaptive omega adjustment
        if iter > 0 and iter % 10 == 0:
            if res_norm < prev_res_norm:
                # Converging, slightly increase omega
                omega = min(omega * 1.05, 0.9)
            else:
                # Diverging, decrease omega
                omega = max(omega * 0.5, 0.001)
        
        prev_res_norm = res_norm
        print(f"Iteration {iter}: Residual Norm = {res_norm}, Omega = {omega}")
        
        # Convergence check
        if res_norm < 1e-4 * div_norm:
            break
    
    return pressure

@njit(parallel=True)
def apply_pressure_correction(n_points, velocity, pressure, neighbors_flat, offsets, weights_matrix):
    corrected_velocity = velocity.copy()
    for i in prange(n_points):
        start, end = offsets[i], offsets[i + 1]
        n_neighbors = end - start
        pressure_diffs = np.zeros(n_neighbors)
        for idx in range(n_neighbors):
            j = neighbors_flat[start + idx]
            pressure_diffs[idx] = pressure[j] - pressure[i]
        grad_p = weights_matrix[i, :, :n_neighbors] @ pressure_diffs
        corrected_velocity[i, :] -= grad_p
    return corrected_velocity


@njit(parallel=True)
def assemble_laplacian(n_points, neighbors_flat, offsets, weights_matrix):
    # Estimate number of non-zero entries (diagonal + off-diagonal)
    nnz_estimate = n_points + neighbors_flat.shape[0]
    row_indices = np.zeros(nnz_estimate, dtype=np.int64)
    col_indices = np.zeros(nnz_estimate, dtype=np.int64)
    data = np.zeros(nnz_estimate, dtype=np.float64)
    
    idx = 0
    
    # Process each point
    for i in prange(n_points):
        start, end = offsets[i], offsets[i + 1]
        n_neighbors = end - start
        
        if n_neighbors == 0:  # Isolated point
            row_indices[idx] = i
            col_indices[idx] = i
            data[idx] = 1.0  # Identity for isolated points
            idx += 1
            continue
        
        # Sum for normalization
        total_weight = 0.0
        weights = np.zeros(n_neighbors, dtype=np.float64)
        
        # First pass: compute weights and their sum
        for k in range(n_neighbors):
            j = neighbors_flat[start + k]
            weight_vec = weights_matrix[i, :, k]
            # Use norm of weight vector (more stable than simply summing components)
            weight = np.sqrt(np.sum(weight_vec**2))
            weights[k] = weight
            total_weight += weight
        
        # Skip if total weight is too small
        if total_weight < 1e-10:
            row_indices[idx] = i
            col_indices[idx] = i
            data[idx] = 1.0  # Identity for degenerate cases
            idx += 1
            continue
        
        # Normalize weights to improve conditioning
        scale_factor = 1.0
        if total_weight > 0:
            scale_factor = 1.0 / total_weight
        
        # Second pass: add off-diagonal entries (negative weights)
        diag_sum = 0.0
        for k in range(n_neighbors):
            j = neighbors_flat[start + k]
            if i != j:  # Off-diagonal
                weight = weights[k] * scale_factor
                row_indices[idx] = i
                col_indices[idx] = j
                data[idx] = -weight  # Negative for off-diagonal
                idx += 1
                diag_sum += weight
        
        # Add diagonal entry (sum of off-diagonal weights)
        row_indices[idx] = i
        col_indices[idx] = i
        data[idx] = diag_sum  # Diagonal is sum of off-diagonal
        idx += 1
    
    # Trim arrays to actual size used
    return row_indices[:idx], col_indices[:idx], data[:idx]
    

class DivergenceFreeProjection:
    def __init__(self, vtk_grid, velocity_array_name="velocity", pressure_array_name="pressure"):
        self.vtk_grid = vtk_grid
        self.velocity_array_name = velocity_array_name
        self.pressure_array_name = pressure_array_name

        self.grid = dsa.WrapDataObject(vtk_grid)
        self.mesh = pv.wrap(vtk_grid)
        self.points = self.grid.Points
        self.n_points = self.points.shape[0]
        self.cells = self._extract_cells()

        self.velocity = self.grid.PointData[velocity_array_name]
        self.pressure = self.grid.PointData[pressure_array_name]

        self.point_neighbors, self.offset = self.build_connectivity(self.n_points, self.cells)
        self.weights_matrix = compute_weights(self.n_points, self.points, self.point_neighbors, self.offset)

    @staticmethod
    def build_connectivity(n_points, cells):
        from collections import defaultdict
        neighbor_sets = defaultdict(set)

        for cell in cells:
            for i in cell:
                neighbor_sets[i].update([j for j in cell if j != i])

        neighbor_counts = np.array([len(neighbor_sets[i]) for i in range(n_points)], dtype=np.int64)
        offsets = np.zeros(n_points + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(neighbor_counts)
        neighbors_flat = np.empty(offsets[-1], dtype=np.int64)

        isolated_points = []
        for i in range(n_points):
            neighbors = list(neighbor_sets[i])
            if len(neighbors) == 0:
                isolated_points.append(i)
            neighbors_flat[offsets[i]:offsets[i + 1]] = neighbors

        # print("=== Build Connectivity Debug ===")
        # print(f"Total Points: {n_points}")
        # print(f"Total Cells: {len(cells)}")
        # print(f"Total Neighbors Recorded: {neighbors_flat.shape[0]}")
        # print(f"Number of Isolated Points: {len(isolated_points)}")
        # if len(isolated_points) > 0:
        #     print(f"Isolated Point Indices: {isolated_points[:50]}{'...' if len(isolated_points) > 50 else ''}")
        # print("=================================")
        return neighbors_flat, offsets

    def _extract_cells(self):
        cells = []
        print("Total number of cells:", self.vtk_grid.GetNumberOfCells())
        print("Total number of points:", self.vtk_grid.GetNumberOfPoints())
        for i in range(self.vtk_grid.GetNumberOfCells()):
            cell = self.vtk_grid.GetCell(i)
            cell_points = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
            cells.append(np.array(cell_points, dtype=np.int64))

        # print("Number of unique points in cells:", np.unique(np.concatenate(cells)).shape[0])
        return cells

    def calculate_divergence(self):
        return compute_divergence(self.n_points, self.velocity, self.point_neighbors, self.offset, self.weights_matrix)

    def build_laplacian_matrix(self):
        # First analyze some sample weights
        # print("=== Weight Matrix Analysis ===")
        # sample_points = np.random.choice(self.n_points, size=5, replace=False)
        # for i in sample_points:
        #     start, end = self.offset[i], self.offset[i + 1]
        #     n_neighbors = end - start
        #     print(f"Point {i}: {n_neighbors} neighbors")
            
        #     if n_neighbors > 0:
        #         weights_sum = 0
        #         for k in range(n_neighbors):
        #             weight_vec = self.weights_matrix[i, :, k]
        #             weight_sum = np.sum(weight_vec)
        #             weight_norm = np.linalg.norm(weight_vec)
        #             weights_sum += weight_sum
        #             print(f"  Neighbor {k}: sum={weight_sum:.6f}, norm={weight_norm:.6f}, vec={weight_vec}")
                
        #         print(f"  Total weights sum: {weights_sum:.6f}")
        rows, cols, data = assemble_laplacian(self.n_points, self.point_neighbors, self.offset, self.weights_matrix)
        
        # # Check for duplicate entries again
        # from collections import defaultdict
        # dupes = defaultdict(int)
        # for i in range(len(rows)):
        #     dupes[(rows[i], cols[i])] += 1
        
        # multi_entries = {k: v for k, v in dupes.items() if v > 1}
        # print(f"Duplicate entries after modified assembly: {len(multi_entries)}")
        
        laplacian = csr_matrix((data, (rows, cols)), shape=(self.n_points, self.n_points))
        
        # # Check matrix properties
        # row_sums = np.abs(laplacian).sum(axis=1).A.flatten()
        # zero_rows = np.sum(row_sums < 1e-12)
        # print(f"Zero rows in final matrix: {zero_rows}")
        
        return laplacian

    def solve_pressure_poisson(self, divergence, tol=1e-5, maxiter=1000):
        laplacian = self.build_laplacian_matrix()
    
        # Check matrix properties
        print("=== Solver Debug ===")
        diag_max = np.max(laplacian.diagonal())
        diag_min = np.min(laplacian.diagonal())
        print(f"Diagonal range: min={diag_min}, max={diag_max}, ratio={diag_max/diag_min}")
        
        # Try algebraic multigrid with stronger settings
        try:
            print("Using AMG solver with V-cycle...")
            ml = pyamg.smoothed_aggregation_solver(
                laplacian, 
                max_levels=40,  # Allow more levels
                max_coarse=50,  # Smaller coarse problem
                strength=('evolution', {'epsilon': 4.0}),  # More aggressive coarsening
                smooth=('energy', {'weighting': 'local'}),  # Better smoother
                presmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 2}),
                postsmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 2})
            )
            print(f"AMG levels: {len(ml.levels)}")
            M = ml.aspreconditioner(cycle='W')
            
            # Increase maxiter and loosen tolerance if needed
            pressure, info = cg(laplacian, -divergence, rtol=tol, maxiter=maxiter, M=M)
            
            # if info > 0:  # Not converged, but didn't fail with error
            #     print(f"CG did not converge in {maxiter} iterations. Trying with W-cycle...")
            #     # Try W-cycle which can be more effective but costlier
            #     M = ml.aspreconditioner(cycle='W')
            #     pressure, info = cg(laplacian, -divergence, rtol=tol*10, maxiter=maxiter, M=M)
                
            # if info > 0:  # Still not converged
            #     print(f"AMG+CG still not converging. Trying GMRES with AMG...")
            #     pressure, info = gmres(laplacian, -divergence, rtol=tol*10, maxiter=min(maxiter, 1000), M=M)
                
            if info != 0:
                print(f"Iterative solvers didn't converge. Trying direct solver...")
                # For moderate-sized problems, try a direct solve
                if self.n_points < 100000:
                    pressure = spsolve(laplacian, -divergence)
                    info = 0  # Direct solve doesn't return info
                else:
                    # For larger problems, try relaxation method
                    print("Problem too large for direct solver. Using relaxation method...")
                    pressure = solve_pressure_poisson_adaptive(laplacian, -divergence)
                        
        except Exception as e:
            print(f"Solver error: {str(e)}")
            raise
        
        print("=================================")
        return pressure

    def apply_pressure_gradient(self, pressure):
        return apply_pressure_correction(self.n_points, self.velocity, pressure, self.point_neighbors, self.weights_matrix)

    def update_vtk_grid(self, corrected_velocity, corrected_pressure):
        pd = self.vtk_grid.GetPointData()
        for name, arr in [(self.velocity_array_name, corrected_velocity), (self.pressure_array_name, corrected_pressure)]:
            vtk_arr = numpy_support.numpy_to_vtk(arr, deep=1)
            vtk_arr.SetName(name)
            if pd.HasArray(name):
                pd.RemoveArray(name)
            pd.AddArray(vtk_arr)
        return self.vtk_grid

    def apply_divergence_free_projection(self, max_iterations=10, tolerance=1e-1):
        """
        Apply divergence-free projection with stability controls and fallback strategies.
        """
        print("Starting divergence-free projection with stability controls...")
        t_start = time.time()
        
        # Original velocity
        original_velocity = self.velocity.copy()
        current_velocity = original_velocity.copy()
        best_velocity = original_velocity.copy()
        best_pressure = np.zeros(self.n_points)
        
        # Compute initial divergence
        divergence = self.calculate_divergence()
        initial_norm = np.linalg.norm(divergence)
        best_div_norm = initial_norm
        divergence_history = [initial_norm]
        
        print(f"Initial divergence norm: {initial_norm:.6e}")
        
        # For really small divergence, skip the process
        if initial_norm < tolerance:
            print(f"Initial divergence {initial_norm:.6e} already below tolerance {tolerance:.6e}")
            return original_velocity, np.zeros(self.n_points), initial_norm, 0
        
        # Relaxation factor for incremental projection
        alpha = 0.2  # Start with conservative value
        
        for iteration in range(max_iterations):
            print(f"Iteration {iteration + 1}")
            
            # Solve pressure Poisson equation
            pressure = self.solve_pressure_poisson(divergence, tol=1e-5*initial_norm)
            
            # Check pressure solution
            pressure_norm = np.linalg.norm(pressure)
            if pressure_norm > 1e3 * initial_norm:
                print(f"Warning: Pressure solution appears unstable (norm={pressure_norm:.6e})")
                # Cap the pressure values
                scale = 1e3 * initial_norm / pressure_norm
                pressure *= scale
                print(f"Scaling pressure by {scale:.6e}")
            
            # Apply pressure correction with relaxation
            next_velocity = apply_pressure_correction(
                current_velocity, pressure, 
                self.point_neighbors, self.offset, self.weights_matrix,
                alpha=alpha  # Use relaxation factor
            )
            
            # Store previous velocity for possible rollback
            prev_velocity = current_velocity.copy()
            current_velocity = next_velocity
            
            # Update velocity and recompute divergence
            self.velocity = current_velocity
            divergence = self.calculate_divergence()
            current_norm = np.linalg.norm(divergence)
            divergence_history.append(current_norm)
            
            print(f"Divergence norm: {current_norm:.6e} (relative: {current_norm/initial_norm:.6e})")
            
            # Check if we improved
            if current_norm < best_div_norm:
                best_div_norm = current_norm
                best_velocity = current_velocity.copy()
                best_pressure = pressure.copy()
                
                # If we're making good progress, increase alpha slightly
                if current_norm < 0.7 * divergence_history[-2]:
                    alpha = min(alpha * 1.2, 0.5)
                    print(f"Good progress, increasing alpha to {alpha:.4f}")
                    
            else:
                # We got worse - roll back and decrease alpha
                current_velocity = prev_velocity.copy()
                self.velocity = current_velocity
                
                # Recompute divergence to ensure it's consistent
                divergence = self.calculate_divergence()
                current_norm = np.linalg.norm(divergence)
                
                # Replace last history entry
                divergence_history[-1] = current_norm
                
                # Decrease alpha
                alpha = max(alpha * 0.5, 0.05)
                print(f"Projection diverged, rolling back. New alpha: {alpha:.4f}")
                
                # If we've decreased alpha multiple times with no improvement, break
                if alpha < 0.06 and iteration > 2:
                    print("Alpha below threshold with no improvement. Stopping early.")
                    break
            
            # Check for early convergence
            if current_norm <= tolerance * initial_norm:
                print(f"Converged after {iteration + 1} iterations.")
                break
        
        # Always use best result
        self.velocity = best_velocity
        final_div_norm = np.linalg.norm(self.calculate_divergence())
        
        print(f"Using best projection result with divergence norm: {final_div_norm:.6e}")
        print(f"Projection completed in {time.time() - t_start:.2f} seconds.")
        print(f"Relative improvement: {initial_norm/final_div_norm:.2f}x")
        
        # If we didn't improve at all, warn the user
        if final_div_norm >= initial_norm:
            print(f"WARNING: Projection failed to reduce divergence. Original: {initial_norm:.6e}, Final: {final_div_norm:.6e}")
            # As a last resort, use a very small correction
            self.velocity = original_velocity * 0.98 + best_velocity * 0.02
            final_div_norm = np.linalg.norm(self.calculate_divergence())
            
            if final_div_norm >= initial_norm:
                # If even that didn't work, revert completely
                self.velocity = original_velocity
                final_div_norm = initial_norm
                print("Completely reverting to original velocity field.")
            
        return self.velocity, best_pressure, final_div_norm, max_iterations

    
    def plot_divergence_reduction(self, divergence_history):
        plt.figure(figsize=(10, 6))
        plt.plot(divergence_history, marker='o')
        plt.yscale('log')
        plt.title("Divergence Reduction Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Divergence Norm (log scale)")
        plt.grid()
        plt.show()