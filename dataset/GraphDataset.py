import os
# os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2' # Only add this for TRACE to work, comment out for other cases! 
import time

from numba import prange, njit
from fenics import *
from dolfin import *
import torch
import numpy as np
from scipy.spatial import KDTree
import tqdm
import vtk
from vtk import vtkFLUENTReader
from vtkmodules.util import numpy_support  
from vtkmodules.numpy_interface import dataset_adapter as dsa
import pyvista
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, bicgstab, gmres
import pandas as pd
import multiprocessing as mp
import torch_geometric as pyg
from torch_geometric.data import Data, InMemoryDataset
# from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
import h5py

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
        os.makedirs(os.path.join(self.root, 'partition'), exist_ok=True)
        if os.path.exists(os.path.join(self.root, 'partition', 'data.pt')):
            subdomains = torch.load(os.path.join(self.root, 'partition', 'data.pt'))
        else:
            if mode == 'train':
                num_processes = mp.cpu_count()
                len_single_process = max(len(data) // (num_processes - 1), 1)
                data_list = [(data[i * len_single_process:(i + 1) * len_single_process], self.sub_size) for i in range(0, len(data), len_single_process)]
                with mp.Pool(num_processes) as pool:
                    # self.data_test = self._get_partiton_domain(data_list[0])
                    subdomains = pool.map(self._get_partiton_domain, data_list)
                torch.save(subdomains, os.path.join(self.root, 'partition', 'data.pt'))
        return subdomains
    
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
    
    # @staticmethod
    # def get_graph_boundary_edges(data, dimension=3):
    #     """
    #     returns the boundary edges of a graph
        
    #     :param data: the graph stored in a torch_geometric.data.Data
    #     :param dimension: the defined geometrical dimension of the graph
    #     """
    #     # get adjacency matrix
    #     adj = pyg.utils.to_dense_adj(data.edge_index).squeeze()
    #     # get boundary edges as edgws with only one cell assignments
    #     boundary_edges = []
    #     boundary_edges = torch.where(adj.sum(dim=0) == 1)[0]

    #     return boundary_edges

        

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
        reconstructed_mesh = self.smooth_vtu_with_continuity(reconstructed_mesh)

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
            projector = DivergenceFreeProjection(vtk_grid, velocity_array_name="velocity", pressure_array_name="pressure")
            initial_divergence = projector.calculate_divergence()

            print(f"Initial divergence: {initial_divergence}")

            corrected_velocity, corrected_pressure, final_div_norm, iterations = projector.apply_divergence_free_projection(
                max_iterations=20,
                tolerance=1e-2,
                pressure_solver_tol=1e-3
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
def _calculate_divergence_kernel(velocity, weights_matrix, neighbors_mask, valid_points, n_points):
    """Numba-optimized kernel for divergence calculation"""
    divergence = np.zeros(n_points)
    
    for i in prange(n_points):
        if valid_points[i] == 0:
            continue
            
        # Find the number of valid neighbors
        n_neighbors = 0
        for j in range(weights_matrix.shape[2]):
            if neighbors_mask[i, j] > 0:
                n_neighbors += 1
            else:
                break
        
        # Compute velocity differences and apply weights
        div_x = 0.0
        div_y = 0.0
        div_z = 0.0
        
        for j in range(n_neighbors):
            neighbor_idx = neighbors_mask[i, j]
            # Velocity difference
            dx = velocity[neighbor_idx, 0] - velocity[i, 0]
            dy = velocity[neighbor_idx, 1] - velocity[i, 1]
            dz = velocity[neighbor_idx, 2] - velocity[i, 2]
            
            # Apply weights
            div_x += weights_matrix[i, 0, j] * dx
            div_y += weights_matrix[i, 1, j] * dy
            div_z += weights_matrix[i, 2, j] * dz
        
        # Sum components to get divergence
        divergence[i] = div_x + div_y + div_z
    
    return divergence


@njit(parallel=True)
def _apply_pressure_gradient_kernel(velocity, pressure, weights_matrix, neighbors_mask, valid_points, n_points):
    """Numba-optimized kernel for pressure gradient application"""
    corrected_velocity = velocity.copy()
    
    for i in prange(n_points):
        if valid_points[i] == 0:
            continue
            
        # Find the number of valid neighbors
        n_neighbors = 0
        for j in range(weights_matrix.shape[2]):
            if neighbors_mask[i, j] > 0:
                n_neighbors += 1
            else:
                break
        
        # Compute pressure gradient
        grad_x = 0.0
        grad_y = 0.0
        grad_z = 0.0
        
        for j in range(n_neighbors):
            neighbor_idx = neighbors_mask[i, j]
            # Pressure difference
            dp = pressure[neighbor_idx] - pressure[i]
            
            # Apply weights
            grad_x += weights_matrix[i, 0, j] * dp
            grad_y += weights_matrix[i, 1, j] * dp
            grad_z += weights_matrix[i, 2, j] * dp
        
        # Apply pressure gradient correction
        corrected_velocity[i, 0] -= grad_x
        corrected_velocity[i, 1] -= grad_y
        corrected_velocity[i, 2] -= grad_z
    
    return corrected_velocity

@njit(parallel=True)
def _build_laplacian_kernel(points, valid_points, neighbors_mask, n_points, max_neighbors):
    """
    Numba-optimized kernel to build the Laplacian matrix data
    
    Parameters:
    -----------
    points : numpy.ndarray
        Array of point coordinates, shape (n_points, 3)
    valid_points : numpy.ndarray
        Mask indicating which points have valid neighbors
    neighbors_mask : numpy.ndarray
        Matrix of neighbor indices, shape (n_points, max_neighbors)
    n_points : int
        Number of points in the mesh
    max_neighbors : int
        Maximum number of neighbors per point
    
    Returns:
    --------
    rows : numpy.ndarray
        Row indices for sparse matrix
    cols : numpy.ndarray
        Column indices for sparse matrix
    data : numpy.ndarray
        Values for sparse matrix
    """
    # Estimate the number of non-zero entries (overestimate)
    estimated_nnz = n_points + n_points * max_neighbors
    
    # Pre-allocate arrays for sparse matrix construction
    rows = np.zeros(estimated_nnz, dtype=np.int32)
    cols = np.zeros(estimated_nnz, dtype=np.int32)
    data = np.zeros(estimated_nnz, dtype=np.float64)
    
    # Counter for filling the arrays
    nnz_count = 0
    
    for i in prange(n_points):
        # For each point, set diagonal element
        rows[nnz_count] = i
        cols[nnz_count] = i
        data[nnz_count] = 1.0  # Will be adjusted later if point has neighbors
        nnz_count += 1
        
        if not valid_points[i]:
            continue
        
        # Find valid neighbors for this point
        n_valid_neighbors = 0
        neighbor_indices = np.zeros(max_neighbors, dtype=np.int32)
        
        for j in range(max_neighbors):
            if neighbors_mask[i, j] > 0:
                neighbor_indices[n_valid_neighbors] = neighbors_mask[i, j]
                n_valid_neighbors += 1
            else:
                break
        
        if n_valid_neighbors == 0:
            continue
        
        # Compute weights based on inverse distance
        weights = np.zeros(n_valid_neighbors, dtype=np.float64)
        total_weight = 0.0
        
        for idx in range(n_valid_neighbors):
            j = neighbor_indices[idx]
            # Use inverse distance weighting
            vec_x = points[j, 0] - points[i, 0]
            vec_y = points[j, 1] - points[i, 1]
            vec_z = points[j, 2] - points[i, 2]
            dist_squared = vec_x*vec_x + vec_y*vec_y + vec_z*vec_z
            
            if dist_squared > 1e-12:
                weight = 1.0 / np.sqrt(dist_squared)
                weights[idx] = weight
                total_weight += weight
        
        if total_weight > 0:
            # Set off-diagonal elements (neighbors)
            for idx in range(n_valid_neighbors):
                j = neighbor_indices[idx]
                normalized_weight = weights[idx] / total_weight
                
                rows[nnz_count] = i
                cols[nnz_count] = j
                data[nnz_count] = -normalized_weight
                nnz_count += 1
            
            # Adjust diagonal element
            for k in range(nnz_count):
                if rows[k] == i and cols[k] == i:
                    data[k] = 1.0
                    break
    
    # Return only the used portion of the arrays
    return rows[:nnz_count], cols[:nnz_count], data[:nnz_count]


class DivergenceFreeProjection:
    """
    An optimized class for applying divergence-free projection to velocity fields on unstructured grids
    """
    def __init__(self, vtk_grid, velocity_array_name="velocity", pressure_array_name="pressure"):
        """Initialize with a VTK unstructured grid"""
        self.vtk_grid = vtk_grid
        self.velocity_array_name = velocity_array_name
        self.pressure_array_name = pressure_array_name
        
        # Convert VTK grid to numpy-friendly wrapper
        self.grid = dsa.WrapDataObject(vtk_grid)
        
        # Get points and cells
        self.points = self.grid.Points
        self.n_points = self.points.shape[0]
        
        # Extract velocity field
        try:
            self.velocity = self.grid.PointData[velocity_array_name]
            if self.velocity.shape[1] != 3:
                raise ValueError(f"Velocity field should have 3 components, but has {self.velocity.shape[1]}")
        except KeyError:
            raise KeyError(f"Velocity field '{velocity_array_name}' not found in point data")
            
        # Extract or initialize pressure field
        try:
            self.pressure = self.grid.PointData[pressure_array_name]
        except KeyError:
            print(f"Pressure field '{pressure_array_name}' not found in point data, initializing to zeros")
            self.pressure = np.zeros(self.n_points)
        
        # Build point-to-cell connectivity and other necessary data structures
        self._build_connectivity()

    def _compute_boundary_points(self):
        """Compute boundary points based on neighbor count"""
        print("Computing boundary points...")
        start_time = time.time()
        
        # Count neighbors for each point
        neighbor_counts = np.zeros(self.n_points, dtype=np.int32)
        for i in range(self.n_points):
            for j in range(self.weights_matrix.shape[2]):
                if self.neighbors_mask[i, j] > 0:
                    neighbor_counts[i] += 1
        
        # Calculate average number of neighbors
        valid_counts = neighbor_counts[neighbor_counts > 0]
        avg_neighbors = np.mean(valid_counts) if len(valid_counts) > 0 else 0
        
        # Identify boundary points (points with few neighbors)
        threshold = 0.2 * avg_neighbors
        boundary_points = np.zeros(self.n_points, dtype=np.bool_)
        for i in range(self.n_points):
            if neighbor_counts[i] > 0 and neighbor_counts[i] < threshold:
                boundary_points[i] = True
        
        print(f"Identified {np.sum(boundary_points)} boundary points in {time.time() - start_time:.2f} seconds")
        
        return boundary_points
        
    def _build_connectivity(self):
        """Build connectivity information for the unstructured grid"""
        print("Building grid connectivity...")
        start_time = time.time()
        
        # Get cell connectivity
        n_cells = self.vtk_grid.GetNumberOfCells()
        n_points = self.n_points
        
        # Process cells with progress reporting
        report_interval = max(1, n_cells // 20)
        
        # Use arrays to store cell data - more compatible with Numba
        self.cells = []
        self.cell_types = []
        
        for i in range(n_cells):
            # Progress reporting
            if i % report_interval == 0:
                print(f"Processing cells: {i}/{n_cells} ({100.0 * i / n_cells:.1f}%)")
                
            cell = self.vtk_grid.GetCell(i)
            cell_type = cell.GetCellType()
            
            # Get points that form this cell
            cell_points = []
            for j in range(cell.GetNumberOfPoints()):
                cell_points.append(cell.GetPointId(j))
            
            self.cells.append(cell_points)
            self.cell_types.append(cell_type)
        
        # Build point-to-cell connectivity
        self.point_to_cells = [[] for _ in range(n_points)]
        for cell_idx, cell in enumerate(self.cells):
            for point_idx in cell:
                self.point_to_cells[point_idx].append(cell_idx)
        
        # Convert to flat array representation for Numba compatibility
        # For each point, calculate neighbors and store as NumPy arrays
        self.point_neighbors = []
        
        # Report progress for this step too
        report_interval = max(1, n_points // 20)
        for i in range(n_points):
            if i % report_interval == 0:
                print(f"Building point neighbors: {i}/{n_points} ({100.0 * i / n_points:.1f}%)")
                
            # Find all neighbors through cells
            neighbors = set()
            for cell_idx in self.point_to_cells[i]:
                for j in self.cells[cell_idx]:
                    if j != i:
                        neighbors.add(j)
            
            # Store as NumPy array
            self.point_neighbors.append(np.array(list(neighbors), dtype=np.int32))
        
        # Store flat arrays for Numba
        neighbors_counts = np.array([len(neighbors) for neighbors in self.point_neighbors], dtype=np.int32)
        max_neighbors = max(neighbors_counts) if neighbors_counts.size > 0 else 0
        print(f"Maximum neighbors per point: {max_neighbors}")
        
        # Pre-allocate weights matrices for Numba
        self.weights_matrix = np.zeros((n_points, 3, max_neighbors), dtype=np.float64)
        self.neighbors_mask = np.zeros((n_points, max_neighbors), dtype=np.int32)
        
        print(f"Grid connectivity built in {time.time() - start_time:.2f} seconds")
        
        # Compute weights
        self._compute_weights()

        self.boundary_points = self._compute_boundary_points()

    def _compute_weights(self):
        """Compute weights for gradient and divergence approximation"""
        print("Computing gradient and divergence weights...")
        start_time = time.time()
        
        # Initialize weights for gradient calculation
        n_points = self.n_points
        report_interval = max(1, n_points // 20)
        
        # For each point, compute weights for gradient approximation
        for i in range(n_points):
            # Progress reporting
            if i % report_interval == 0:
                print(f"Computing weights: {i}/{n_points} ({100.0 * i / n_points:.1f}%)")
                
            neighbors = self.point_neighbors[i]
            n_neighbors = len(neighbors)
            
            if n_neighbors == 0:
                # Isolated point, skip
                continue
            
            # Set up least squares problem for gradient
            A = np.zeros((n_neighbors, 3))
            for idx, j in enumerate(neighbors):
                A[idx, 0] = self.points[j, 0] - self.points[i, 0]
                A[idx, 1] = self.points[j, 1] - self.points[i, 1]
                A[idx, 2] = self.points[j, 2] - self.points[i, 2]
            
            # Solve for weights using pseudo-inverse (least squares)
            try:
                # Use SVD for more stable computation
                U, S, Vh = np.linalg.svd(A, full_matrices=False)
                # Check for very small singular values that could cause instability
                S_inv = np.where(S > 1e-10, 1.0 / S, 0.0)
                pinv = (Vh.T * S_inv) @ U.T
                
                # The shape of pinv is (3, n_neighbors)
                # Store the weights in our pre-allocated arrays
                for j in range(n_neighbors):
                    if j < self.weights_matrix.shape[2]:  # Prevent index errors
                        self.neighbors_mask[i, j] = neighbors[j]
                        # Access each component directly
                        self.weights_matrix[i, 0, j] = pinv[0, j]
                        self.weights_matrix[i, 1, j] = pinv[1, j]
                        self.weights_matrix[i, 2, j] = pinv[2, j]
                        
            except np.linalg.LinAlgError:
                # Fall back to simple averaging if matrix is singular
                weight_value = 1.0 / n_neighbors
                for j in range(n_neighbors):
                    if j < self.weights_matrix.shape[2]:  # Prevent index errors
                        self.neighbors_mask[i, j] = neighbors[j]
                        self.weights_matrix[i, 0, j] = weight_value
                        self.weights_matrix[i, 1, j] = weight_value
                        self.weights_matrix[i, 2, j] = weight_value
        
        # Create a mask for valid points (those with neighbors)
        self.valid_points = np.array([len(neighbors) > 0 for neighbors in self.point_neighbors], dtype=np.int32)
        
        print(f"Weights computed in {time.time() - start_time:.2f} seconds")
        print(f"Points with valid weights: {np.sum(self.valid_points)}/{n_points}")
    
    def calculate_divergence(self):
        """Calculate divergence of the velocity field"""
        print("Calculating divergence...")
        start_time = time.time()
        
        # Use the Numba-optimized kernel for calculation
        divergence = _calculate_divergence_kernel(
            self.velocity, 
            self.weights_matrix, 
            self.neighbors_mask,
            self.valid_points,
            self.n_points
        )
        
        print(f"Divergence calculated in {time.time() - start_time:.2f} seconds")
        print(f"Max absolute divergence: {np.max(np.abs(divergence))}")
        print(f"Average absolute divergence: {np.mean(np.abs(divergence))}")
        
        return divergence
    
    def build_laplacian_matrix(self):
        """
        Build the Laplacian matrix for the unstructured grid using Numba acceleration
        
        Returns:
        --------
        scipy.sparse.csr_matrix
            Sparse Laplacian matrix
        """
        print("Building Laplacian matrix...")
        start_time = time.time()
        
        # Get dimensionality info needed for Numba kernel
        n_points = self.n_points
        max_neighbors = self.weights_matrix.shape[2]
        
        # Call the Numba-optimized kernel
        rows, cols, data = _build_laplacian_kernel(
            self.points, 
            self.valid_points, 
            self.neighbors_mask, 
            n_points, 
            max_neighbors
        )
        
        # Create sparse matrix from COO format
        laplacian = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
        
        print(f"Laplacian matrix built in {time.time() - start_time:.2f} seconds")
        
        return laplacian
    
    def solve_pressure_poisson(self, divergence, max_iterations=2000, tolerance=1e-5, use_direct_solver=False):
        """Solve the pressure Poisson equation"""
        print("Solving pressure Poisson equation...")
        start_time = time.time()
        
        # Build Laplacian matrix
        laplacian = self.build_laplacian_matrix()
        
        # Prepare right-hand side
        rhs = -divergence
        
        rhs[self.boundary_points] = 0
        
        # Solve the system
        if use_direct_solver:
            print("Using direct solver...")
            pressure = spsolve(laplacian, rhs)
        else:
            print(f"Using BiCGSTAB iterative solver with tolerance {tolerance}...")
            pressure, info = bicgstab(laplacian, rhs, rtol=tolerance, maxiter=max_iterations)
            
            if info != 0:
                print(f"Warning: BiCGSTAB did not converge, info = {info}")
                print("Falling back to GMRES...")
                pressure, info = gmres(laplacian, rhs, rtol=tolerance, maxiter=max_iterations)
                
                if info != 0:
                    print(f"Warning: GMRES did not converge, info = {info}")
                    print("Falling back to direct solver...")
                    pressure = spsolve(laplacian, rhs)
        
        print(f"Pressure Poisson equation solved in {time.time() - start_time:.2f} seconds")
        
        return pressure
    
    def apply_pressure_gradient(self, pressure):
        """Apply pressure gradient correction to velocity field using numba parallelization"""
        print("Applying pressure gradient correction...")
        start_time = time.time()
        
        # Apply gradient correction in parallel
        corrected_velocity = _apply_pressure_gradient_kernel(
            self.velocity, 
            pressure, 
            self.weights_matrix, 
            self.neighbors_mask,
            self.valid_points,
            self.n_points
        )
        
        print(f"Pressure gradient correction applied in {time.time() - start_time:.2f} seconds")
        
        return corrected_velocity
    
    def apply_divergence_free_projection(self, max_iterations=10, tolerance=1e-4, pressure_solver_tol=1e-5):
        """Apply divergence-free projection to the velocity field"""
        print("Starting divergence-free projection...")
        total_start_time = time.time()
        
        # Initialize
        corrected_velocity = np.copy(self.velocity)
        final_pressure = np.copy(self.pressure)
        
        # Calculate initial divergence
        self.velocity = corrected_velocity
        divergence = self.calculate_divergence()
        initial_div_norm = np.linalg.norm(divergence)
        current_div_norm = initial_div_norm
        
        print(f"Initial divergence norm: {initial_div_norm}")
        
        # Iterative projection
        iteration = 0
        
        while current_div_norm > tolerance * initial_div_norm and iteration < max_iterations:
            print(f"\nIteration {iteration+1}/{max_iterations}:")
            
            # Solve pressure Poisson equation
            pressure = self.solve_pressure_poisson(divergence, tolerance=pressure_solver_tol)
            
            # Update pressure
            final_pressure = pressure
            
            # Apply pressure gradient correction
            self.velocity = corrected_velocity
            corrected_velocity = self.apply_pressure_gradient(pressure)
            
            # Recalculate divergence
            self.velocity = corrected_velocity
            divergence = self.calculate_divergence()
            current_div_norm = np.linalg.norm(divergence)
            
            print(f"Divergence norm: {current_div_norm} ({current_div_norm/initial_div_norm:.4f} of initial)")
            
            iteration += 1
            
            if current_div_norm <= tolerance * initial_div_norm:
                print(f"Converged after {iteration} iterations")
                break
        
        if iteration == max_iterations:
            print(f"Warning: Did not converge to tolerance after {max_iterations} iterations")
        
        # Calculate divergence reduction
        div_reduction = 100 * (1 - current_div_norm / initial_div_norm)
        print(f"\nDivergence reduced by {div_reduction:.2f}%")
        print(f"Final max absolute divergence: {np.max(np.abs(divergence))}")
        print(f"Total projection time: {time.time() - total_start_time:.2f} seconds")
        
        return corrected_velocity, final_pressure, current_div_norm, iteration
    
    def update_vtk_grid(self, corrected_velocity, corrected_pressure):
        """
        Update the VTK grid with corrected velocity and pressure fields
        
        Parameters:
        -----------
        corrected_velocity : numpy.ndarray
            Corrected velocity field
        corrected_pressure : numpy.ndarray
            Corrected pressure field
            
        Returns:
        --------
        vtkUnstructuredGrid
            Updated VTK grid
        """
        print("Updating VTK grid...")
        
        # Convert velocity to VTK array
        vtk_velocity = numpy_support.numpy_to_vtk(corrected_velocity, deep=1)
        vtk_velocity.SetName(self.velocity_array_name)
        
        # Convert pressure to VTK array
        vtk_pressure = numpy_support.numpy_to_vtk(corrected_pressure, deep=1)
        vtk_pressure.SetName(self.pressure_array_name)
        
        # Update point data
        point_data = self.vtk_grid.GetPointData()
        
        # Update or add velocity
        if point_data.HasArray(self.velocity_array_name):
            point_data.RemoveArray(self.velocity_array_name)
        point_data.AddArray(vtk_velocity)
        
        # Update or add pressure
        if point_data.HasArray(self.pressure_array_name):
            point_data.RemoveArray(self.pressure_array_name)
        point_data.AddArray(vtk_pressure)
        
        # Calculate divergence for visualization
        divergence = self.calculate_divergence()
        vtk_divergence = numpy_support.numpy_to_vtk(divergence, deep=1)
        vtk_divergence.SetName("divergence")
        
        if point_data.HasArray("divergence"):
            point_data.RemoveArray("divergence")
        point_data.AddArray(vtk_divergence)
        
        print("VTK grid updated successfully")
        return self.vtk_grid