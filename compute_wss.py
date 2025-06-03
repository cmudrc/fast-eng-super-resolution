import vtk
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

def compute_wall_shear_stress(grid, velocity_array_name='velocity', 
                             wall_boundary_ids=None, 
                             dynamic_viscosity=1.0,
                             output_filename='wall_shear_stress.vtu'):
    """
    Compute wall shear stress from VTK unstructured grid with velocity data.
    
    Parameters:
    - grid: vtkUnstructuredGrid containing the mesh and data
    - velocity_array_name: name of velocity array in point data
    - wall_boundary_ids: list of cell IDs representing wall boundaries (if None, will attempt to detect)
    - dynamic_viscosity: fluid dynamic viscosity (Pa·s)
    - output_filename: filename for output with wall shear stress data
    """
    
    print("Computing wall shear stress...")
    
    # Get velocity data
    point_data = grid.GetPointData()
    velocity_data = point_data.GetArray(velocity_array_name)
    
    if velocity_data is None:
        raise ValueError(f"Velocity array '{velocity_array_name}' not found in point data")
    
    # Convert VTK arrays to numpy
    velocity = vtk_to_numpy(velocity_data)
    
    print(f"Grid has {grid.GetNumberOfPoints()} points and {grid.GetNumberOfCells()} cells")
    print(f"Velocity array shape: {velocity.shape}")
    
    # Compute velocity gradients
    gradient_filter = vtk.vtkGradientFilter()
    gradient_filter.SetInputData(grid)
    gradient_filter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, velocity_array_name)
    gradient_filter.SetResultArrayName('velocity_gradient')
    gradient_filter.Update()
    
    grid_with_gradients = gradient_filter.GetOutput()
    
    # Extract surface (boundary) mesh
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(grid_with_gradients)
    surface_filter.Update()
    surface_mesh = surface_filter.GetOutput()
    
    print(f"Surface mesh has {surface_mesh.GetNumberOfPoints()} points and {surface_mesh.GetNumberOfCells()} cells")
    
    # Compute surface normals
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(surface_mesh)
    normals_filter.ComputePointNormalsOn()
    normals_filter.ComputeCellNormalsOn()
    normals_filter.Update()
    surface_with_normals = normals_filter.GetOutput()
    
    # Get gradient data and normals
    gradient_data = surface_with_normals.GetPointData().GetArray('velocity_gradient')
    normals_data = surface_with_normals.GetPointData().GetArray('Normals')
    
    if gradient_data is None or normals_data is None:
        raise ValueError("Failed to compute gradients or normals")
    
    # Convert to numpy arrays
    gradients = vtk_to_numpy(gradient_data)  # Shape: (n_points, 9) for 3x3 gradient tensor
    normals = vtk_to_numpy(normals_data)     # Shape: (n_points, 3)
    
    # Reshape gradients to (n_points, 3, 3) tensor format
    if gradients.shape[1] == 9:
        gradients = gradients.reshape(-1, 3, 3)
    else:
        raise ValueError(f"Expected gradient array with 9 components, got {gradients.shape[1]}")
    
    # Compute wall shear stress
    n_points = surface_with_normals.GetNumberOfPoints()
    wall_shear_stress = np.zeros((n_points, 3))
    wall_shear_stress_magnitude = np.zeros(n_points)
    
    for i in range(n_points):
        # Get gradient tensor and normal vector at this point
        grad_u = gradients[i]  # 3x3 gradient tensor
        n = normals[i]         # Normal vector
        
        # Compute stress tensor: τ = μ * (∇u + ∇u^T)
        # For wall shear stress, we need: τ_wall = μ * (∇u + ∇u^T) * n
        stress_tensor = dynamic_viscosity * (grad_u + grad_u.T)
        
        # Wall shear stress vector: τ_wall = (τ·n) - ((τ·n)·n)*n
        # This removes the normal component, leaving only the tangential (shear) component
        tau_total = np.dot(stress_tensor, n)
        tau_normal = np.dot(tau_total, n)
        tau_wall = tau_total - tau_normal * n
        
        wall_shear_stress[i] = tau_wall
        wall_shear_stress_magnitude[i] = np.linalg.norm(tau_wall)
    
    # Add wall shear stress data to the surface mesh
    wss_vector_array = numpy_to_vtk(wall_shear_stress)
    wss_vector_array.SetName('WallShearStressVector')
    surface_with_normals.GetPointData().AddArray(wss_vector_array)
    
    wss_magnitude_array = numpy_to_vtk(wall_shear_stress_magnitude)
    wss_magnitude_array.SetName('WallShearStressMagnitude')
    surface_with_normals.GetPointData().AddArray(wss_magnitude_array)
    
    print(f"Wall shear stress computed. Max magnitude: {np.max(wall_shear_stress_magnitude):.6f} Pa")
    print(f"Mean magnitude: {np.mean(wall_shear_stress_magnitude):.6f} Pa")
    
    # Write results to file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_filename.replace('.vtu', '.vtp'))  # Use .vtp for polydata
    writer.SetInputData(surface_with_normals)
    writer.Write()
    
    print(f"Results written to: {output_filename.replace('.vtu', '.vtp')}")
    
    return surface_with_normals, wall_shear_stress, wall_shear_stress_magnitude

def load_vtk_grid(filename):
    """Load VTK unstructured grid from file."""
    if filename.endswith('.vtu'):
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif filename.endswith('.vtk'):
        reader = vtk.vtkUnstructuredGridReader()
    else:
        raise ValueError("Unsupported file format. Use .vtu or .vtk files.")
    
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

# Example usage
if __name__ == "__main__":
    # Load your VTK file
    input_filename = "logs/vtk/ansys_neuralop/pred_0.vtu"  # Replace with your file
    
    try:
        # Load the grid
        print(f"Loading VTK grid from: {input_filename}")
        grid = load_vtk_grid(input_filename)
        
        # Print available arrays for reference
        point_data = grid.GetPointData()
        print(f"\nAvailable point data arrays:")
        for i in range(point_data.GetNumberOfArrays()):
            array_name = point_data.GetArrayName(i)
            array = point_data.GetArray(i)
            print(f"  - {array_name}: {array.GetNumberOfComponents()} components, {array.GetNumberOfTuples()} tuples")
        
        # Compute wall shear stress
        # Adjust these parameters based on your data:
        surface_mesh, wss_vector, wss_magnitude = compute_wall_shear_stress(
            grid=grid,
            velocity_array_name='velocity',  # Change to match your velocity array name
            dynamic_viscosity=1.0e-3,       # Water at 20°C: ~1e-3 Pa·s, Air: ~1.8e-5 Pa·s
            output_filename='wall_shear_stress_results_pred.vtp'
        )

        surface_mesh, wss_vector, wss_magnitude = compute_wall_shear_stress(
            grid=grid,
            velocity_array_name='interpolated_velocity',  # Change to match your velocity array name
            dynamic_viscosity=1.0e-3,       # Water at 20°C: ~1e-3 Pa·s, Air: ~1.8e-5 Pa·s
            output_filename='wall_shear_stress_results_interpolated.vtp'
        )

        surface_mesh, wss_vector, wss_magnitude = compute_wall_shear_stress(
            grid=grid,
            velocity_array_name='ref_velocity',  # Change to match your velocity array name
            dynamic_viscosity=1.0e-3,       # Water at 20°C: ~1e-3 Pa·s, Air: ~1.8e-5 Pa·s
            output_filename='wall_shear_stress_results_reference.vtp'
        )
        
        print("\nWall shear stress computation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Replace 'your_grid_file.vtu' with your actual file path")
        print("2. Adjust velocity_array_name and pressure_array_name to match your data")
        print("3. Set appropriate dynamic_viscosity for your fluid")