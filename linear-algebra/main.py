from array import NDArray

if __name__ == "__main__":
    print("BASIC ARRAY CREATION")
    print("-------------------")
    
    # Create from data
    matrix = NDArray.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    print(f"Matrix from data:\n{matrix}")
    print(f"Shape: {matrix.shape}")
    print(f"Size: {matrix.size}")
    print(f"Data type: {matrix.dtype.__name__}")
    
    # Create zeros array
    zeros = NDArray.zeros((2, 3), dtype=float)
    print(f"\nZeros array:\n{zeros}")
    print(f"Shape: {zeros.shape}")
    
    # Create ones array
    ones = NDArray.ones((2, 2), dtype=int)
    print(f"\nOnes array:\n{ones}")
    print(f"Shape: {ones.shape}")
    
    # Create identity matrix
    identity = NDArray.identity(3, dtype=float)
    print(f"\nIdentity matrix (3x3):\n{identity}")
    
    # Create eye matrix with offset
    eye_offset = NDArray.eye(3, k=1, dtype=int)
    print(f"\nEye matrix with offset (k=1):\n{eye_offset}")
    
    print("\nNEWLY ADDED METHODS")
    print("------------------")
    
    # Arange: Create array with evenly spaced values
    print("\nArange method examples:")
    arr1 = NDArray.arange(10)
    print(f"arange(10): {arr1.data}")
    
    arr2 = NDArray.arange(1, 11)
    print(f"arange(1, 11): {arr2.data}")
    
    arr3 = NDArray.arange(0, 1, 0.1)
    print(f"arange(0, 1, 0.1): {arr3.data}")
    
    # Linspace: Create array with num evenly spaced samples
    print("\nLinspace method examples:")
    lin1 = NDArray.linspace(0, 1, 5)
    print(f"linspace(0, 1, 5): {lin1.data}")
    
    lin2 = NDArray.linspace(0, 1, 5, endpoint=False)
    print(f"linspace(0, 1, 5, endpoint=False): {lin2.data}")
    
    # Reshape method
    print("\nReshape method examples:")
    flat_array = NDArray.array([1, 2, 3, 4, 5, 6])
    print(f"Original array: {flat_array.data}")
    
    reshaped = flat_array.reshape((2, 3))
    print(f"After reshape(2, 3):\n{reshaped}")
    
    # Using -1 for automatic dimension calculation
    reshaped_auto = flat_array.reshape((2, -1))
    print(f"After reshape(2, -1):\n{reshaped_auto}")
    
    # Flatten method
    print("\nFlatten method examples:")
    nested_array = NDArray.array([[1, 2], [3, 4], [5, 6]])
    print(f"Original array:\n{nested_array}")
    
    flattened = nested_array.flatten()
    print(f"After flatten(): {flattened.data}")
    
    # Transpose method
    print("\nTranspose method examples:")
    matrix = NDArray.array([[1, 2, 3], [4, 5, 6]])
    print(f"Original matrix (2x3):\n{matrix}")
    
    transposed = matrix.transpose()
    print(f"After transpose (3x2):\n{transposed}")
    
    # Higher dimensional transpose
    print("\nHIGHER DIMENSIONAL TRANSPOSE")
    print("--------------------------")
    
    # Create a 3D array
    array_3d = NDArray.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ])
    print(f"Original 3D array (shape {array_3d.shape}):")
    
    # Print each 2D slice for better visualization
    print("First slice:")
    print(array_3d.data[0])
    print("Second slice:")
    print(array_3d.data[1])
    
    # Transpose the 3D array
    transposed_3d = array_3d.transpose()
    print(f"\nTransposed 3D array (shape {transposed_3d.shape}):")
    
    # Print each 2D slice of the transposed array
    for i in range(transposed_3d.shape[0]):
        print(f"Slice {i}:")
        print(transposed_3d.data[i])
    
    # Create a 4D array
    array_4d = NDArray.zeros((2, 2, 2, 2))
    # Fill with sequential values
    flat = array_4d.flatten()
    for i in range(flat.size):
        flat.data[i] = i + 1
    array_4d = flat.reshape((2, 2, 2, 2))
    
    print(f"\n4D array shape: {array_4d.shape}")
    print(f"Transposed 4D array shape: {array_4d.transpose().shape}")
    
    # Error cases
    print("\nERROR CASES")
    print("-----------")
    
    try:
        # Reshape with incompatible dimensions
        NDArray.array([1, 2, 3]).reshape((2, 2))
    except ValueError as e:
        print(f"Expected error: {e}")