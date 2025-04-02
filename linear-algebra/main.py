from array import NArray as na

if __name__ == "__main__":
    # Using the class methods
    print("Testing class methods:\n")
    
    matrix_data = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    # na.array() method
    matrix = na.array(data=matrix_data, dtype=float)
    print(f"2D array using na.array():\n {matrix}")
    print(f"Shape: {matrix.shape}")
    print(f"Size: {matrix.size}")
    print(f"Data type: {matrix.dtype.__name__}\n")
    
    # na.zeros() method
    zeros_array = na.zeros(shape=(2, 3, 4), dtype=float)
    print(f"3D zeros array:\n {zeros_array}")
    print(f"Shape: {zeros_array.shape}")
    print(f"Size: {zeros_array.size}")
    print(f"Data type: {zeros_array.dtype.__name__}\n")
    
    # na.ones() method
    ones_array = na.ones(shape=(2, 2), dtype=float)
    print(f"2D ones array:\n {ones_array}")
    print(f"Shape: {ones_array.shape}")
    print(f"Size: {ones_array.size}")
    print(f"Data type: {ones_array.dtype.__name__}\n")
    
    # Different data types
    int_array = na.array(data=[1, 2, 3, 4], dtype=int)
    print(f"Integer array: {int_array}")
    print(f"Data type: {int_array.dtype.__name__}\n")
    
    # Demonstrating that direct instantiation is not allowed
    print("Testing direct instantiation (should raise an error):")
    try:
        # This should now raise an error
        array = na(data=[[1, 2], [3, 4]], dtype=float)
        print("Direct instantiation succeeded (this shouldn't happen)")
    except RuntimeError as e:
        print(f"Expected error: {e}\n")
    
    # Error cases with proper usage
    print("Testing data validation errors:")
    
    # Test case 1: Different dimensions
    bad_data1 = [
        [[1, 2], [3, 4]],  # 2D array
        [5, 6]  # 1D array
    ]
    try:
        array = na.array(data=bad_data1, dtype=float)
        print(f"Inconsistent shape (should have raised error): {array}")
    except ValueError as e:
        print(f"Expected error case 1: {e}\n")
    
    # Test case 2: Mixed types at same level
    bad_data2 = [
        [[1, 2], 4],  # Mixed types
        [[5, 6], 7]  # Mixed types
    ]
    try:
        array = na.array(data=bad_data2, dtype=float)
        print(f"Inconsistent shape (should have raised error): {array}")
    except ValueError as e:
        print(f"Expected error case 2: {e}")


    ##############################################################################

    print("\n\n\nIDENTITY MATRIX EXAMPLES")
    print("-----------------------")
    
    # Create a 3x3 identity matrix
    id_matrix = na.identity(3, dtype=float)
    print("3x3 Identity Matrix:")
    print(id_matrix)
    print(f"Shape: {id_matrix.shape}")
    print(f"Size: {id_matrix.size}")
    print()
    
    # Create a 5x5 identity matrix with integer type
    id_matrix_int = na.identity(5, dtype=int)
    print("5x5 Identity Matrix (int type):")
    print(id_matrix_int)
    print()
    
    print("EYE MATRIX EXAMPLES")
    print("------------------")
    
    # Create a 3x3 eye matrix (same as identity)
    eye_matrix = na.eye(3, dtype=float)
    print("3x3 Eye Matrix (same as identity):")
    print(eye_matrix)
    print()
    
    # Create a 3x5 rectangular eye matrix
    rect_eye = na.eye(3, 5, dtype=float)
    print("3x5 Rectangular Eye Matrix:")
    print(rect_eye)
    print()
    
    # Create a 4x4 matrix with ones on the first super-diagonal (k=1)
    super_diag = na.eye(4, k=1, dtype=float)
    print("4x4 Matrix with 1s on the super-diagonal (k=1):")
    print(super_diag)
    print()
    
    # Create a 4x4 matrix with ones on the first sub-diagonal (k=-1)
    sub_diag = na.eye(4, k=-1, dtype=float)
    print("4x4 Matrix with 1s on the sub-diagonal (k=-1):")
    print(sub_diag)
    print()
    
    # Create a 4x6 matrix with ones on diagonal offset by 2
    complex_eye = na.eye(4, 6, k=2, dtype=int)
    print("4x6 Matrix with 1s on diagonal offset by 2:")
    print(complex_eye)


    ##############################################################################

    print("\n\n\nRANDOM ARRAY EXAMPLES")
    print("----------------------")
    
    # Set a seed for reproducibility
    seed = 42
    
    # Create a 3x3 array with random values between 0 and 1
    random_array = na.random(shape=(3, 3, 4), seed=seed, decimals=4)
    print("3x3 Random Array (values between 0 and 1):")
    print(random_array)
    print(f"Shape: {random_array.shape}")
    print(f"Data type: {random_array.dtype.__name__}")
    print()
    
    # Create a 2x4 array with random values from a uniform distribution
    uniform_array = na.uniform(low=-10.0, high=10.0, shape=(2, 4), seed=seed, decimals=2)
    print("2x4 Uniform Array (values between -10 and 10):")
    print(uniform_array)
    print(f"Shape: {uniform_array.shape}")
    print()
    
    # Create a 3x2 array with random integers
    randint_array = na.randint(low=1, high=100, shape=(3, 2), seed=seed)
    print("3x2 Random Integer Array (values between 1 and 99):")
    print(randint_array)
    print()
    
    # Create a 2x2x2 array with random values from a normal distribution
    normal_array = na.normal(loc=0.0, scale=1.0, shape=(2, 2, 2), seed=seed, decimals=3)
    print("2x2x2 Normal Distribution Array (mean=0, std=1):")
    print(normal_array)
    print(f"Shape: {normal_array.shape}")
    print()
    
    # Create arrays with different data types
    int_random = na.random(shape=(2, 2), seed=seed, dtype=int)
    print("Random Array with Integer Type (values will be truncated to 0):")
    print(int_random)
    print(f"Data type: {int_random.dtype.__name__}")
