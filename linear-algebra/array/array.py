from typing import List, Tuple, Optional, Any, Union

class NArray:
    """
    Base class for N-dimensional arrays with comprehensive functionality.
    
    This class provides the foundation for all array operations in the linear algebra library.
    It handles N-dimensional data structures and implements core array operations with
    operator overloading for intuitive usage.
    
    Attributes:
        data: Nested list structure containing the array elements.
        _shape: Tuple representing dimensions of the array.
        dtype: Data type of the array elements.
    """
    
    def __init__(
        self,
        data: Optional[List] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[type] = None,
    ) -> None:
        """
        Initialize an N-dimensional array.
        
        Args:
            data: Data to initialize the array with.
            shape: Shape of the array if data is not provided.
            dtype: Data type for the array elements.
        
        Raises:
            AssertionError: If neither data nor shape is provided.
            ValueError: If the input data structure is not valid or has inconsistent dimensions.
        """
        assert data is not None or shape is not None, "You must provide either data or shape"
        
        # Set or infer dtype
        self.dtype = dtype
        
        # Create array from either data or shape
        if data is None:
            # Create empty array of zeros with given shape
            self.data = self._zeros(shape, dtype or float)
            self._shape = shape
        else:
            # Validate and convert data
            if isinstance(data, (int, float)):
                # Handle scalar input
                self.data = data
                self._shape = ()
                self.dtype = self.dtype or type(data)
            else:
                # Handle list/tuple input
                # Validate data structure and shape consistency
                self.data = self._validate_data(data)
                # Apply type conversion if needed
                if dtype:
                    self.data = self._cast_to_type(self.data, dtype)
                
                self._shape = self._compute_shape(data)
                self.dtype = dtype if dtype else self._infer_dtype(data)

    def _validate_data(self, data: Union[List, Tuple, Any]) -> Any:
        """
        Validate both nesting structure and shape consistency of the data.
        
        Args:
            data: The data to validate.
            
        Returns:
            The validated data (unchanged if valid).
            
        Raises:
            ValueError: If inconsistent nesting or shapes are detected.
        """
        # Base case: not a list/tuple
        if not isinstance(data, (list, tuple)):
            return data
        
        # Empty list case
        if not data:
            return []
        
        # Check nesting consistency at this level
        first_is_nested = isinstance(data[0], (list, tuple))
        for i, item in enumerate(data):
            current_is_nested = isinstance(item, (list, tuple))
            if current_is_nested != first_is_nested:
                raise ValueError(f"Inconsistent nesting at index {i}: "
                                f"expected {'list' if first_is_nested else 'scalar'}, "
                                f"got {'list' if current_is_nested else 'scalar'}")
        
        # If elements are nested, check shape consistency and validate recursively
        if first_is_nested:
            first_shape = self._compute_shape(data[0])
            for i, item in enumerate(data):
                item_shape = self._compute_shape(item)
                if item_shape != first_shape:
                    raise ValueError(f"Inconsistent shapes at index {i}: "
                                    f"expected {first_shape}, got {item_shape}")
                
                # Recursively validate nested structures
                data[i] = self._validate_data(item)
        
        return data
    
    def _zeros(self, shape: Tuple[int, ...], dtype: Optional[type] = None) -> Any:
        """
        Create a nested list of zeros with the given shape.
        
        Args:
            shape: Dimensions of the array.
            dtype: Data type for the elements.
            
        Returns:
            Nested list of zeros.
        """
        dtype = dtype or float
        
        # Handle empty shape
        if not shape:
            return dtype(0)
            
        # Handle 1D arrays
        if len(shape) == 1:
            if shape[0] == 0:
                return []
            return [dtype(0)] * shape[0]
            
        # Handle multi-dimensional arrays recursively
        return [self._zeros(shape[1:], dtype) for _ in range(shape[0])]
    
    def _compute_shape(self, data: Any) -> Tuple[int, ...]:
        """
        Recursively determine the shape of nested data.
        
        Args:
            data: The data to determine shape for.
            
        Returns:
            Shape of the data.
        """
        if not isinstance(data, (list, tuple)):
            return ()
        
        if not data:
            return (0,)
            
        if isinstance(data[0], (list, tuple)):
            return (len(data),) + self._compute_shape(data[0])
        else:
            return (len(data),)
    
    def _infer_dtype(self, data: Any) -> type:
        """
        Infer the data type from the data.
        
        Args:
            data: The data to infer type from.
            
        Returns:
            Inferred data type.
        """
        if not isinstance(data, (list, tuple)) or not data:
            return float
            
        if isinstance(data[0], (list, tuple)):
            return self._infer_dtype(data[0])
        else:
            # Get the first non-None value
            for item in data:
                if item is not None:
                    return type(item)
            return float  # Default if all None
    
    def _cast_to_type(self, data: Any, dtype: type) -> Any:
        """
        Convert all elements in the data to the specified dtype.
        
        Args:
            data: The data to convert.
            dtype: The target data type.
            
        Returns:
            Data with all elements converted to dtype.
        """
        if not isinstance(data, (list, tuple)):
            try:
                return dtype(data)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Cannot convert {data} to {dtype}: {e}")
                
        return [self._cast_to_type(item, dtype) for item in data]
    
    # Properties
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the array."""
        return self._shape
    
    @property
    def ndims(self) -> int:
        """Get the number of dimensions of the array."""
        return len(self._shape)
    
    # String representations
    
    def __repr__(self) -> str:
        """String representation of the array for debugging."""
        if not self._shape:  # Scalar
            return f"NArray({self.data}, dtype={self.dtype.__name__})"
            
        # For 1D arrays
        if len(self._shape) == 1:
            return f"NArray({self.data}, dtype={self.dtype.__name__})"
            
        # For multi-dimensional arrays
        return f"NArray(shape={self._shape}, dtype={self.dtype.__name__})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        if not self._shape or len(self._shape) == 0 or len(self._shape) == 1:  # Scalar
            return f"NArray({self.data}, dtype={self.dtype.__name__})"
            
        # For 2D arrays, format as a matrix
        if len(self._shape) == 2:
            rows = []
            for row in self.data:
                rows.append("[" + ", ".join(str(x) for x in row) + "]")
            matrix_str = "[" + ",\n ".join(rows) + "]"
            return f"NArray({matrix_str}, dtype={self.dtype.__name__})"
            
        # For higher dimensions, just show shape and type
        return f"NArray(shape={self._shape}, dtype={self.dtype.__name__})"


if __name__ == "__main__":
    # 3D array
    nested_data = [
        [[1, 2, 10], [3, 6, 9]],  
        [[5, 6, 7], [8, 9, 10]]  
    ]

    array = NArray(data=nested_data, dtype=float)
    print(f"3D array: {array}")

    # 2D array
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    array = NArray(data=matrix, dtype=float)
    print(f"\n\n2D array:\n {array}") 

    # 1D array
    array = NArray(data=[1, 2, 3], dtype=float)
    print(f"\n\n1D array: {array}")  

    # Scalar
    array = NArray(data=1, dtype=float)
    print(f"\n\nScalar: {array}") 

    # Empty array
    array = NArray(data=[], dtype=float)
    print(f"\n\nEmpty array: {array}")  

    # data from shape
    array = NArray(shape=(3, 3), dtype=float)
    print(f"\n\nArray from shape: {array}") 

    # Test case 1: Different dimensions
    bad_data1 = [
        [[1, 2], [3, 4]],  # 2D array
        [5, 6]             # 1D array
    ]

    try:
        array = NArray(data=bad_data1, dtype=float)
        print(f"\n\nInconsistent shape (should have raised error): {array}")  
    except ValueError as e:
        print(f"\n\nExpected error case 1: {e}")

    # Test case 2: Mixed types at same level
    bad_data2 = [
        [[1, 2], 4],       # Mixed types
        [[5, 6], 7]             # 1D array
    ]

    try:
        array = NArray(data=bad_data2, dtype=float)
        print(f"\n\nInconsistent shape (should have raised error): {array}")  
    except ValueError as e:
        print(f"\n\nExpected error case 2: {e}")

    # Test case 3: Different length lists at same nesting level
    bad_data3 = [
        [[1, 2], [3, 4]],   # 2x2
        [[5, 6, 7], [8, 9, 10]]  # 2x3
    ]

    try:
        array = NArray(data=bad_data3, dtype=float)
        print(f"\n\nInconsistent shape (should have raised error): {array}")  
    except ValueError as e:
        print(f"\n\nExpected error case 3: {e}")