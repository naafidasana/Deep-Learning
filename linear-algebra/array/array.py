import random
from typing import Callable, List, Tuple, Optional, Any, Union, Sequence

class NDArray:
    """
    Base class for N-dimensional arrays with core functionality.
    
    This class provides the foundation for array operations in the linear algebra library.
    It handles N-dimensional data structures and implements basic array operations.
    
    Arrays can be created using the static methods:
        NDArray.array() - Create from data
        NDArray.zeros() - Create array of zeros with shape
        NDArray.ones() - Create array of ones with shape
        NDArray.eye() - Create a 2D array with ones on the diagonal and zeros elsewhere
        NDArray.identity() - Create a square identity matrix
        NDArray.arange() - Create an array with evenly spaced values within a given interval
        NDArray.linspace() - Create an array with evenly spaced values within a given interval
    
    Attributes:
        data: Nested list structure containing the array elements.
        shape: Tuple representing dimensions of the array.
        dtype: Data type of the array elements.
        size: Total number of elements in the array.
    """
    
    def __init__(self, data: Any = None, dtype: Optional[type] = None) -> None:
        """
        Initialize an N-dimensional array.
        
        Args:
            data: Data to initialize the array with.
            dtype: Data type for the array elements.
        
        Raises:
            ValueError: If the input data structure is not valid or has inconsistent dimensions.
        """
        # Handle scalar inputs
        if not isinstance(data, (list, tuple)):
            self.data = data
            self._shape = ()  # Scalar has empty shape
            self._dtype = type(data) if dtype is None else dtype
            self._size = 1  # Scalar has size 1
            return
            
        # Handle list/tuple inputs
        if not data:  # Empty list
            self.data = []
            self._shape = (0,)
            self._dtype = dtype or float
            self._size = 0
            return
            
        # Validate and process data
        self.data = self._validate_data(data)
        self._shape = self._compute_shape(self.data)
        self._dtype = dtype or self._infer_dtype(self.data)
        
        # Ensure all elements have the correct type if dtype is specified
        if dtype:
            self.data = self._cast_to_type(self.data, self._dtype)
            
        # Calculate size
        self._size = self._calculate_size_from_data(self.data)

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
    
    def _calculate_size_from_data(self, data: Any) -> int:
        """
        Calculate the total number of elements in the data.
        
        Args:
            data: The data structure.
            
        Returns:
            Total number of elements.
        """
        # For scalars
        if not isinstance(data, (list, tuple)):
            return 1
            
        # For empty lists
        if not data:
            return 0
            
        # For nested structures
        if isinstance(data[0], (list, tuple)):
            return sum(self._calculate_size_from_data(item) for item in data)
            
        # For 1D arrays
        return len(data)
    
    @staticmethod
    def _zeros_internal(shape: Tuple[int, ...], dtype: Optional[type] = None) -> Any:
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
        return [NDArray._zeros_internal(shape[1:], dtype) for _ in range(shape[0])]
    
    @staticmethod
    def _ones_internal(shape: Tuple[int, ...], dtype: Optional[type] = None) -> Any:
        """
        Create a nested list of ones with the given shape.
        
        Args:
            shape: Dimensions of the array.
            dtype: Data type for the elements.
            
        Returns:
            Nested list of ones.
        """
        dtype = dtype or float
        
        # Handle empty shape
        if not shape:
            return dtype(1)
            
        # Handle 1D arrays
        if len(shape) == 1:
            if shape[0] == 0:
                return []
            return [dtype(1)] * shape[0]
            
        # Handle multi-dimensional arrays recursively
        return [NDArray._ones_internal(shape[1:], dtype) for _ in range(shape[0])]
    
    @staticmethod
    def _eye_internal(n: int, m: Optional[int] = None, k: int = 0, dtype: Optional[type] = None) -> List[List]:
        """
        Create a 2D array with ones on the diagonal and zeros elsewhere.
        
        Args:
            n: Number of rows.
            m: Number of columns. If None, defaults to n.
            k: Index of the diagonal. 0 (default) refers to the main diagonal,
               a positive value refers to an upper diagonal, and a negative value
               to a lower diagonal.
            dtype: Data type for the elements.
            
        Returns:
            2D array with ones on the specified diagonal.
        """
        dtype = dtype or float
        m = m if m is not None else n
        
        # Create a 2D array of zeros first
        result = [[dtype(0) for _ in range(m)] for _ in range(n)]
        
        # Fill the specified diagonal with ones
        for i in range(n):
            j = i + k
            if 0 <= j < m:
                result[i][j] = dtype(1)
                
        return result
    
    # Static methods for array creation
    @staticmethod
    def array(data: Any, dtype: Optional[type] = None) -> 'NDArray':
        """
        Create a new array from data.
        
        Args:
            data: Data to initialize the array with.
            dtype: Data type for the array elements.
            
        Returns:
            New NDArray instance.
        """
        return NDArray(data=data, dtype=dtype)
    
    @staticmethod
    def zeros(shape: Tuple[int, ...], dtype: Optional[type] = None) -> 'NDArray':
        """
        Create an array of zeros with the given shape.
        
        Args:
            shape: Dimensions of the array.
            dtype: Data type for the elements.
            
        Returns:
            New NDArray instance filled with zeros.
        """
        data = NDArray._zeros_internal(shape, dtype)
        return NDArray(data=data, dtype=dtype)
    
    @staticmethod
    def ones(shape: Tuple[int, ...], dtype: Optional[type] = None) -> 'NDArray':
        """
        Create an array of ones with the given shape.
        
        Args:
            shape: Dimensions of the array.
            dtype: Data type for the elements.
            
        Returns:
            New NDArray instance filled with ones.
        """
        data = NDArray._ones_internal(shape, dtype)
        return NDArray(data=data, dtype=dtype)
    
    @staticmethod
    def eye(n: int, m: Optional[int] = None, k: int = 0, dtype: Optional[type] = None) -> 'NDArray':
        """
        Create a 2D array with ones on the diagonal and zeros elsewhere.
        
        This is a more flexible form of the identity matrix that can create rectangular matrices
        and place the ones on any diagonal.
        
        Args:
            n: Number of rows.
            m: Number of columns. If None, defaults to n (square matrix).
            k: Index of the diagonal. 0 (default) refers to the main diagonal,
               a positive value refers to an upper diagonal, and a negative value
               to a lower diagonal.
            dtype: Data type for the elements.
            
        Returns:
            New NDArray instance with ones on the specified diagonal.
            
        Examples:
            eye(3) creates a 3x3 identity matrix.
            eye(3, 4) creates a 3x4 matrix with ones on the main diagonal.
            eye(3, 3, 1) creates a 3x3 matrix with ones on the first super-diagonal.
            eye(3, 3, -1) creates a 3x3 matrix with ones on the first sub-diagonal.
        """
        data = NDArray._eye_internal(n, m, k, dtype)
        return NDArray(data=data, dtype=dtype)
    
    @staticmethod
    def identity(n: int, dtype: Optional[type] = None) -> 'NDArray':
        """
        Create a square identity matrix of size n x n.
        
        An identity matrix is a square matrix with ones on the main diagonal
        and zeros elsewhere.
        
        Args:
            n: Size of the identity matrix (n x n).
            dtype: Data type for the elements.
            
        Returns:
            New NDArray instance representing an identity matrix.
            
        Example:
            identity(3) creates a 3x3 identity matrix:
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        """
        # Use the eye method with default parameters
        return NDArray.eye(n, dtype=dtype)
    
    @staticmethod
    def arange(start: float, stop: Optional[float] = None, step: float = 1, 
               dtype: Optional[type] = None) -> 'NDArray':
        """
        Create an array with evenly spaced values within a given interval.
        
        Args:
            start: Start of interval. If stop is None, start at 0 and stop at this value.
            stop: End of interval. The endpoint is not included.
            step: Spacing between values.
            dtype: Data type of output array.
            
        Returns:
            NDArray instance with evenly spaced values.
            
        Examples:
            arange(10) -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            arange(1, 11) -> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            arange(0, 5, 0.5) -> [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        """
        if stop is None:
            stop = start
            start = 0
            
        # Calculate the number of elements
        num = int((stop - start) / step)
        
        # Generate the values
        values = [start + i * step for i in range(num)]
        
        # Apply dtype if provided
        if dtype:
            values = [dtype(x) for x in values]
            
        return NDArray(data=values, dtype=dtype)
    
    @staticmethod
    def linspace(start: float, stop: float, num: int = 50, 
                 endpoint: bool = True, dtype: Optional[type] = None) -> 'NDArray':
        """
        Create an array with num evenly spaced values over the specified interval.
        
        Args:
            start: The starting value of the sequence.
            stop: The end value of the sequence.
            num: Number of samples to generate.
            endpoint: If True, stop is the last sample. Otherwise, stop is not included.
            dtype: Data type of output array.
            
        Returns:
            NDArray instance with evenly spaced values.
            
        Examples:
            linspace(0, 1, 5) -> [0.0, 0.25, 0.5, 0.75, 1.0]
            linspace(0, 1, 5, endpoint=False) -> [0.0, 0.2, 0.4, 0.6, 0.8]
        """
        if num <= 0:
            raise ValueError("Number of samples must be positive")
            
        if endpoint:
            # Include the endpoint
            step = (stop - start) / (num - 1) if num > 1 else 0
        else:
            # Exclude the endpoint
            step = (stop - start) / num
        
        values = [start + i * step for i in range(num)]
            
        # Apply dtype if provided
        if dtype:
            values = [dtype(x) for x in values]
            
        return NDArray(data=values, dtype=dtype)
    
    # Instance methods for array manipulation
    
    def flatten(self) -> 'NDArray':
        """
        Flatten the array to 1D.
        
        Returns:
            Flattened NDArray.
            
        Example:
            If arr has shape (2, 3), arr.flatten() will have shape (6,)
        """
        if not self._shape:  # Scalar
            return NDArray([self.data], dtype=self._dtype)
            
        flat_data = self._flatten_list(self.data)
        return NDArray(flat_data, dtype=self._dtype)
    
    def _flatten_list(self, data: Any) -> List:
        """
        Recursively flatten nested lists.
        
        Args:
            data: Nested list structure.
            
        Returns:
            Flattened list.
        """
        if not isinstance(data, (list, tuple)):
            return [data]
            
        result = []
        for item in data:
            result.extend(self._flatten_list(item))
            
        return result
    
    def reshape(self, new_shape: Tuple[int, ...]) -> 'NDArray':
        """
        Reshape the array to a new shape.
        
        Args:
            new_shape: New shape for the array. One dimension can be -1, 
                       which will be automatically calculated.
            
        Returns:
            Reshaped NDArray.
            
        Raises:
            ValueError: If the new shape is not compatible with the number of elements.
            
        Examples:
            If arr has shape (6,), arr.reshape((2, 3)) will have shape (2, 3)
            If arr has shape (6,), arr.reshape((2, -1)) will also have shape (2, 3)
        """
        # Calculate total elements
        total_elements = self._size
        
        # Calculate the total size of new shape
        new_elements = 1
        for dim in new_shape:
            if dim == -1:  # One dimension can be specified as -1
                continue
            new_elements *= dim
        
        # Check if shapes are compatible
        if -1 in new_shape:
            # One dimension is inferred
            inferred_idx = new_shape.index(-1)
            inferred_dim = total_elements // new_elements
            new_shape_list = list(new_shape)
            new_shape_list[inferred_idx] = inferred_dim
            new_shape = tuple(new_shape_list)
            new_elements *= inferred_dim
        
        if total_elements != new_elements:
            raise ValueError(f"Cannot reshape array of size {total_elements} into shape {new_shape}")
        
        # Flatten the array and then reshape
        flattened = self.flatten().data
        result_data = self._reshape_list(flattened, new_shape)
        
        return NDArray(result_data, dtype=self._dtype)
    
    def _reshape_list(self, flat_data: List, shape: Tuple[int, ...]) -> Any:
        """
        Reshape a flat list into nested lists according to shape.
        
        Args:
            flat_data: Flattened list of data.
            shape: Target shape.
            
        Returns:
            Nested list with the specified shape.
        """
        if not shape:  # Empty shape, return scalar
            return flat_data[0] if flat_data else None
            
        if len(shape) == 1:
            return flat_data[:shape[0]]
            
        # Multi-dimensional reshaping
        result = []
        items_per_sublist = 1
        for dim in shape[1:]:
            items_per_sublist *= dim
            
        for i in range(shape[0]):
            start_idx = i * items_per_sublist
            end_idx = start_idx + items_per_sublist
            sublist = flat_data[start_idx:end_idx]
            result.append(self._reshape_list(sublist, shape[1:]))
            
        return result
    
    def transpose(self) -> 'NDArray':
        """
        Transpose the array by reversing the order of dimensions.
        For 2D arrays, this flips the array over its diagonal.
        For N-dimensional arrays, this reverses the order of axes.
        
        Returns:
            Transposed NDArray.
        """
        # Handle scalar case
        if not self._shape:
            return NDArray(self.data, dtype=self._dtype)
            
        # Handle 1D case
        if len(self._shape) == 1:
            return NDArray(self.data[:], dtype=self._dtype)
        
        # Helper function for recursive transposition
        def recursive_transpose(matrix):
            if not matrix:  # Handle empty arrays
                return matrix
            if not isinstance(matrix[0], list):  # Base case - 1D array
                return matrix
            return [recursive_transpose(list(row)) for row in zip(*matrix)]
        
        transposed_data = recursive_transpose(self.data)
        return NDArray(transposed_data, dtype=self._dtype)
    
    def squeeze(self) -> 'NDArray':
        """
        Remove all dimensions of size 1 from the array shape.
        
        Returns:
            NDArray: A view of the array with all dimensions of size 1 removed.
            
        Examples:
            If arr has shape (3, 1, 4, 1), arr.squeeze() will have shape (3, 4)
            If arr has shape (1, 1, 1), arr.squeeze() will be a scalar
            If arr has shape (3, 4), arr.squeeze() will also have shape (3, 4)
        """
        # For scalar arrays, return a copy
        if not self._shape:
            return NDArray(self.data, dtype=self._dtype)
        
        # Find dimensions of size 1
        new_shape = tuple(dim for dim in self._shape if dim != 1)
        
        # If all dimensions were 1, return a scalar
        if not new_shape:
            return NDArray(self.data if isinstance(self.data, (list, tuple)) and len(self.data) > 0 else self.data[0][0], 
                        dtype=self._dtype)
        
        # Otherwise, reshape to remove the dimensions of size 1
        return self.reshape(new_shape)

    def expand_dims(self, axis: int) -> 'NDArray':
        """
        Expand the shape of the array by inserting a new dimension of size 1 at the given position.
        
        Args:
            axis: Position in the expanded axes where the new axis is placed.
                Allowed range is [-ndim-1, ndim], where ndim is the number of dimensions
                before expansion.
                
        Returns:
            NDArray: View of the array with an additional dimension.
            
        Raises:
            ValueError: If axis is out of the allowed range.
            
        Examples:
            If arr has shape (3, 4):
            - arr.expand_dims(0) will have shape (1, 3, 4)
            - arr.expand_dims(1) will have shape (3, 1, 4)
            - arr.expand_dims(2) will have shape (3, 4, 1)
            - arr.expand_dims(-1) will have shape (3, 4, 1)
        """
        # Handle scalar case
        if not self._shape:
            if axis not in [0, -1]:
                raise ValueError(f"Axis {axis} is out of bounds for scalar array")
            # Return a 1D array with a single element
            return NDArray([self.data], dtype=self._dtype)
        
        # Calculate number of dimensions
        ndim = len(self._shape)
        
        # Validate axis
        if axis < -ndim - 1 or axis > ndim:
            raise ValueError(f"Axis {axis} is out of bounds for array of dimension {ndim}")
        
        # Convert negative axis to positive
        if axis < 0:
            axis = ndim + axis + 1
        
        # Create new shape by inserting 1 at the specified position
        new_shape = list(self._shape)
        new_shape.insert(axis, 1)
        
        # Reshape the array
        return self.reshape(tuple(new_shape))
    
    # Properties with getters
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the array."""
        return self._shape
    
    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the array."""
        return len(self._shape)
    
    @property
    def dtype(self) -> type:
        """Get the data type of the array."""
        return self._dtype
    
    @property
    def size(self) -> int:
        """Get the total number of elements in the array."""
        return self._size
    
    # String representations
    
    def __repr__(self) -> str:
        """String representation of the array for debugging."""
        # Ensure dtype is not None before accessing __name__
        dtype_name = self._dtype.__name__ if self._dtype is not None else "None"
        
        if not self._shape:  # Scalar
            return f"NDArray({self.data}, dtype={dtype_name})"
            
        # For 1D arrays
        if len(self._shape) == 1:
            return f"NDArray({self.data}, dtype={dtype_name})"
            
        # For multi-dimensional arrays
        return f"NDArray(shape={self._shape}, dtype={dtype_name})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        if not self._shape:  # Scalar
            return str(self.data)
            
        # For 1D arrays
        if len(self._shape) == 1:
            return str(self.data)
            
        # For 2D arrays, format as a matrix
        if len(self._shape) == 2:
            rows = []
            for row in self.data:
                rows.append("[" + ", ".join(str(x) for x in row) + "]")
            return "[" + ",\n ".join(rows) + "]"
            
        # For higher dimensions, just show shape and type
        return f"NDArray(shape={self._shape}, dtype={self._dtype.__name__})"
 
    # Dunder methods --> Helpers
    def _broadcast_operation(self, other: Union['NDArray', Any], operation: Callable[[Any, Any], Any]) -> 'NDArray':
        """
        Apply a binary operation with broadcasting.
        
        Broadcasting rules similar to NumPy:
        1. If arrays don't have same number of dimensions, prepend shape with 1s
        2. If shapes don't match in any dimension, one of them must be 1
        3. In the output, each dimension is the max of the input dimensions
        
        Args:
            other: Another NDArray or a scalar value.
            operation: Function that takes two arguments and returns the result.
            
        Returns:
            NDArray: Result of the broadcasted operation.
            
        Raises:
            ValueError: If shapes are incompatible for broadcasting.
        """
        # Handle scalar case
        if not isinstance(other, NDArray):
            other = NDArray(other)
            
        # Fast path 1: Both are scalars
        if not self._shape and not other._shape:
            return NDArray(operation(self.data, other.data))
            
        # Fast path 2: One is scalar, one is array
        if not self._shape:  # self is scalar
            return _apply_operation_with_scalar(other, self.data, operation, scalar_on_right=False)
        if not other._shape:  # other is scalar
            return _apply_operation_with_scalar(self, other.data, operation, scalar_on_right=True)
            
        # Fast path 3: Same shape arrays (no broadcasting needed)
        if self._shape == other._shape:
            return _apply_operation_same_shape(self, other, operation)
            
        # Fast path 4: Simple 1D broadcasting
        if len(self._shape) == 1 and len(other._shape) == 1:
            if self._shape[0] == 1:
                # Broadcast self to other's shape
                result = [operation(self.data[0], x) for x in other.data]
                return NDArray(result)
            elif other._shape[0] == 1:
                # Broadcast other to self's shape
                result = [operation(x, other.data[0]) for x in self.data]
                return NDArray(result)
        
        # General case: n-dimensional broadcasting
        # Calculate the broadcast shape
        broadcast_shape = _compute_broadcast_shape(self._shape, other._shape)
        
        # Create the output array structure
        result_data = _create_broadcast_result(self, other, broadcast_shape, operation)
        
        return NDArray(result_data)
    
    # Dunder methods --> Basics maths operations

    # 1. Basic Arithmetic Operations
    def __add__(self, other: Any) -> 'NDArray':
        """
        Element-wise addition with broadcasting support.
        
        This method implements the '+' operator between two NDArrays or an NDArray and a scalar.
        
        Args:
            other: Another NDArray or a scalar value.
            
        Returns:
            NDArray: Result of element-wise addition.
            
        Examples:
            arr1 + arr2  # Element-wise addition of two arrays
            arr + 5      # Add 5 to each element
        """
        # Handle scalar case
        if not isinstance(other, NDArray):
            other = NDArray(other)
            
        # Define addition operation
        def add_op(x: Any, y: Any) -> Any:
            return x + y
        
        return self._broadcast_operation(other, add_op)
    
    def __radd__(self, other: Any) -> 'NDArray':
        """
        Reverse element-wise addition with broadcasting support.
        
        This method implements the '+' operator between a scalar and an NDArray.
        
        Args:
            other: A scalar value.
            
        Returns:
            NDArray: Result of element-wise addition.
            
        Examples:
            5 + arr  # Add 5 to each element
        """
        return self.__add__(other)
    
    def __sub__(self, other: Any) -> 'NDArray':
        """
        Element-wise subtraction with broadcasting support.
        
        This method implements the '-' operator between two NDArrays or an NDArray and a scalar.
        
        Args:
            other: Another NDArray or a scalar value.
            
        Returns:
            NDArray: Result of element-wise subtraction.
            
        Examples:
            arr1 - arr2  # Element-wise subtraction of two arrays
            arr - 5      # Subtract 5 from each element
        """
        # Handle scalar case
        if not isinstance(other, NDArray):
            other = NDArray(other)

        # Define subtraction operation
        def sub_op(x: Any, y: Any) -> Any:
            return x - y
        return self._broadcast_operation(other, sub_op)
    
    def __rsub__(self, other: Any) -> 'NDArray':
        """
        Reverse element-wise subtraction with broadcasting support.
        
        This method implements the '-' operator between a scalar and an NDArray.
        
        Args:
            other: A scalar value.
            
        Returns:
            NDArray: Result of element-wise subtraction.
            
        Examples:
            5 - arr  # Subtract 5 from each element
        """
        return self.__sub__(other)

    def __mul__(self, other: Any) -> 'NDArray':
        """
        Element-wise multiplication with broadcasting support.
        
        This method implements the '*' operator between two NDArrays or an NDArray and a scalar.
        
        Args:
            other: Another NDArray or a scalar value.
            
        Returns:
            NDArray: Result of element-wise multiplication.
            
        Examples:
            arr1 * arr2  # Element-wise multiplication
            arr * 5      # Multiply each element by 5
        """
        # Handle scalar case
        if not isinstance(other, NDArray):
            other = NDArray(other)
            
        # Define multiplication operation
        def mul_op(x: Any, y: Any) -> Any:
            return x * y
            
        # Apply operation with broadcasting
        return self._broadcast_operation(other, mul_op)
        
    def __rmul__(self, other: Any) -> 'NDArray':
        """
        Reverse multiplication (for scalar * array).
        
        This method is called when a non-NDArray is on the left side of the '*' operator.
        
        Args:
            other: A scalar value or non-NDArray object.
            
        Returns:
            NDArray: Result of element-wise multiplication.
            
        Examples:
            5 * arr  # Multiply each element by 5
        """
        return self.__mul__(other)
        
    def __truediv__(self, other: Any) -> 'NDArray':
        """
        Element-wise true division with broadcasting support.
        
        This method implements the '/' operator between two NDArrays or an NDArray and a scalar.
        
        Args:
            other: Another NDArray or a scalar value.
            
        Returns:
            NDArray: Result of element-wise division.
            
        Examples:
            arr1 / arr2  # Element-wise division
            arr / 5      # Divide each element by 5
            
        Raises:
            ZeroDivisionError: If division by zero occurs.
        """
        # Handle scalar case
        if not isinstance(other, NDArray):
            other = NDArray(other)
            
        # Define division operation
        def div_op(x: Any, y: Any) -> Any:
            if y == 0:
                raise ZeroDivisionError("division by zero")
            return x / y
            
        # Apply operation with broadcasting
        return self._broadcast_operation(other, div_op)
        
    def __rtruediv__(self, other: Any) -> 'NDArray':
        """
        Reverse true division (for scalar / array).
        
        This method is called when a non-NDArray is on the left side of the '/' operator.
        
        Args:
            other: A scalar value or non-NDArray object.
            
        Returns:
            NDArray: Result of element-wise division.
            
        Examples:
            5 / arr  # Divide 5 by each element
            
        Raises:
            ZeroDivisionError: If division by zero occurs.
        """
        # Handle scalar case
        if not isinstance(other, NDArray):
            other = NDArray(other)
            
        # Define reverse division operation
        def rdiv_op(x: Any, y: Any) -> Any:
            if x == 0:
                raise ZeroDivisionError("division by zero")
            return y / x
            
        # Apply operation with broadcasting
        return self._broadcast_operation(other, rdiv_op)
        
    def __floordiv__(self, other: Any) -> 'NDArray':
        """
        Element-wise floor division with broadcasting support.
        
        This method implements the '//' operator between two NDArrays or an NDArray and a scalar.
        
        Args:
            other: Another NDArray or a scalar value.
            
        Returns:
            NDArray: Result of element-wise floor division.
            
        Examples:
            arr1 // arr2  # Element-wise floor division
            arr // 5      # Floor divide each element by 5
            
        Raises:
            ZeroDivisionError: If division by zero occurs.
        """
        # Handle scalar case
        if not isinstance(other, NDArray):
            other = NDArray(other)
            
        # Define floor division operation
        def floordiv_op(x: Any, y: Any) -> Any:
            if y == 0:
                raise ZeroDivisionError("division by zero")
            return x // y
            
        # Apply operation with broadcasting
        return self._broadcast_operation(other, floordiv_op)
        
    def __rfloordiv__(self, other: Any) -> 'NDArray':
        """
        Reverse floor division (for scalar // array).
        
        This method is called when a non-NDArray is on the left side of the '//' operator.
        
        Args:
            other: A scalar value or non-NDArray object.
            
        Returns:
            NDArray: Result of element-wise floor division.
            
        Examples:
            5 // arr  # Floor divide 5 by each element
            
        Raises:
            ZeroDivisionError: If division by zero occurs.
        """
        # Handle scalar case
        if not isinstance(other, NDArray):
            other = NDArray(other)
            
        # Define reverse floor division operation
        def rfloordiv_op(x: Any, y: Any) -> Any:
            if x == 0:
                raise ZeroDivisionError("division by zero")
            return y // x
            
        # Apply operation with broadcasting
        return self._broadcast_operation(other, rfloordiv_op)
        
    def __mod__(self, other: Any) -> 'NDArray':
        """
        Element-wise modulo operation with broadcasting support.
        
        This method implements the '%' operator between two NDArrays or an NDArray and a scalar.
        
        Args:
            other: Another NDArray or a scalar value.
            
        Returns:
            NDArray: Result of element-wise modulo.
            
        Examples:
            arr1 % arr2  # Element-wise modulo
            arr % 5      # Modulo each element by 5
            
        Raises:
            ZeroDivisionError: If modulo by zero occurs.
        """
        # Handle scalar case
        if not isinstance(other, NDArray):
            other = NDArray(other)
            
        # Define modulo operation
        def mod_op(x: Any, y: Any) -> Any:
            if y == 0:
                raise ZeroDivisionError("modulo by zero")
            return x % y
            
        # Apply operation with broadcasting
        return self._broadcast_operation(other, mod_op)
        
    def __rmod__(self, other: Any) -> 'NDArray':
        """
        Reverse modulo (for scalar % array).
        
        This method is called when a non-NDArray is on the left side of the '%' operator.
        
        Args:
            other: A scalar value or non-NDArray object.
            
        Returns:
            NDArray: Result of element-wise modulo.
            
        Examples:
            5 % arr  # Modulo 5 by each element
            
        Raises:
            ZeroDivisionError: If modulo by zero occurs.
        """
        # Handle scalar case
        if not isinstance(other, NDArray):
            other = NDArray(other)
            
        # Define reverse modulo operation
        def rmod_op(x: Any, y: Any) -> Any:
            if x == 0:
                raise ZeroDivisionError("modulo by zero")
            return y % x
            
        # Apply operation with broadcasting
        return self._broadcast_operation(other, rmod_op)
        
    def __pow__(self, other: Any) -> 'NDArray':
        """
        Element-wise power operation with broadcasting support.
        
        This method implements the '**' operator between two NDArrays or an NDArray and a scalar.
        
        Args:
            other: Another NDArray or a scalar value.
            
        Returns:
            NDArray: Result of element-wise power.
            
        Examples:
            arr1 ** arr2  # Element-wise power
            arr ** 2      # Square each element
        """
        # Handle scalar case
        if not isinstance(other, NDArray):
            other = NDArray(other)
            
        # Define power operation
        def pow_op(x: Any, y: Any) -> Any:
            return x ** y
            
        # Apply operation with broadcasting
        return self._broadcast_operation(other, pow_op)
        
    def __rpow__(self, other: Any) -> 'NDArray':
        """
        Reverse power (for scalar ** array).
        
        This method is called when a non-NDArray is on the left side of the '**' operator.
        
        Args:
            other: A scalar value or non-NDArray object.
            
        Returns:
            NDArray: Result of element-wise power.
            
        Examples:
            2 ** arr  # Raise 2 to the power of each element
        """
        # Handle scalar case
        if not isinstance(other, NDArray):
            other = NDArray(other)
            
        # Define reverse power operation
        def rpow_op(x: Any, y: Any) -> Any:
            return y ** x
            
        # Apply operation with broadcasting
        return self._broadcast_operation(other, rpow_op)





# Helpers
def _apply_operation_with_scalar(array: 'NDArray', scalar: Any, 
                                operation: Callable[[Any, Any], Any], 
                                scalar_on_right: bool) -> 'NDArray':
    """Apply operation between array and scalar efficiently."""
    
    def _apply_recursive(data, scalar):
        """Recursively apply operation to nested lists."""
        if not isinstance(data, (list, tuple)):
            # Base case: data is a scalar
            return operation(data, scalar) if scalar_on_right else operation(scalar, data)
        
        # Recursive case: data is a list/tuple
        return [_apply_recursive(item, scalar) for item in data]
    
    result_data = _apply_recursive(array.data, scalar)
    return NDArray(result_data)


def _apply_operation_same_shape(arr1: 'NDArray', arr2: 'NDArray', 
                               operation: Callable[[Any, Any], Any]) -> 'NDArray':
    """Apply operation between arrays of the same shape efficiently."""
    
    def _apply_recursive(data1, data2):
        """Recursively apply operation to nested lists of same shape."""
        if not isinstance(data1, (list, tuple)):
            # Base case: both data1 and data2 are scalars
            return operation(data1, data2)
        
        # Recursive case: both are lists/tuples
        return [_apply_recursive(item1, item2) for item1, item2 in zip(data1, data2)]
    
    result_data = _apply_recursive(arr1.data, arr2.data)
    return NDArray(result_data)


def _compute_broadcast_shape(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute the resulting shape when broadcasting two arrays."""
    # Pad shapes to the same length
    padded_shape1 = list(shape1)
    padded_shape2 = list(shape2)
    
    while len(padded_shape1) < len(padded_shape2):
        padded_shape1.insert(0, 1)
    while len(padded_shape2) < len(padded_shape1):
        padded_shape2.insert(0, 1)
    
    # Compute the broadcast shape
    result_shape = []
    for s1, s2 in zip(padded_shape1, padded_shape2):
        if s1 == s2 or s1 == 1 or s2 == 1:
            result_shape.append(max(s1, s2))
        else:
            raise ValueError(f"Incompatible shapes for broadcasting: {shape1} and {shape2}")
    
    return tuple(result_shape)


def _create_broadcast_result(arr1: 'NDArray', arr2: 'NDArray', 
                            broadcast_shape: Tuple[int, ...], 
                            operation: Callable[[Any, Any], Any]) -> Any:
    """Create the result array by applying the operation with broadcasting."""
    
    def _get_broadcast_value(arr, indices, original_shape):
        """Get value from array using broadcast indices."""
        # Convert broadcast indices to original indices
        original_indices = []
        offset = len(broadcast_shape) - len(original_shape)
        
        for i, dim in enumerate(original_shape):
            # If dimension is 1, use index 0 (broadcasting)
            # Otherwise use the corresponding broadcast index
            idx = indices[i + offset]
            if dim == 1:
                idx = 0
            original_indices.append(idx)
        
        # Navigate to the value using indices
        value = arr.data
        for idx in original_indices:
            value = value[idx]
        
        return value
    
    def _create_nested_result(indices, depth):
        """Recursively create the nested result structure."""
        if depth == len(broadcast_shape):
            # We've reached the target depth, compute the value
            value1 = _get_broadcast_value(arr1, indices, arr1._shape)
            value2 = _get_broadcast_value(arr2, indices, arr2._shape)
            return operation(value1, value2)
        
        # Create a list for this dimension
        result = []
        for i in range(broadcast_shape[depth]):
            indices[depth] = i
            result.append(_create_nested_result(indices, depth + 1))
        
        return result
    
    # Initialize indices and create the result
    indices = [0] * len(broadcast_shape)
    return _create_nested_result(indices, 0)


    