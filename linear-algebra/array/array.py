import random
from typing import List, Tuple, Optional, Any, Union

class NArray:
    """
    Base class for N-dimensional arrays with comprehensive functionality.
    
    This class provides the foundation for all array operations in the linear algebra library.
    It handles N-dimensional data structures and implements core array operations with
    operator overloading for intuitive usage.
    
    Arrays should be created using the class methods:
        NArray.array() - Create from data
        NArray.zeros() - Create array of zeros with shape
        NArray.ones() - Create array of ones with shape
        NArray.eye() - Create a 2D array with ones on the diagonal and zeros elsewhere
        NArray.identity() - Create a square identity matrix
        NArray.random() - Create an array with random values between 0 and 1
        NArray.uniform() - Create an array with random values from a uniform distribution
        NArray.normal() - Create an array with random values from a normal distribution
        NArray.randint() - Create an array with random integer values
    
    Attributes:
        data: Nested list structure containing the array elements.
        _shape: Tuple representing dimensions of the array.
        _dtype: Data type of the array elements.
        _size: Total number of elements in the array.
    """
    
    def __init__(
        self,
        data: Optional[List] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[type] = None,
        _internal_call: bool = False
    ) -> None:
        """
        Initialize an N-dimensional array. This should not be called directly.
        Instead, use the class methods: array(), zeros(), ones(), etc.
        
        Args:
            data: Data to initialize the array with.
            shape: Shape of the array if data is not provided.
            dtype: Data type for the array elements.
            _internal_call: Flag to indicate if constructor was called internally.
        
        Raises:
            AssertionError: If neither data nor shape is provided.
            ValueError: If the input data structure is not valid or has inconsistent dimensions.
            RuntimeError: If constructor is called directly instead of through class methods.
        """
        if not _internal_call:
            raise RuntimeError(
                "Direct instantiation of NArray is not allowed. "
                "Use NArray.array(), NArray.zeros(), NArray.ones(), or other class methods instead."
            )
        
        if data is None and shape is None:
            self._shape = ()
        
        # Use instance variables instead of properties for initialization
        self._dtype = dtype
        
        # Create array from either data or shape
        if data is None:
            # Create empty array of zeros with given shape
            self.data = self._zeros_internal(shape, self._dtype or float)
            self._shape = shape
            # Calculate size from shape
            self._size = self._calculate_size_from_shape(shape)
        else:
            # Validate and convert data
            if isinstance(data, (int, float)):
                # Handle scalar input
                self.data = data
                self._shape = ()
                self._dtype = self._dtype or type(data)
                self._size = 1  # Scalar has size 1
            else:
                # Handle list/tuple input
                # Validate data structure and shape consistency
                self.data = self._validate_data(data)
                
                # Apply type conversion if needed
                if dtype:
                    self.data = self._cast_to_type(self.data, dtype)
                
                self._shape = self._compute_shape(data)
                self._dtype = dtype if dtype else self._infer_dtype(data)
                # Calculate size from data
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
    
    def _calculate_size_from_shape(self, shape: Tuple[int, ...]) -> int:
        """
        Calculate the total number of elements from the shape.
        
        Args:
            shape: The shape of the array.
            
        Returns:
            Total number of elements.
        """
        if not shape:
            return 1  # Scalar has size 1
            
        # Multiply all dimensions
        size = 1
        for dim in shape:
            size *= dim
        return size
    
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
    
    def _zeros_internal(self, shape: Tuple[int, ...], dtype: Optional[type] = None) -> Any:
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
        return [self._zeros_internal(shape[1:], dtype) for _ in range(shape[0])]
    
    def _ones_internal(self, shape: Tuple[int, ...], dtype: Optional[type] = None) -> Any:
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
        return [self._ones_internal(shape[1:], dtype) for _ in range(shape[0])]
    
    def _full_internal(self, shape: Tuple[int, ...], value: Any, dtype: Optional[type] = None) -> Any:
        """
        Create a nested list of the given value with the given shape.
        
        Args:
            shape: Dimensions of the array.
            value: Value to fill the array with.
            dtype: Data type for the elements.
            
        Returns:
            Nested list of the given value.
        """
        dtype = dtype or type(value)
        
        # Handle empty shape
        if not shape:
            return value
            
        # Handle 1D arrays
        if len(shape) == 1:
            if shape[0] == 0:
                return []
            return [value] * shape[0]
            
        # Handle multi-dimensional arrays recursively
        return [self._full_internal(shape[1:], value, dtype) for _ in range(shape[0])]
    
    def _eye_internal(self, n: int, m: Optional[int] = None, k: int = 0, dtype: Optional[type] = None) -> List[List]:
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
    
    def _random_internal(self, shape: Tuple[int, ...], seed: Optional[int] = None, 
                        decimals: Optional[int] = None) -> Any:
        """
        Create a nested list of random values between 0 and 1 with the given shape.
        
        Args:
            shape: Dimensions of the array.
            seed: Random seed for reproducibility.
            decimals: Number of decimal places to round to.
            
        Returns:
            Nested list of random values.
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
        
        # Handle empty shape
        if not shape:
            val = random.random()
            return round(val, decimals) if decimals is not None else val
            
        # Handle 1D arrays
        if len(shape) == 1:
            if shape[0] == 0:
                return []
            result = []
            for _ in range(shape[0]):
                val = random.random()
                result.append(round(val, decimals) if decimals is not None else val)
            return result
            
        # Handle multi-dimensional arrays recursively
        return [self._random_internal(shape[1:], None, decimals) for _ in range(shape[0])]

    def _uniform_internal(self, low: float, high: float, shape: Tuple[int, ...], 
                        seed: Optional[int] = None, decimals: Optional[int] = None) -> Any:
        """
        Create a nested list of random values from a uniform distribution with the given shape.
        
        Args:
            low: Lower bound of the distribution.
            high: Upper bound of the distribution.
            shape: Dimensions of the array.
            seed: Random seed for reproducibility.
            decimals: Number of decimal places to round to.
            
        Returns:
            Nested list of random values.
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
        
        # Handle empty shape
        if not shape:
            val = random.uniform(low, high)
            return round(val, decimals) if decimals is not None else val
            
        # Handle 1D arrays
        if len(shape) == 1:
            if shape[0] == 0:
                return []
            result = []
            for _ in range(shape[0]):
                val = random.uniform(low, high)
                result.append(round(val, decimals) if decimals is not None else val)
            return result
            
        # Handle multi-dimensional arrays recursively
        return [self._uniform_internal(low, high, shape[1:], None, decimals) for _ in range(shape[0])]

    def _normal_internal(self, loc: float, scale: float, shape: Tuple[int, ...], 
                        seed: Optional[int] = None, decimals: Optional[int] = None) -> Any:
        """
        Create a nested list of random values from a normal distribution with the given shape.
        
        Args:
            loc: Mean of the distribution.
            scale: Standard deviation of the distribution.
            shape: Dimensions of the array.
            seed: Random seed for reproducibility.
            decimals: Number of decimal places to round to.
            
        Returns:
            Nested list of random values.
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
        
        # Handle empty shape
        if not shape:
            val = random.normalvariate(loc, scale)
            return round(val, decimals) if decimals is not None else val
            
        # Handle 1D arrays
        if len(shape) == 1:
            if shape[0] == 0:
                return []
            result = []
            for _ in range(shape[0]):
                val = random.normalvariate(loc, scale)
                result.append(round(val, decimals) if decimals is not None else val)
            return result
            
        # Handle multi-dimensional arrays recursively
        return [self._normal_internal(loc, scale, shape[1:], None, decimals) for _ in range(shape[0])]

    def _randint_internal(self, low: int, high: int, shape: Tuple[int, ...], 
                        seed: Optional[int] = None, decimals: Optional[int] = None) -> Any:
        """
        Create a nested list of random integers with the given shape.
        
        Args:
            low: Lower bound of the distribution (inclusive).
            high: Upper bound of the distribution (exclusive).
            shape: Dimensions of the array.
            seed: Random seed for reproducibility.
            decimals: Not used for integers, but kept for API consistency.
            
        Returns:
            Nested list of random integers.
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
        
        # Handle empty shape
        if not shape:
            return random.randint(low, high - 1)
            
        # Handle 1D arrays
        if len(shape) == 1:
            if shape[0] == 0:
                return []
            return [random.randint(low, high - 1) for _ in range(shape[0])]
            
        # Handle multi-dimensional arrays recursively
        return [self._randint_internal(low, high, shape[1:], None, decimals) for _ in range(shape[0])]
    
    # Class methods for array creation
    @classmethod
    def array(cls, data: Any, dtype: Optional[type] = None) -> 'NArray':
        """
        Create a new array from data.
        
        Args:
            data: Data to initialize the array with.
            dtype: Data type for the array elements.
            
        Returns:
            New NArray instance.
        """
        return cls(data=data, dtype=dtype, _internal_call=True)
    
    @classmethod
    def zeros(cls, shape: Tuple[int, ...], dtype: Optional[type] = None) -> 'NArray':
        """
        Create an array of zeros with the given shape.
        
        Args:
            shape: Dimensions of the array.
            dtype: Data type for the elements.
            
        Returns:
            New NArray instance filled with zeros.
        """
        return cls(shape=shape, dtype=dtype, _internal_call=True)
    
    @classmethod
    def ones(cls, shape: Tuple[int, ...], dtype: Optional[type] = None) -> 'NArray':
        """
        Create an array of ones with the given shape.
        
        Args:
            shape: Dimensions of the array.
            dtype: Data type for the elements.
            
        Returns:
            New NArray instance filled with ones.
        """
        instance = cls(shape=shape, dtype=dtype, _internal_call=True)
        # Replace zeros with ones
        instance.data = instance._ones_internal(shape, dtype or float)
        return instance
    
    @classmethod
    def full(cls, shape: Tuple[int, ...], value: Any, dtype: Optional[type] = None) -> 'NArray':
        """
        Create an array with the given shape and fill it with the given value.
        
        Args:
            shape: Dimensions of the array.
            value: Value to fill the array with.
            dtype: Data type for the elements.
            
        Returns:
            New NArray instance filled with the given value.
        """
        # Ensure dtype has a default value based on the value's type
        dtype = dtype or type(value)
        
        instance = cls(shape=shape, dtype=dtype, _internal_call=True)
        instance.data = instance._full_internal(shape, value, dtype)
        return instance
    
    @classmethod
    def eye(cls, n: int, m: Optional[int] = None, k: int = 0, dtype: Optional[type] = None) -> 'NArray':
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
            New NArray instance with ones on the specified diagonal.
            
        Examples:
            eye(3) creates a 3x3 identity matrix.
            eye(3, 4) creates a 3x4 matrix with ones on the main diagonal.
            eye(3, 3, 1) creates a 3x3 matrix with ones on the first super-diagonal.
            eye(3, 3, -1) creates a 3x3 matrix with ones on the first sub-diagonal.
        """
        # Ensure dtype has a default value
        dtype = dtype or float
        
        instance = cls(_internal_call=True, dtype=dtype)
        instance.data = instance._eye_internal(n, m, k, dtype)
        instance._shape = (n, m if m is not None else n)
        instance._size = n * (m if m is not None else n)
        return instance
    
    @classmethod
    def identity(cls, n: int, dtype: Optional[type] = None) -> 'NArray':
        """
        Create a square identity matrix of size n x n.
        
        An identity matrix is a square matrix with ones on the main diagonal
        and zeros elsewhere.
        
        Args:
            n: Size of the identity matrix (n x n).
            dtype: Data type for the elements.
            
        Returns:
            New NArray instance representing an identity matrix.
            
        Example:
            identity(3) creates a 3x3 identity matrix:
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        """
        # Use the eye method with default parameters
        return cls.eye(n, dtype=dtype)
    
    @classmethod
    def uniform(cls, low: float, high: float, shape: Tuple[int, ...], 
                dtype: Optional[type] = None, 
                seed: Optional[int] = None, 
                decimals: Optional[int] = None) -> 'NArray':
        """
        Create an array filled with random numbers from a uniform distribution.
        
        Args:
            low: Lower bound of the distribution.
            high: Upper bound of the distribution.
            shape: Dimensions of the array.
            dtype: Data type for the elements.
            seed: Random seed for reproducibility.
            decimals: Number of decimal places to round to.
            
        Returns:
            New NArray instance filled with random numbers from a uniform distribution.
        """
        # Ensure dtype has a default value
        dtype = dtype or float
        
        instance = cls(shape=shape, dtype=dtype, _internal_call=True)
        instance.data = instance._uniform_internal(low, high, shape, seed, decimals)
        return instance
    
    @classmethod
    def normal(cls, loc: float, scale: float, shape: Tuple[int, ...], 
               dtype: Optional[type] = None, 
               seed: Optional[int] = None, 
               decimals: Optional[int] = None) -> 'NArray':
        """
        Create an array filled with random numbers from a normal (Gaussian) distribution.
        
        Args:
            loc: Mean of the distribution.
            scale: Standard deviation of the distribution.
            shape: Dimensions of the array.
            dtype: Data type for the elements.
            seed: Random seed for reproducibility.
            decimals: Number of decimal places to round to.
            
        Returns:
            New NArray instance filled with random numbers from a normal distribution.
        """
        # Ensure dtype has a default value
        dtype = dtype or float
        
        instance = cls(shape=shape, dtype=dtype, _internal_call=True)
        instance.data = instance._normal_internal(loc, scale, shape, seed, decimals)
        return instance
    
    @classmethod
    def random(cls, shape: Tuple[int, ...], 
               dtype: Optional[type] = None,
               seed: Optional[int] = None,
               decimals: Optional[int] = None) -> 'NArray':
        """
        Create an array filled with random numbers from a uniform distribution between 0 and 1.
        
        Args:
            shape: Dimensions of the array.
            dtype: Data type for the elements.
            seed: Random seed for reproducibility.
            decimals: Number of decimal places to round to.
            
        Returns:
            New NArray instance filled with random numbers from a uniform distribution.
        """
        # Ensure dtype has a default value
        dtype = dtype or float
        
        instance = cls(shape=shape, dtype=dtype, _internal_call=True)
        instance.data = instance._random_internal(shape, seed, decimals)
        return instance
    
    @classmethod
    def randint(cls, low: int, high: int, shape: Tuple[int, ...], 
                dtype: Optional[type] = None,
                seed: Optional[int] = None,
                decimals: Optional[int] = None) -> 'NArray':
        """
        Create an array filled with random integers from a uniform distribution.
        
        Args:
            low: Lower bound of the distribution (inclusive).
            high: Upper bound of the distribution (exclusive).
            shape: Dimensions of the array.
            dtype: Data type for the elements.
            seed: Random seed for reproducibility.
            decimals: Not used for integers, but kept for API consistency.
            
        Returns:
            New NArray instance filled with random integers from a uniform distribution.
        """
        # Ensure dtype has a default value, int is more appropriate for randint
        dtype = dtype or int
        
        instance = cls(shape=shape, dtype=dtype, _internal_call=True)
        instance.data = instance._randint_internal(low, high, shape, seed, decimals)
        return instance
    

    # Properties with getters - now properly implemented
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the array."""
        return self._shape
    
    @property
    def ndims(self) -> int:
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
            return f"NArray({self.data}, dtype={dtype_name})"
            
        # For 1D arrays
        if len(self._shape) == 1:
            return f"NArray({self.data}, dtype={dtype_name})"
            
        # For multi-dimensional arrays
        return f"NArray(shape={self._shape}, dtype={dtype_name})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        # Ensure dtype is not None before accessing __name__
        dtype_name = self._dtype.__name__ if self._dtype is not None else "None"
        
        if not self._shape or len(self._shape) == 0 or len(self._shape) == 1:  # Scalar or 1D
            return f"NArray({self.data}, dtype={dtype_name})"
            
        # For 2D arrays, format as a matrix
        if len(self._shape) == 2:
            rows = []
            for row in self.data:
                rows.append("[" + ", ".join(str(x) for x in row) + "]")
            matrix_str = "[" + ",\n ".join(rows) + "]"
            return f"NArray({matrix_str}, dtype={dtype_name})"
            
        # For higher dimensions, just show shape and type
        return f"NArray(shape={self._shape}, dtype={dtype_name})"