from typing import Union


class NArray:
    def __init__(
        self,
        data: list | None = None,
        shape: tuple | None = None,
        dtype: Union[float, int] = None,
    ) -> None:
        assert data is not None or shape is not None, (
            "You must provide either data or shape"
        )
        self.dtype = dtype
        if not data:
            self.data = self._zeros(shape, dtype)
            self._shape = shape

        else:
            self.data = data if dtype is None else self._cast_to_type(data, dtype)
            self._shape = self._compute_shape(data)
            self.dtype = dtype if dtype else self._infer_dtype(data)

    def array(self, array: list) -> None:
        pass

    def _zeros(self, shape: tuple, dtype: Union[int, float]):
        if len(shape) == 1:
            if shape[0] == 0:
                return []
            return [dtype(0)] * shape[0]
        return [self._zeros(shape[1:]) for _ in shape]

    def zeros(self, shape, dtype: Union[int, float]):
        return self._zeros(shape)
    
    def _compute_shape(self, data):
        if not data:
            return (0,)
        if not isinstance(data, list):
            return ()
        return (len(data), ) + (self._compute_shape(data[0]))
    
    def _infer_dtype(self, data):
        if isinstance(data, list):
            return self._infer_dtype(data[0])
        return type(data)
    
    def _cast_to_type(self, data, dtype):
        if isinstance(data, list):
            return [self._cast_to_type(item, dtype) for item in data]
        return dtype(data) 

    def __repr__(self):
        return f"{self.data} dtype=float64"
    
    def __str__(self):
        return f"(Narray, {self.data}\ndtype={self.dtype})"
    
    @property
    def shape(self):
        return self._shape


if __name__ == "__main__":
    #new = NArray(shape=(2, 3, 4))
    new = NArray(data=[1., 2.], dtype=int)
    print(new)
    print(new.shape)