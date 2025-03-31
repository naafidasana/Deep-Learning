# Python Dunder Methods: The Complete Course

## Introduction

Welcome to this comprehensive course on Python's dunder methods (double underscore methods, also called "magic methods"). By the end of this course, you'll understand how to leverage these special methods to create your own powerful, intuitive data types with custom behaviors.

Dunder methods are the secret sauce that makes Python's built-in types feel so natural to use. They're what allows you to write `len(my_list)` instead of `my_list.get_length()` or `a + b` instead of `a.add(b)`. By implementing these methods in your own classes, you can create objects that work seamlessly with Python's syntax and built-in functions.

Let's begin our journey into Python's object model.

## Part 1: Understanding Dunder Methods

### What Are Dunder Methods?

Dunder methods are special methods in Python classes that begin and end with double underscores (`__`). They allow you to define how your objects behave in various contexts, from creation to destruction, representation, comparison, and much more.

```python
class MyClass:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"MyClass instance with value: {self.value}"
```

In this example, `__init__` and `__str__` are dunder methods that define initialization and string representation behaviors.

### The Python Data Model

Python's data model is built around the concept of protocols. A protocol is a set of dunder methods that define how objects of a class behave. By implementing specific dunder methods, your class can participate in Python's various protocols:

- **Object protocol**: Basic object behavior (`__init__`, `__str__`, etc.)
- **Sequence protocol**: Makes your object act like a sequence (`__len__`, `__getitem__`, etc.)
- **Numeric protocol**: Allows numeric operations (`__add__`, `__mul__`, etc.)
- **Context manager protocol**: Enables `with` statement usage (`__enter__`, `__exit__`)

The beauty of Python's approach is that you only need to implement the methods relevant to your class. If you don't need your object to be multipliable, you can simply skip implementing `__mul__`.

### Why Dunder Methods Matter

Dunder methods allow your custom classes to:

1. **Integrate seamlessly** with Python's syntax
2. **Behave consistently** with built-in types
3. **Support operator overloading** (using +, -, *, etc. with your objects)
4. **Customize object behavior** at a fundamental level
5. **Create intuitive interfaces** for your class users

## Part 2: Basic Object Dunder Methods

### Object Creation and Lifecycle

#### `__new__(cls, *args, **kwargs)`

The `__new__` method is called before an object is created. It's a static method that takes the class as its first parameter and returns a new instance of the class.

```python
class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# Both variables reference the same object
a = Singleton()
b = Singleton()
print(a is b)  # True
```

#### `__init__(self, *args, **kwargs)`

The `__init__` method initializes a newly created object. It's called after `__new__` and is where you typically set up the initial state of the object.

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
```

#### `__del__(self)`

The `__del__` method is called when an object is about to be destroyed. It's used for cleanup operations.

```python
class FileHandler:
    def __init__(self, filename):
        self.file = open(filename, 'w')

    def __del__(self):
        print(f"Closing file: {self.file.name}")
        self.file.close()
```

### String Representation

#### `__str__(self)`

The `__str__` method returns a human-readable string representation of an object. It's called by the built-in `str()` function and when using `print()`.

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"Point({self.x}, {self.y})"
```

#### `__repr__(self)`

The `__repr__` method returns a string representation that should ideally be an expression that recreates the object. It's called by the built-in `repr()` function and in interactive mode.

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"
```

The difference between `__str__` and `__repr__` is that `__str__` is meant for users, while `__repr__` is meant for developers. A good rule is that `eval(repr(obj))` should create an equivalent object when possible.

### Object Comparison

#### `__eq__(self, other)` and `__ne__(self, other)`

These methods define equality (`==`) and inequality (`!=`) comparisons:

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __eq__(self, other):
        if not isinstance(other, Person):
            return NotImplemented
        return self.name == other.name and self.age == other.age

    def __ne__(self, other):
        # Python calls __eq__ and negates the result if __ne__ is not defined
        if not isinstance(other, Person):
            return NotImplemented
        return not (self == other)
```

#### Ordering Comparisons

Python provides four dunder methods for ordering comparisons:

```python
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius

    def __lt__(self, other):  # 
        if not isinstance(other, Temperature):
            return NotImplemented
        return self.celsius < other.celsius

    def __le__(self, other):  # <=
        if not isinstance(other, Temperature):
            return NotImplemented
        return self.celsius <= other.celsius

    def __gt__(self, other):  # >
        if not isinstance(other, Temperature):
            return NotImplemented
        return self.celsius > other.celsius

    def __ge__(self, other):  # >=
        if not isinstance(other, Temperature):
            return NotImplemented
        return self.celsius >= other.celsius
```

You can simplify implementation using the `functools.total_ordering` decorator, which requires only one of the comparison methods plus `__eq__`:

```python
from functools import total_ordering

@total_ordering
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius

    def __eq__(self, other):
        if not isinstance(other, Temperature):
            return NotImplemented
        return self.celsius == other.celsius

    def __lt__(self, other):
        if not isinstance(other, Temperature):
            return NotImplemented
        return self.celsius < other.celsius
```

### Boolean Conversion

#### `__bool__(self)`

The `__bool__` method defines the truthiness of an object when used in a boolean context (like an `if` statement).

```python
class DatabaseConnection:
    def __init__(self):
        self.connected = False

    def connect(self):
        # Connection logic here
        self.connected = True

    def disconnect(self):
        # Disconnection logic here
        self.connected = False

    def __bool__(self):
        return self.connected

# Usage
conn = DatabaseConnection()
conn.connect()
if conn:  # Uses __bool__
    print("Connection is active")
```

If `__bool__` is not defined, Python falls back to `__len__` (returning `True` if the length is non-zero). If neither method is defined, objects are considered `True` by default.

### Attribute Access

#### `__getattr__(self, name)`

Called when an attribute lookup fails. It's not called if the attribute exists in the instance dictionary or is found through normal attribute resolution.

```python
class DynamicAttributes:
    def __init__(self):
        self.existing_attr = "I exist"

    def __getattr__(self, name):
        return f"Dynamically generated value for {name}"

obj = DynamicAttributes()
print(obj.existing_attr)  # "I exist" (normal attribute access)
print(obj.nonexistent)    # "Dynamically generated value for nonexistent"
```

#### `__setattr__(self, name, value)`

Called when an attribute is set. Be careful with this one, as it's easy to create infinite recursion.

```python
class ValidatedAttributes:
    def __setattr__(self, name, value):
        if name == 'age' and not isinstance(value, int):
            raise TypeError("Age must be an integer")
        if name == 'name' and not isinstance(value, str):
            raise TypeError("Name must be a string")
        # This avoids recursion by using the superclass's __setattr__
        super().__setattr__(name, value)

person = ValidatedAttributes()
person.name = "Alice"  # OK
person.age = 30       # OK
# person.age = "thirty"  # Raises TypeError
```

#### `__delattr__(self, name)`

Called when an attribute is deleted using the `del` statement.

```python
class ProtectedAttributes:
    def __init__(self):
        self.normal_attr = "Can be deleted"
        self.protected_attr = "Cannot be deleted"

    def __delattr__(self, name):
        if name == 'protected_attr':
            raise AttributeError("Cannot delete protected_attr")
        super().__delattr__(name)

obj = ProtectedAttributes()
del obj.normal_attr  # OK
# del obj.protected_attr  # Raises AttributeError
```

## Part 3: Container Dunder Methods

### Basic Container Methods

#### `__len__(self)`

The `__len__` method defines the behavior of the built-in `len()` function when applied to your object.

```python
class Playlist:
    def __init__(self):
        self.songs = []

    def add_song(self, song):
        self.songs.append(song)

    def __len__(self):
        return len(self.songs)

playlist = Playlist()
playlist.add_song("Bohemian Rhapsody")
playlist.add_song("Stairway to Heaven")
print(len(playlist))  # 2
```

#### `__getitem__(self, key)`

The `__getitem__` method allows your object to use indexing syntax (`obj[key]`).

```python
class Playlist:
    def __init__(self):
        self.songs = []

    def add_song(self, song):
        self.songs.append(song)

    def __getitem__(self, index):
        return self.songs[index]

playlist = Playlist()
playlist.add_song("Bohemian Rhapsody")
playlist.add_song("Stairway to Heaven")
print(playlist[0])  # "Bohemian Rhapsody"
```

#### `__setitem__(self, key, value)`

The `__setitem__` method defines the behavior when an item is assigned using the subscript notation (`obj[key] = value`).

```python
class SparseArray:
    def __init__(self):
        self.data = {}

    def __setitem__(self, index, value):
        self.data[index] = value

    def __getitem__(self, index):
        return self.data.get(index, 0)

arr = SparseArray()
arr[1000000] = 42  # Only stores the non-zero elements
print(arr[1000000])  # 42
print(arr[0])       # 0 (default value)
```

#### `__delitem__(self, key)`

The `__delitem__` method defines the behavior when an item is deleted using the `del` statement and subscript notation.

```python
class SimpleDict:
    def __init__(self):
        self.data = {}

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __delitem__(self, key):
        del self.data[key]

d = SimpleDict()
d["key"] = "value"
print(d["key"])  # "value"
del d["key"]
# print(d["key"])  # Would raise KeyError
```

#### `__contains__(self, item)`

The `__contains__` method defines the behavior of the `in` operator.

```python
class Team:
    def __init__(self):
        self.members = []

    def add_member(self, name):
        self.members.append(name)

    def __contains__(self, member):
        return member in self.members

team = Team()
team.add_member("Alice")
team.add_member("Bob")
print("Alice" in team)  # True
print("Charlie" in team)  # False
```

### Advanced Container Methods

#### Supporting Iteration

To make your object iterable, implement `__iter__`.

```python
class Playlist:
    def __init__(self):
        self.songs = []

    def add_song(self, song):
        self.songs.append(song)

    def __iter__(self):
        return iter(self.songs)

playlist = Playlist()
playlist.add_song("Bohemian Rhapsody")
playlist.add_song("Stairway to Heaven")

for song in playlist:  # Uses __iter__
    print(song)
```

#### Supporting Slicing

The `__getitem__` method can also handle slice objects:

```python
class CustomList:
    def __init__(self, data):
        self.data = list(data)

    def __getitem__(self, index):
        if isinstance(index, slice):
            # Return a new CustomList for slices
            return CustomList(self.data[index])
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"CustomList({self.data})"

cl = CustomList([1, 2, 3, 4, 5])
print(cl[1:4])  # CustomList([2, 3, 4])
```

## Part 4: Numeric Dunder Methods

### Basic Arithmetic Operators

Python provides dunder methods for all arithmetic operations:

```python
class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        if isinstance(other, Vector2D):
            return Vector2D(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vector2D):
            return Vector2D(self.x - other.x, self.y - other.y)
        return NotImplemented

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vector2D(self.x * scalar, self.y * scalar)
        return NotImplemented

    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vector2D(self.x / scalar, self.y / scalar)
        return NotImplemented

    def __repr__(self):
        return f"Vector2D({self.x}, {self.y})"

# Usage
v1 = Vector2D(1, 2)
v2 = Vector2D(3, 4)
print(v1 + v2)      # Vector2D(4, 6)
print(v1 - v2)      # Vector2D(-2, -2)
print(v1 * 3)       # Vector2D(3, 6)
print(v1 / 2)       # Vector2D(0.5, 1.0)
```

### Reversed Operators

When Python evaluates `a + b`, it first tries `a.__add__(b)`. If that returns `NotImplemented`, it tries `b.__radd__(a)`. This allows right-side operations:

```python
class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vector2D(self.x * scalar, self.y * scalar)
        return NotImplemented

    def __rmul__(self, scalar):
        # Called when left operand doesn't support the operation
        # For example: 3 * vector
        return self.__mul__(scalar)

    def __repr__(self):
        return f"Vector2D({self.x}, {self.y})"

v = Vector2D(1, 2)
print(v * 3)  # Uses __mul__: Vector2D(3, 6)
print(3 * v)  # Uses __rmul__: Vector2D(3, 6)
```

### In-place Operators

In-place operators modify the object directly and (usually) return `self`:

```python
class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __iadd__(self, other):
        # In-place addition (+=)
        if isinstance(other, Vector2D):
            self.x += other.x
            self.y += other.y
            return self
        return NotImplemented

    def __isub__(self, other):
        # In-place subtraction (-=)
        if isinstance(other, Vector2D):
            self.x -= other.x
            self.y -= other.y
            return self
        return NotImplemented

    def __repr__(self):
        return f"Vector2D({self.x}, {self.y})"

v1 = Vector2D(1, 2)
v2 = Vector2D(3, 4)
v1 += v2  # Uses __iadd__
print(v1)  # Vector2D(4, 6)
```

### Unary Operators

Python also supports unary operators:

```python
class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __neg__(self):
        # Unary negation: -v
        return Vector2D(-self.x, -self.y)

    def __pos__(self):
        # Unary plus: +v
        return Vector2D(+self.x, +self.y)

    def __abs__(self):
        # Absolute value: abs(v)
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __repr__(self):
        return f"Vector2D({self.x}, {self.y})"

v = Vector2D(3, 4)
print(-v)      # Vector2D(-3, -4)
print(+v)      # Vector2D(3, 4)
print(abs(v))  # 5.0
```

### Type Conversion Methods

These dunder methods allow conversion between types:

```python
class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __int__(self):
        # Called by int()
        return int(abs(self))

    def __float__(self):
        # Called by float()
        return float(abs(self))

    def __complex__(self):
        # Called by complex()
        return complex(self.x, self.y)

    def __abs__(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __repr__(self):
        return f"Vector2D({self.x}, {self.y})"

v = Vector2D(3, 4)
print(int(v))      # 5
print(float(v))    # 5.0
print(complex(v))  # (3+4j)
```

## Part 5: Advanced Dunder Methods

### Callable Objects with `__call__`

The `__call__` method allows instances of a class to be called like functions:

```python
class Multiplier:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return x * self.factor

double = Multiplier(2)
triple = Multiplier(3)

print(double(5))  # 10
print(triple(5))  # 15
```

This is powerful for creating function-like objects that maintain state.

### Context Managers with `__enter__` and `__exit__`

Context managers support the `with` statement:

```python
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        # Return True to suppress exceptions, False to propagate them
        return False

# Usage
with FileManager('test.txt', 'w') as f:
    f.write('Hello, World!')
# File is automatically closed after the with block
```

### Descriptor Protocol with `__get__`, `__set__`, and `__delete__`

Descriptors control attribute access at the class level:

```python
class Validator:
    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        self.name = None

    def __set_name__(self, owner, name):
        # Called when the descriptor is assigned to a class attribute
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if not isinstance(value, int):
            raise TypeError(f"{self.name} must be an integer")
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name} must be >= {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name} must be <= {self.max_value}")
        instance.__dict__[self.name] = value

class Person:
    age = Validator(min_value=0, max_value=150)

    def __init__(self, name, age):
        self.name = name
        self.age = age  # This will use the descriptor's __set__

# Usage
person = Person("Alice", 30)  # OK
# person.age = -1  # Raises ValueError
# person.age = 200  # Raises ValueError
# person.age = "thirty"  # Raises TypeError
```

### Iterator Protocol with `__iter__` and `__next__`

The iterator protocol allows custom iteration behavior:

```python
class Countdown:
    def __init__(self, start):
        self.start = start

    def __iter__(self):
        # Return an iterator (which happens to be self)
        self.current = self.start
        return self

    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

# Usage
for num in Countdown(5):
    print(num)  # Prints 5, 4, 3, 2, 1
```

### Hashing and Dictionary Keys with `__hash__`

The `__hash__` method allows your objects to be used as dictionary keys or in sets:

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        # Objects that are equal must have the same hash
        return hash((self.x, self.y))

# Usage
p1 = Point(1, 2)
p2 = Point(1, 2)
p3 = Point(3, 4)

point_data = {p1: "First point", p3: "Third point"}
print(point_data[p2])  # "First point" (p2 is equal to p1)
```

Note: If you define `__eq__` but not `__hash__`, your objects will be unhashable by default.

## Part 6: Building a Vector Class

Now let's apply what we've learned to build a full-featured Vector class:

```python
import math
import numbers

class Vector:
    def __init__(self, *components):
        # Store components in a tuple to make the Vector immutable
        self._components = tuple(components)

    @property
    def components(self):
        return self._components

    def __repr__(self):
        return f"Vector{self.components}"

    def __str__(self):
        return str(self.components)

    def __len__(self):
        return len(self._components)

    def __getitem__(self, index):
        return self._components[index]

    def __eq__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return self.components == other.components

    def __hash__(self):
        return hash(self.components)

    def __add__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("Vectors must have the same dimension")
        return Vector(*(a + b for a, b in zip(self, other)))

    def __sub__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("Vectors must have the same dimension")
        return Vector(*(a - b for a, b in zip(self, other)))

    def __mul__(self, scalar):
        if not isinstance(scalar, numbers.Number):
            return NotImplemented
        return Vector(*(component * scalar for component in self))

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        if not isinstance(scalar, numbers.Number):
            return NotImplemented
        return Vector(*(component / scalar for component in self))

    def __neg__(self):
        return Vector(*(-component for component in self))

    def __abs__(self):
        return math.sqrt(sum(component ** 2 for component in self))

    def dot(self, other):
        """Calculate the dot product with another vector."""
        if not isinstance(other, Vector):
            raise TypeError("Dot product requires another Vector")
        if len(self) != len(other):
            raise ValueError("Vectors must have the same dimension")
        return sum(a * b for a, b in zip(self, other))

    def __matmul__(self, other):
        """Implement the @ operator for dot product (Python 3.5+)."""
        return self.dot(other)

# Usage examples
v1 = Vector(1, 2, 3)
v2 = Vector(4, 5, 6)

print(v1 + v2)      # Vector(5, 7, 9)
print(v1 - v2)      # Vector(-3, -3, -3)
print(v1 * 2)       # Vector(2, 4, 6)
print(v1 / 2)       # Vector(0.5, 1.0, 1.5)
print(abs(v1))      # 3.7416573867739413 (magnitude)
print(v1 @ v2)      # 32 (dot product using @)
print(v1.dot(v2))   # 32 (dot product using method)
```

## Part 7: Building a Matrix Class

Let's create a Matrix class that works with our Vector class:

```python
class Matrix:
    def __init__(self, rows):
        """Initialize a matrix from a list of lists (rows)."""
        # Validate input
        if not rows:
            raise ValueError("Matrix cannot be empty")

        # Check if all rows have the same length
        row_length = len(rows[0])
        if any(len(row) != row_length for row in rows):
            raise ValueError("All rows must have the same length")

        # Store data as a tuple of tuples for immutability
        self._rows = tuple(tuple(row) for row in rows)
        self._shape = (len(rows), row_length)

    @property
    def shape(self):
        """Return the shape of the matrix as (rows, columns)."""
        return self._shape

    @property
    def rows(self):
        """Return the number of rows."""
        return self._shape[0]

    @property
    def columns(self):
        """Return the number of columns."""
        return self._shape[1]

    def __repr__(self):
        return f"Matrix({self._rows})"

    def __str__(self):
        return '\n'.join(' '.join(f"{val:8.3f}" for val in row) for row in self._rows)

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            # Handle (row, col) indexing
            row, col = key
            return self._rows[row][col]
        # Handle single index (return a row)
        return self._rows[key]

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        return self._rows == other._rows

    def __add__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Matrices must have the same shape")

        result = []
        for i in range(self.rows):
            result.append([self[i, j] + other[i, j] for j in range(self.columns)])

        return Matrix(result)

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Matrices must have the same shape")

        result = []
        for i in range(self.rows):
            result.append([self[i, j] - other[i, j] for j in range(self.columns)])

        return Matrix(result)

    def __mul__(self, scalar):
        """Scalar multiplication."""
        if not isinstance(scalar, (int, float)):
            return NotImplemented

        result = []
        for i in range(self.rows):
            result.append([self[i, j] * scalar for j in range(self.columns)])

        return Matrix(result)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __matmul__(self, other):
        """Matrix multiplication using @ operator."""
        if not isinstance(other, Matrix):
            return NotImplemented

        if self.columns != other.rows:
            raise ValueError(f"Matrix shapes incompatible for multiplication: {self.shape} and {other.shape}")

        result = []
        for i in range(self.rows):
            result_row = []
            for j in range(other.columns):
                # Compute the dot product of row i and column j
                dot_product = sum(self[i, k] * other[k, j] for k in range(self.columns))
                result_row.append(dot_product)
            result.append(result_row)

        return Matrix(result)

    def transpose(self):
        """Return the transpose of the matrix."""
        result = []
        for j in range(self.columns):
            result.append([self[i, j] for i in range(self.rows)])

        return Matrix(result)

    def __pow__(self, power):
        """Compute matrix to an integer power."""
        if not isinstance(power, int):
            return NotImplemented
        if power < 0:
            raise ValueError("Negative powers require matrix inversion, not implemented")
        if power == 0:
            # Return identity matrix of the same size
            return Matrix([[1 if i == j else 0 for j in range(self.columns)] for i in range(self.rows)])
        if power == 1:
            return Matrix(self._rows)

        # Use square-and-multiply for efficient exponentiation
        result = self
        for _ in range(power - 1):
            result = result @ self

        return result

# Usage examples
m1 = Matrix([[1, 2], [3, 4]])
m2 = Matrix([[5, 6], [7, 8]])

print(m1 + m2)      # Element-wise addition
print(m1 - m2)      # Element-wise subtraction
print(m1 * 2)       # Scalar multiplication
print(m1 @ m2)      # Matrix multiplication
print(m1.transpose())  # Transpose
print(m1 ** 2)      # Matrix squared (m1 @ m1)
```

## Part 8: Building a Custom NumPy-like Array

Now, let's create a more sophisticated array class that mimics some of the functionality of NumPy's ndarray:

```python
import math
import numbers
import itertools

class Array:
    def __init__(self, data, shape=None):
        """
        Initialize an Array from a nested list or flat data with a shape.

        Args:
            data: Nested list or flat iterable
            shape: Optional shape tuple
        """
        if shape is None:
            # Infer shape from nested data structure
            self._data, self._shape = self._process_nested_data(data)
        else:
            # Flat data with explicit shape
            self._data = list(data)
            if math.prod(shape) != len(self._data):
                raise ValueError(f"Data size {len(self._data)} doesn't match shape {shape}")
            self._shape = tuple(shape)

    @staticmethod
    def _process_nested_data(data):
        """Convert nested lists to flat data and infer shape."""
        if not isinstance(data, (list, tuple)):
            return [data], (1,)

        # Check if this is a 1D list
        if not data or not isinstance(data[0], (list, tuple)):
            return list(data), (len(data),)

        # Process multi-dimensional data
        shape = [len(data)]
        first_child_shape = Array._process_nested_data(data[0])[1]
        shape.extend(first_child_shape)

        # Validate all nested lists have the same shape
        flat_data = []
        for item in data:
            item_data, item_shape = Array._process_nested_data(item)
            if item_shape != first_child_shape:
                raise ValueError("Inconsistent shape in nested data")
            flat_data.extend(item_data)

        return flat_data, tuple(shape)

    @property
    def shape(self):
        """Return the shape of the array."""
        return self._shape

    @property
    def ndim(self):
        """Return the number of dimensions."""
        return len(self._shape)

    @property
    def size(self):
        """Return the total number of elements."""
        return math.prod(self._shape)

    def _compute_index(self, indices):
        """Convert n-dimensional indices to a flat index."""
        if len(indices) != self.ndim:
            raise IndexError(f"Expected {self.ndim} indices, got {len(indices)}")

        for i, (idx, dim_size) in enumerate(zip(indices, self._shape)):
            if idx < 0 or idx >= dim_size:
                raise IndexError(f"Index {idx} out of bounds for dimension {i} with size {dim_size}")

        # Compute flat index using stride-based indexing
        strides = [math.prod(self._shape[i+1:]) for i in range(self.ndim)]
        flat_idx = sum(idx * stride for idx, stride in zip(indices, strides))
        return flat_idx

    def __getitem__(self, indices):
        """Get an item or a slice of items."""
        # Handle tuple indexing
        if isinstance(indices, tuple):
            # Check for slices
            if any(isinstance(idx, slice) for idx in indices):
                return self._get_slice(indices)

            # Regular indexing
            flat_idx = self._compute_index(indices)
            return self._data[flat_idx]

        # Handle single index or slice
        if isinstance(indices, slice):
            return self._get_slice((indices,))

        # Single integer index for the first dimension
        if self.ndim == 1:
            return self._data[indices]

        # Return a slice of the array (one less dimension)
        start_idx = indices * math.prod(self._shape[1:])
        end_idx = (indices + 1) * math.prod(self._shape[1:])
        return Array(self._data[start_idx:end_idx], self._shape[1:])

    def _get_slice(self, slice_indices):
        """Handle slicing operations."""
        # First, normalize the slice indices to full slices for all dimensions
        full_slices = []
        for i, idx in enumerate(slice_indices):
            if i >= self.ndim:
                break

            if isinstance(idx, slice):
                # Normalize the slice
                start = 0 if idx.start is None else idx.start
                stop = self._shape[i] if idx.stop is None else idx.stop
                step = 1 if idx.step is None else idx.step
                full_slices.append(slice(start, stop, step))
            else:
                # Convert integer index to single-element slice
                full_slices.append(slice(idx, idx + 1, 1))

        # Add full slices for remaining dimensions
        for i in range(len(full_slices), self.ndim):
            full_slices.append(slice(0, self._shape[i], 1))

        # Calculate the resulting shape and determine indices to extract
        result_shape = []
        for i, s in enumerate(full_slices):
            indices_in_dim = range(s.start, min(s.stop, self._shape[i]), s.step)
            size = len(list(indices_in_dim))
            if size > 0:
                result_shape.append(size)

        # Create all possible indices combinations
        indices_ranges = [range(s.start, min(s.stop, self._shape[i]), s.step) 
                          for i, s in enumerate(full_slices)]

        if not indices_ranges:
            return Array([], (0,))

        all_indices = list(itertools.product(*indices_ranges))

        # Extract the data
        result_data = [self[idx] for idx in all_indices]

        # If result is a scalar (empty shape), just return the value
        if not result_shape:
            return result_data[0] if result_data else None

        return Array(result_data, result_shape)

    def __setitem__(self, indices, value):
        """Set an item or slice of items."""
        # Handle tuple indexing
        if isinstance(indices, tuple):
            # Check for slices
            if any(isinstance(idx, slice) for idx in indices):
                self._set_slice(indices, value)
                return

            # Regular indexing
            flat_idx = self._compute_index(indices)
            self._data[flat_idx] = value
            return

        # Handle single index or slice
        if isinstance(indices, slice):
            self._set_slice((indices,), value)
            return

        # Single integer index
        if self.ndim == 1:
            self._data[indices] = value
            return

        # Set a slice of the array
        if not isinstance(value, Array):
            # Broadcast scalar value
            start_idx = indices * math.prod(self._shape[1:])
            end_idx = (indices + 1) * math.prod(self._shape[1:])
            self._data[start_idx:end_idx] = [value] * (end_idx - start_idx)
        else:
            # Check shape compatibility
            if value.shape != self._shape[1:]:
                raise ValueError(f"Cannot set array with shape {value.shape} "
                                 f"to slice with shape {self._shape[1:]}")

            start_idx = indices * math.prod(self._shape[1:])
            self._data[start_idx:start_idx + value.size] = value._data

    def _set_slice(self, slice_indices, value):
        """Handle setting slices."""
        # First normalize slice indices (similar to _get_slice)
        # For simplicity, we'll just get the slice and then set each element
        sliced_array = self._get_slice(slice_indices)

        # If value is a scalar, broadcast it
        if not isinstance(value, Array):
            for indices in itertools.product(*(range(dim) for dim in sliced_array.shape)):
                full_indices = self._map_slice_indices(slice_indices, indices)
                self[full_indices] = value
        else:
            # Check shape compatibility
            if value.shape != sliced_array.shape:
                raise ValueError(f"Cannot set array with shape {value.shape} "
                                 f"to slice with shape {sliced_array.shape}")

            # Set each element
            for indices in itertools.product(*(range(dim) for dim in sliced_array.shape)):
                full_indices = self._map_slice_indices(slice_indices, indices)
                value_indices = indices[:value.ndim]
                self[full_indices] = value[value_indices]

    def _map_slice_indices(self, slice_indices, rel_indices):
        """Map relative indices in a slice to absolute indices in the array."""
        full_indices = []
        rel_idx_pos = 0

        for i, idx in enumerate(slice_indices):
            if i >= self.ndim:
                break

            if isinstance(idx, slice):
                # Calculate the absolute index from the relative one
                start = 0 if idx.start is None else idx.start
                step = 1 if idx.step is None else idx.step
                abs_idx = start + rel_indices[rel_idx_pos] * step
                full_indices.append(abs_idx)
                rel_idx_pos += 1
            else:
                # Integer index, just use it directly
                full_indices.append(idx)

        # Add remaining relative indices
        for i in range(rel_idx_pos, len(rel_indices)):
            full_indices.append(rel_indices[i])

        return tuple(full_indices)

    def __eq__(self, other):
        if not isinstance(other, Array):
            return NotImplemented
        return self._shape == other._shape and self._data == other._data

    def __repr__(self):
        return f"Array(shape={self._shape}, data={self._data})"

    def __str__(self):
        if self.ndim == 0:
            return str(self._data[0])

        if self.ndim == 1:
            return str(self._data)

        # For multi-dimensional arrays, format in a structured way
        if self.ndim == 2:
            rows = []
            for i in range(self._shape[0]):
                row_data = self._data[i * self._shape[1]:(i + 1) * self._shape[1]]
                rows.append("[" + ", ".join(map(str, row_data)) + "]")
            return "[" + ",\n ".join(rows) + "]"

        # For higher dimensions, just show shape and first few elements
        return f"Array(shape={self._shape}, data={self._data[:min(10, len(self._data))]}...)"

    def __add__(self, other):
        if isinstance(other, numbers.Number):
            # Scalar addition
            return Array([x + other for x in self._data], self._shape)

        if not isinstance(other, Array):
            return NotImplemented

        if self._shape != other._shape:
            raise ValueError(f"Cannot add arrays with shapes {self._shape} and {other._shape}")

        return Array([a + b for a, b in zip(self._data, other._data)], self._shape)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, numbers.Number):
            # Scalar subtraction
            return Array([x - other for x in self._data], self._shape)

        if not isinstance(other, Array):
            return NotImplemented

        if self._shape != other._shape:
            raise ValueError(f"Cannot subtract arrays with shapes {self._shape} and {other._shape}")

        return Array([a - b for a, b in zip(self._data, other._data)], self._shape)

    def __rsub__(self, other):
        if isinstance(other, numbers.Number):
            # Scalar subtraction (reversed)
            return Array([other - x for x in self._data], self._shape)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            # Scalar multiplication
            return Array([x * other for x in self._data], self._shape)

        if not isinstance(other, Array):
            return NotImplemented

        if self._shape != other._shape:
            raise ValueError(f"Cannot multiply arrays with shapes {self._shape} and {other._shape}")

        return Array([a * b for a, b in zip(self._data, other._data)], self._shape)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        """Matrix multiplication (@ operator)."""
        if not isinstance(other, Array):
            return NotImplemented

        # Check dimensions for matrix multiplication
        if self.ndim < 1 or other.ndim < 1:
            raise ValueError("Arrays must have at least 1 dimension for matrix multiplication")

        # For simplicity, implement only 2D matrix multiplication
        if self.ndim != 2 or other.ndim != 2:
            raise NotImplementedError("Only 2D matrix multiplication is implemented")

        if self._shape[1] != other._shape[0]:
            raise ValueError(f"Incompatible shapes for matrix multiplication: {self._shape} and {other._shape}")

        # Perform matrix multiplication
        result_shape = (self._shape[0], other._shape[1])
        result_data = []

        for i in range(self._shape[0]):
            for j in range(other._shape[1]):
                # Compute the dot product of row i and column j
                dot_product = sum(self[i, k] * other[k, j] for k in range(self._shape[1]))
                result_data.append(dot_product)

        return Array(result_data, result_shape)

    def transpose(self):
        """Return the transpose of the array."""
        if self.ndim < 2:
            return Array(self._data, self._shape)

        # For simplicity, implement only 2D transpose
        if self.ndim != 2:
            raise NotImplementedError("Only 2D transpose is implemented")

        result_shape = (self._shape[1], self._shape[0])
        result_data = []

        for j in range(self._shape[1]):
            for i in range(self._shape[0]):
                result_data.append(self[i, j])

        return Array(result_data, result_shape)

    # Additional methods can be added as needed

# Usage examples
arr1 = Array([[1, 2, 3], [4, 5, 6]])
arr2 = Array([[7, 8, 9], [10, 11, 12]])

print(arr1.shape)  # (2, 3)
print(arr1[0, 1])  # 2
print(arr1[0])     # Array with shape (3,) containing [1, 2, 3]
print(arr1 + arr2)  # Element-wise addition
print(arr1 * 2)     # Scalar multiplication

# Slicing
print(arr1[:, 1])   # Column 1: Array with shape (2,) containing [2, 5]
print(arr1[0, :])   # Row 0: Array with shape (3,) containing [1, 2, 3]

# Matrix multiplication
arr3 = Array([[1, 2], [3, 4], [5, 6]])  # 3x2 matrix
print(arr1 @ arr3)  # Result should be 2x2 matrix
```

## Conclusion

Throughout this course, we've explored Python's dunder methods and how they can be used to create powerful, intuitive custom data types. We've progressed from basic dunder methods to implementing sophisticated classes that mimic NumPy's functionality.

The examples we've built demonstrate how dunder methods enable:

1. **Custom behavior**: Making your objects respond to operators and built-in functions
2. **Pythonic interfaces**: Creating APIs that feel natural to Python programmers
3. **Domain-specific abstractions**: Building types that model your problem domain

By mastering dunder methods, you've gained the ability to create rich, expressive APIs that leverage Python's syntax and built-in functions effectively. Whether you're building scientific computing tools, domain-specific languages, or custom collections, these techniques will serve you well.

Remember that the most Pythonic code often uses dunder methods sparingly and with purpose. Always ask whether a dunder method truly improves your interface before implementing it.

Happy coding!


