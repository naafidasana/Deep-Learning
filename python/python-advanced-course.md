# Advanced Python Built-in Features: A Comprehensive Course

## Module 1: Advanced Iteration & Built-in Functions

### 1.1 Understanding `enumerate()`

`enumerate()` is a built-in function that adds a counter to an iterable and returns it as an enumerate object, which can then be used in loops to get both the index and value simultaneously.

#### How `enumerate()` Works Internally

Under the hood, `enumerate()` creates an iterator that yields pairs of (index, value). Its simplified implementation would look something like:

```python
def my_enumerate(iterable, start=0):
    count = start
    for element in iterable:
        yield count, element
        count += 1
```

When Python evaluates:
```python
for i, value in enumerate(some_iterable):
    # do something
```

It's actually creating an iterator that produces tuples, and the loop unpacks these tuples into the variables `i` and `value`.

#### Why and When to Use `enumerate()`

Using `enumerate()` offers several advantages over manual indexing:

1. **Code Readability**: It makes your intention clear - you're working with both indices and values.
2. **Error Prevention**: It eliminates off-by-one errors that can occur with manual counters.
3. **Performance**: It's more efficient than maintaining a separate counter variable.
4. **Pythonic Style**: It follows Python's philosophy of explicit, clean code.

#### Practical Examples

**Example 1: Basic usage**

```python
fruits = ['apple', 'banana', 'cherry']

# Without enumerate
for i in range(len(fruits)):
    print(f"Index {i}: {fruits[i]}")

# With enumerate
for i, fruit in enumerate(fruits):
    print(f"Index {i}: {fruit}")
```

**Example 2: Starting from a different index**

```python
# Start counting from 1 instead of 0
for i, fruit in enumerate(fruits, 1):
    print(f"Fruit #{i}: {fruit}")
```

**Example 3: Working with data science applications**

```python
import numpy as np

# When processing time series data
time_series = np.array([10.2, 15.1, 12.3, 11.0, 9.8])
anomalies = []

for timestamp, value in enumerate(time_series):
    if value > 15 or value < 10:
        anomalies.append((timestamp, value))
        
print(f"Anomalies detected at timestamps: {anomalies}")
```

**Example 4: Combining with other iterables**

```python
names = ['Alice', 'Bob', 'Charlie']
scores = [85, 92, 78]

for i, (name, score) in enumerate(zip(names, scores), 1):
    print(f"Student #{i}: {name} scored {score}")
```

#### Best Practices

1. **Start parameter**: Use the `start` parameter when you need to begin counting from a value other than 0.
2. **Tuple unpacking**: Unpack the tuple returned by `enumerate()` directly in the `for` statement for cleaner code.
3. **Avoid list conversion**: Don't convert `enumerate()` to a list unless necessary, as it defeats the purpose of lazy iteration.
4. **Use with `zip()`**: Combine with `zip()` when you need to iterate over multiple sequences with indices.

### 1.2 Mastering `zip()`

`zip()` is a powerful built-in function that allows you to combine multiple iterables into a single iterable of tuples, where each tuple contains one element from each input iterable.

#### Internal Mechanism of `zip()`

The `zip()` function works by creating an iterator that aggregates elements from each of the iterables. It stops when the shortest input iterable is exhausted. A simplified implementation might look like:

```python
def my_zip(*iterables):
    # Get iterators from all iterables
    iterators = [iter(iterable) for iterable in iterables]
    while iterators:
        result = []
        for iterator in iterators:
            try:
                result.append(next(iterator))
            except StopIteration:
                return  # One iterator is exhausted, so we're done
        yield tuple(result)
```

#### Key Features and Use Cases

1. **Parallel Iteration**: Iterate through multiple sequences simultaneously.
2. **Data Transformation**: Transform data structures from one form to another.
3. **Matching Related Items**: Pair related items from different collections.
4. **Creating Dictionaries**: Combine two lists into a dictionary.
5. **Matrix Transposition**: Transpose rows and columns in a matrix.

#### Practical Examples

**Example 1: Basic Parallel Iteration**

```python
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
cities = ['New York', 'Boston', 'Chicago']

for name, age, city in zip(names, ages, cities):
    print(f"{name} is {age} years old and lives in {city}.")
```

**Example 2: Handling Unequal Length Iterables**

```python
# Python 3.10+ introduced zip() with the strict parameter
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30]

# Regular zip stops at the shortest iterable (2 iterations)
for name, age in zip(names, ages):
    print(f"{name} is {age} years old.")

# Using strict=True raises an exception if lengths differ
try:
    for name, age in zip(names, ages, strict=True):
        print(f"{name} is {age} years old.")
except ValueError as e:
    print(f"Error: {e}")
```

**Example 3: Creating a Dictionary**

```python
keys = ['name', 'age', 'job']
values = ['Alice', 30, 'Data Scientist']

person = dict(zip(keys, values))
print(person)  # {'name': 'Alice', 'age': 30, 'job': 'Data Scientist'}
```

**Example 4: Matrix Transposition**

```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Transpose using zip (converts rows to columns)
transposed = list(zip(*matrix))
print(transposed)  # [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
```

**Example 5: Data Science Application - Feature-Label Splitting**

```python
# For machine learning data preparation
dataset = [
    [1.2, 0.5, 0.1, 'A'],
    [2.3, 1.1, 0.5, 'B'],
    [0.7, 0.9, 1.3, 'A'],
    [1.5, 2.1, 0.3, 'C']
]

# Unzip the data into features and labels
features, labels = zip(*[(row[:-1], row[-1]) for row in dataset])
print("Features:", features)
print("Labels:", labels)
```

#### Advanced Usage: `zip_longest` from `itertools`

When you need to iterate beyond the shortest iterable, use `zip_longest`:

```python
from itertools import zip_longest

names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30]

# Fill missing values with 'Unknown'
for name, age in zip_longest(names, ages, fillvalue='Unknown'):
    print(f"{name} is {age} years old.")
```

#### Best Practices

1. **Be aware of stopping behavior**: Standard `zip()` stops at the shortest iterable.
2. **Use `strict=True`** (Python 3.10+): When iterables must be of equal length.
3. **Use `zip_longest`**: When you need to process all elements from all iterables.
4. **Avoid materializing large zips**: Avoid converting to list for large data unless necessary.
5. **Remember unzipping syntax**: Use `zip(*zipped_data)` to "unzip" data.

### 1.3 How `isinstance()` and `issubclass()` Work Internally

These two built-in functions are crucial for type checking and object-oriented programming in Python.

#### Understanding `isinstance(obj, classinfo)`

`isinstance()` checks if an object is an instance of a class or a subclass thereof.

##### Internal Mechanism

The internal implementation of `isinstance()` is complex, but at a high level:

1. If `classinfo` is a tuple, it checks if `obj` is an instance of any class in the tuple.
2. Otherwise, it checks if `obj.__class__` is the same as or a subclass of `classinfo`.
3. It uses the `__instancecheck__` special method of the class's metaclass if available.

A simplified version might look like:

```python
def my_isinstance(obj, classinfo):
    # Handle tuple case
    if isinstance(classinfo, tuple):
        return any(my_isinstance(obj, cls) for cls in classinfo)
    
    # Get obj's class
    obj_class = obj.__class__
    
    # Handle metaclass with __instancecheck__
    if hasattr(classinfo, '__instancecheck__'):
        return classinfo.__instancecheck__(obj)
        
    # Check if obj_class is classinfo or a subclass
    return obj_class is classinfo or my_issubclass(obj_class, classinfo)
```

#### Understanding `issubclass(cls, classinfo)`

`issubclass()` checks if a class is a subclass of another class (or any class in a tuple of classes).

##### Internal Mechanism

The implementation is similar to `isinstance()` but works with class relationships:

1. If `classinfo` is a tuple, it checks if `cls` is a subclass of any class in the tuple.
2. Otherwise, it checks the method resolution order (MRO) of `cls` to see if `classinfo` is in it.
3. It uses the `__subclasscheck__` special method if available.

A simplified version:

```python
def my_issubclass(cls, classinfo):
    # Handle tuple case
    if isinstance(classinfo, tuple):
        return any(my_issubclass(cls, c) for c in classinfo)
    
    # Handle metaclass with __subclasscheck__
    if hasattr(classinfo, '__subclasscheck__'):
        return classinfo.__subclasscheck__(cls)
        
    # Check cls's MRO (Method Resolution Order)
    return classinfo in cls.__mro__
```

#### Practical Examples

**Example 1: Basic Type Checking**

```python
# Basic type checking
value = 42
print(isinstance(value, int))       # True
print(isinstance(value, (int, float)))  # True - value is an int
print(isinstance(value, str))       # False

# Class hierarchy checking
class Animal: pass
class Dog(Animal): pass

fido = Dog()
print(isinstance(fido, Dog))     # True
print(isinstance(fido, Animal))  # True - Dog is a subclass of Animal
```

**Example 2: Working with Abstract Base Classes (ABCs)**

```python
from collections.abc import Sequence, Mapping

# Lists are Sequences
print(isinstance([], Sequence))  # True

# Dictionaries are Mappings
print(isinstance({}, Mapping))   # True
print(isinstance({}, Sequence))  # False

# ABC relationship checking
print(issubclass(list, Sequence))  # True
print(issubclass(dict, Mapping))   # True
```

**Example 3: Custom Type Checking**

```python
class Shape:
    pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

def calculate_area(shape):
    if not isinstance(shape, Shape):
        raise TypeError("Input must be a Shape")
        
    if isinstance(shape, Circle):
        return 3.14159 * shape.radius ** 2
    elif isinstance(shape, Rectangle):
        return shape.width * shape.height
    else:
        raise NotImplementedError("Area calculation not implemented for this shape")
```

**Example 4: Duck Typing vs. Type Checking**

```python
# Duck typing - focusing on behavior rather than type
def process_sequence(seq):
    try:
        return [x * 2 for x in seq]
    except TypeError:
        return "Input is not iterable"

# Type checking - explicit about required types
def process_sequence_safe(seq):
    from collections.abc import Sequence
    if not isinstance(seq, Sequence):
        raise TypeError("Input must be a sequence")
    return [x * 2 for x in seq]
```

#### Custom Type Checking with `__instancecheck__` and `__subclasscheck__`

```python
class NumericMeta(type):
    def __instancecheck__(self, instance):
        return hasattr(instance, '__int__') and hasattr(instance, '__float__')

class Numeric(metaclass=NumericMeta):
    """A class representing any numeric-like object"""
    pass

# Now we can check if something behaves like a number
print(isinstance(42, Numeric))        # True
print(isinstance(3.14, Numeric))      # True
print(isinstance("not a number", Numeric))  # False

# Even custom classes with numeric interfaces
class MyNumeric:
    def __int__(self):
        return 0
    def __float__(self):
        return 0.0

print(isinstance(MyNumeric(), Numeric))  # True
```

#### Best Practices

1. **Use for type safety**: Use `isinstance()` for runtime type checking when necessary.
2. **Consider duck typing**: In many cases, duck typing ("if it walks like a duck...") is more Pythonic than explicit type checking.
3. **Check against ABCs**: Use collections.abc for checking against abstract interfaces rather than concrete implementations.
4. **Avoid checking for exact types**: Check against base classes or interfaces when possible for more flexible code.
5. **Performance considerations**: These functions have some overhead, so avoid in performance-critical loops.

### 1.4 Understanding `all()`, `any()`, `sorted()`, `filter()`, `map()`, and `reduce()`

These built-in functions provide powerful tools for operating on iterables in Python, enabling concise, functional programming styles.

#### The `all()` Function

`all()` returns `True` if all elements in the iterable are truthy (or if the iterable is empty).

##### Internal Implementation

A simplified version of `all()` might look like:

```python
def my_all(iterable):
    for element in iterable:
        if not element:
            return False
    return True
```

##### Practical Examples

```python
# Check if all numbers are positive
numbers = [1, 2, 3, 4, 5]
print(all(num > 0 for num in numbers))  # True

# Check if all strings are uppercase
words = ['HELLO', 'WORLD', 'python']
print(all(word.isupper() for word in words))  # False

# Data validation example
def validate_user_data(user):
    required_fields = ['name', 'email', 'age']
    return all(field in user for field in required_fields)

user1 = {'name': 'Alice', 'email': 'alice@example.com', 'age': 30}
user2 = {'name': 'Bob', 'email': 'bob@example.com'}

print(validate_user_data(user1))  # True
print(validate_user_data(user2))  # False
```

#### The `any()` Function

`any()` returns `True` if at least one element in the iterable is truthy. It returns `False` if the iterable is empty.

##### Internal Implementation

A simplified version of `any()` might be:

```python
def my_any(iterable):
    for element in iterable:
        if element:
            return True
    return False
```

##### Practical Examples

```python
# Check if any number is negative
numbers = [1, 2, -3, 4, 5]
print(any(num < 0 for num in numbers))  # True

# Check if any string contains 'python'
technologies = ['java', 'python', 'javascript', 'rust']
print(any('python' in tech for tech in technologies))  # True

# Error checking example
def has_errors(results):
    return any('error' in result for result in results)

api_results = [
    {'status': 'success', 'data': [1, 2, 3]},
    {'status': 'error', 'message': 'Invalid input'}
]
print(has_errors(api_results))  # True
```

#### The `sorted()` Function

`sorted()` returns a new sorted list from the elements of any iterable.

##### Internal Implementation

Python's `sorted()` uses the Timsort algorithm, a hybrid sorting algorithm derived from merge sort and insertion sort. It's optimized for real-world data and has a worst-case complexity of O(n log n).

A very simplified version (not the actual Timsort implementation):

```python
def my_sorted(iterable, key=None, reverse=False):
    # Convert to list
    result = list(iterable)
    
    # Apply key function if provided
    if key:
        # Sort using key function
        result.sort(key=key, reverse=reverse)
    else:
        # Sort directly
        result.sort(reverse=reverse)
        
    return result
```

##### Practical Examples

```python
# Basic sorting
numbers = [3, 1, 4, 1, 5, 9, 2]
print(sorted(numbers))  # [1, 1, 2, 3, 4, 5, 9]

# Sorting with a key function
words = ['banana', 'apple', 'Cherry', 'date']
print(sorted(words))  # ['Cherry', 'apple', 'banana', 'date'] (sorts by ASCII value)
print(sorted(words, key=str.lower))  # ['apple', 'banana', 'Cherry', 'date'] (case-insensitive)

# Sorting by multiple criteria
students = [
    {'name': 'Alice', 'grade': 85, 'age': 19},
    {'name': 'Bob', 'grade': 92, 'age': 20},
    {'name': 'Charlie', 'grade': 85, 'age': 18}
]

# Sort by grade (descending), then by age (ascending)
sorted_students = sorted(
    students,
    key=lambda s: (-s['grade'], s['age'])
)
for student in sorted_students:
    print(f"{student['name']}: Grade {student['grade']}, Age {student['age']}")
```

**Example: Custom Sort with `attrgetter` and `itemgetter`**

```python
from operator import attrgetter, itemgetter
from collections import namedtuple

# Using itemgetter for dictionary sorting
contacts = [
    {'name': 'Alice', 'phone': '555-1234'},
    {'name': 'Bob', 'phone': '555-5678'},
    {'name': 'Charlie', 'phone': '555-9012'}
]
sorted_contacts = sorted(contacts, key=itemgetter('name'))

# Using attrgetter for object sorting
Person = namedtuple('Person', ['name', 'age', 'height'])
people = [
    Person('Alice', 30, 165),
    Person('Bob', 25, 180),
    Person('Charlie', 35, 175)
]
sorted_by_age = sorted(people, key=attrgetter('age'))
sorted_by_height_and_age = sorted(people, key=attrgetter('height', 'age'))
```

#### The `filter()` Function

`filter()` constructs an iterator from elements of an iterable for which a function returns `True`.

##### Internal Implementation

A simplified implementation might look like:

```python
def my_filter(function, iterable):
    if function is None:
        function = bool
    for item in iterable:
        if function(item):
            yield item
```

##### Practical Examples

```python
# Filter even numbers
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(even_numbers)  # [2, 4, 6, 8, 10]

# Filter non-empty strings
strings = ['hello', '', 'world', '', 'python']
non_empty = list(filter(None, strings))  # None is equivalent to bool
print(non_empty)  # ['hello', 'world', 'python']

# Filter objects by attribute
class Product:
    def __init__(self, name, price, in_stock):
        self.name = name
        self.price = price
        self.in_stock = in_stock

products = [
    Product('Laptop', 1200, True),
    Product('Phone', 800, False),
    Product('Tablet', 500, True)
]

available_products = list(filter(lambda p: p.in_stock, products))
print([p.name for p in available_products])  # ['Laptop', 'Tablet']
```

#### The `map()` Function

`map()` applies a specified function to each item of an iterable and returns an iterator of the results.

##### Internal Implementation

A simplified implementation of `map()` might be:

```python
def my_map(function, *iterables):
    # Get iterators from all iterables
    iterators = [iter(iterable) for iterable in iterables]
    
    while True:
        # Get next items from all iterators
        items = []
        for iterator in iterators:
            try:
                items.append(next(iterator))
            except StopIteration:
                return
        
        # Apply the function and yield the result
        yield function(*items)
```

##### Practical Examples

```python
# Square each number
numbers = [1, 2, 3, 4, 5]
squares = list(map(lambda x: x ** 2, numbers))
print(squares)  # [1, 4, 9, 16, 25]

# Convert strings to integers
string_numbers = ['1', '2', '3', '4', '5']
int_numbers = list(map(int, string_numbers))
print(int_numbers)  # [1, 2, 3, 4, 5]

# Apply a function to multiple iterables
def add(a, b):
    return a + b

list1 = [1, 2, 3]
list2 = [10, 20, 30]
result = list(map(add, list1, list2))
print(result)  # [11, 22, 33]

# Data transformation for machine learning
raw_features = [
    [1.2, 0.5, 2.1],
    [0.8, 1.1, 1.9],
    [3.2, 0.9, 0.2]
]

# Normalize each feature vector
def normalize(feature_vector):
    total = sum(feature_vector)
    return [val/total for val in feature_vector]

normalized_features = list(map(normalize, raw_features))
for features in normalized_features:
    print(features)
```

#### The `reduce()` Function

`reduce()` applies a function of two arguments cumulatively to the items of an iterable, reducing it to a single value. It's found in the `functools` module since Python 3.

##### Internal Implementation

A simplified implementation:

```python
def my_reduce(function, iterable, initializer=None):
    iterator = iter(iterable)
    
    # Handle initializer
    if initializer is None:
        try:
            accumulator = next(iterator)
        except StopIteration:
            raise TypeError("reduce() of empty sequence with no initial value")
    else:
        accumulator = initializer
    
    # Apply function to each element
    for element in iterator:
        accumulator = function(accumulator, element)
    
    return accumulator
```

##### Practical Examples

```python
from functools import reduce

# Sum of numbers
numbers = [1, 2, 3, 4, 5]
sum_result = reduce(lambda x, y: x + y, numbers)
print(sum_result)  # 15

# Finding the maximum value
max_value = reduce(lambda x, y: x if x > y else y, numbers)
print(max_value)  # 5

# Flattening a list of lists
nested_list = [[1, 2], [3, 4], [5, 6]]
flattened = reduce(lambda x, y: x + y, nested_list)
print(flattened)  # [1, 2, 3, 4, 5, 6]

# Composing functions
def compose(*functions):
    def inner(arg):
        return reduce(lambda arg, f: f(arg), functions, arg)
    return inner

# Example of function composition
def double(x): return x * 2
def increment(x): return x + 1
def square(x): return x ** 2

composed_fn = compose(double, increment, square)
print(composed_fn(3))  # Square of (increment of (double of 3)) = (3*2+1)^2 = 7^2 = 49
```

#### Best Practices for These Functions

1. **Use list comprehensions**: For simple transformations, list comprehensions may be more readable than `map()` or `filter()`.
2. **Avoid unnecessary lists**: Remember that `map()`, `filter()`, and other functions return iterators. Only convert to lists when necessary.
3. **Consider generator expressions**: When working with large datasets, generator expressions can be more memory efficient.
4. **Use key functions effectively**: Master the `key` parameter in `sorted()` for complex sorting.
5. **Use `reduce()` judiciously**: While powerful, `reduce()` can sometimes make code less readable. Consider alternatives when they're clearer.
6. **Use lambda functions wisely**: While convenient, complex operations may be better as named functions for readability.

## Module 2: Comprehensions & Generators

### 2.1 List, Set, and Dictionary Comprehensions

Comprehensions are concise syntactic constructs that allow you to create collections (lists, sets, dictionaries) from existing collections. They combine the functionality of `map()` and `filter()` into a more readable syntax.

#### How Comprehensions Work Internally

Comprehensions are syntactic sugar for more verbose loop-based approaches. When you write a comprehension, Python's interpreter translates it into roughly equivalent code using loops, conditionals, and the appropriate collection type.

For example, a list comprehension like `[x*2 for x in range(5) if x % 2 == 0]` is conceptually translated to:

```python
result = []
for x in range(5):
    if x % 2 == 0:
        result.append(x*2)
```

#### List Comprehensions

List comprehensions create new lists by applying an expression to each item in an iterable, optionally filtering elements with a condition.

**Basic Syntax:**
```
[expression for item in iterable if condition]
```

**Examples:**

```python
# Basic list comprehension
numbers = [1, 2, 3, 4, 5]
squared = [x**2 for x in numbers]
print(squared)  # [1, 4, 9, 16, 25]

# With filtering condition
even_squared = [x**2 for x in numbers if x % 2 == 0]
print(even_squared)  # [4, 16]

# With multiple conditions
complex_filter = [x for x in range(100) if x % 2 == 0 if x % 3 == 0]
print(complex_filter)  # [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]

# Nested list comprehension
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]
print(flattened)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Transforming nested structures
transposed = [[row[i] for row in matrix] for i in range(3)]
print(transposed)  # [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
```

#### Set Comprehensions

Set comprehensions are similar to list comprehensions but create sets (unordered collections of unique elements) instead of lists.

**Basic Syntax:**
```
{expression for item in iterable if condition}
```

**Examples:**

```python
# Basic set comprehension
numbers = [1, 2, 2, 3, 3, 4, 5, 5]
unique_squares = {x**2 for x in numbers}
print(unique_squares)  # {1, 4, 9, 16, 25}

# Filtering with conditions
vowels = {char.lower() for char in "Hello World" if char.lower() in 'aeiou'}
print(vowels)  # {'e', 'o'}

# Extracting unique values from a list of dictionaries
users = [
    {'id': 1, 'name': 'Alice', 'age': 30},
    {'id': 2, 'name': 'Bob', 'age': 25},
    {'id': 3, 'name': 'Charlie', 'age': 35},
    {'id': 4, 'name': 'Diana', 'age': 30}
]
unique_ages = {user['age'] for user in users}
print(unique_ages)  # {25, 30, 35}
```

#### Dictionary Comprehensions

Dictionary comprehensions create dictionaries by defining both keys and values based on iterable elements.

**Basic Syntax:**
```
{key_expression: value_expression for item in iterable if condition}
```

**Examples:**

```python
# Basic dictionary comprehension
numbers = [1, 2, 3, 4, 5]
squared_dict = {x: x**2 for x in numbers}
print(squared_dict)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# Filtering with conditions
even_squared_dict = {x: x**2 for x in numbers if x % 2 == 0}
print(even_squared_dict)  # {2: 4, 4: 16}

# Transforming an existing dictionary
prices = {'apple': 0.50, 'banana': 0.25, 'orange': 0.75, 'pear': 0.60}
doubled_prices = {fruit: price*2 for fruit, price in prices.items()}
print(doubled_prices)  # {'apple': 1.0, 'banana': 0.5, 'orange': 1.5, 'pear': 1.2}

# Creating a dictionary from two lists
fruits = ['apple', 'banana', 'orange']
prices = [0.5, 0.25, 0.75]
fruit_prices = {fruit: price for fruit, price in zip(fruits, prices)}
print(fruit_prices)  # {'apple': 0.5, 'banana': 0.25, 'orange': 0.75}

# Inverting a dictionary (swapping keys and values)
# Note: This only works if values are unique and hashable
original = {'a': 1, 'b': 2, 'c': 3}
inverted = {value: key for key, value in original.items()}
print(inverted)  # {1: 'a', 2: 'b', 3: 'c'}
```

#### Performance Considerations

1. **Memory Usage**: Comprehensions create the entire collection in memory at once, which can be problematic for very large datasets.
2. **Speed**: Comprehensions are generally faster than equivalent `for` loops because they're optimized at the implementation level.
3. **Readability Threshold**: While comprehensions are concise, complex ones with multiple conditions or nested iterations can become hard to read.

#### Optimizations and Best Practices

1. **Balance Readability and Conciseness**: Avoid overly complex comprehensions. If it requires more than one condition or nested iteration, consider using a loop instead.

2. **Choose the Right Comprehension Type**:
   - Use list comprehensions when order matters and duplicates are allowed
   - Use set comprehensions when you need unique values and order doesn't matter
   - Use dictionary comprehensions for key-value mappings

3. **Avoid Side Effects**: Comprehensions should be used for creating new collections, not for side effects. For operations with side effects, use traditional loops.

4. **Measure Performance**: For performance-critical code, measure the performance of comprehensions vs. loops in your specific context.

5. **Use Generator Expressions for Large Datasets**: When working with large datasets, consider generator expressions instead of comprehensions to save memory.

**Example: Too Complex vs. More Readable**

```python
# Too complex - hard to read
result = [x*y for x in range(10) if x % 2 == 0 for y in range(5) if y % 2 == 1]

# More readable alternative
result = []
for x in range(10):
    if x % 2 == 0:
        for y in range(5):
            if y % 2 == 1:
                result.append(x*y)
```

**Example: Using Comprehensions in Data Science**

```python
import pandas as pd
import numpy as np

# Create a dataset from a list of dictionaries
data = [
    {'name': 'Alice', 'age': 30, 'city': 'New York', 'score': 85},
    {'name': 'Bob', 'age': 25, 'city': 'Boston', 'score': 92},
    {'name': 'Charlie', 'age': 35, 'city': 'Chicago', 'score': 78},
    {'name': 'Diana', 'age': 40, 'city': 'Denver', 'score': 95}
]

# Extract and transform data using comprehensions
names = [person['name'] for person in data]
scores = {person['name']: person['score'] for person in data}
city_counts = {city: sum(1 for person in data if person['city'].startswith(city[0])) 
               for city in set(person['city'] for person in data)}

# Data normalization
max_score = max(person['score'] for person in data)
normalized_scores = {person['name']: person['score']/max_score for person in data}

# Filtering outliers
outliers = [person for person in data if abs(person['score'] - np.mean([p['score'] for p in data])) > 10]
```

### 2.2 Generators and the `yield` Statement

Generators are iterators created using functions with the `yield` statement. They allow you to create sequences of values on-the-fly without storing the entire sequence in memory, making them memory-efficient for large datasets.

#### How Generators Work Internally

When a generator function is called, it returns a generator object (an iterator) rather than executing the function body. Each time the `next()` function is called on the generator, execution proceeds to the next `yield` statement, and the yielded value is returned.

The function's state (local variables and execution point) is saved between calls, allowing it to resume where it left off.

The simplified internal mechanism involves:

1. Creating a generator object when the function is called
2. Saving and restoring the function's stack frame for each `next()` call
3. Raising a `StopIteration` exception when the function exits

#### Basic Generator Functions

```python
def count_up_to(max):
    count = 1
    while count <= max:
        yield count
        count += 1

# Using the generator
counter = count_up_to(5)
print(next(counter))  # 1
print(next(counter))  # 2
print(next(counter))  # 3

# Iterating over the generator
for num in count_up_to(3):
    print(num)  # Prints 1, 2, 3
```

#### Generator Expressions

Generator expressions are similar to list comprehensions but create generators instead of lists. They use parentheses instead of square brackets.

```python
# List comprehension (creates a list in memory)
squares_list = [x**2 for x in range(1000000)]  # Uses a lot of memory

# Generator expression (creates a generator)
squares_gen = (x**2 for x in range(1000000))  # Uses minimal memory

# The generator is evaluated lazily
print(next(squares_gen))  # 0
print(next(squares_gen))  # 1
```

#### Memory Efficiency Comparison

```python
import sys

# List comprehension
squares_list = [x**2 for x in range(10000)]
print(f"List size: {sys.getsizeof(squares_list)} bytes")

# Generator expression
squares_gen = (x**2 for x in range(10000))
print(f"Generator size: {sys.getsizeof(squares_gen)} bytes")

# Output might be something like:
# List size: 87624 bytes
# Generator size: 112 bytes
```

#### Generator Applications

**Example 1: Reading Large Files**

```python
def read_large_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()

# Process a large file line by line without loading it all into memory
for line in read_large_file('large_log_file.txt'):
    if 'ERROR' in line:
        print(f"Found error: {line}")
```

**Example 2: Infinite Sequences**

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Get the first 10 Fibonacci numbers
fib_gen = fibonacci()
for _ in range(10):
    print(next(fib_gen))
```

**Example 3: Data Pipeline Processing**

```python
def read_data(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()

def parse_csv(lines):
    for line in lines:
        yield line.split(',')

def convert_types(rows):
    for row in rows:
        # Convert string values to appropriate types
        yield [int(row[0]), float(row[1]), str(row[2])]

def filter_data(rows, threshold):
    for row in rows:
        if row[1] > threshold:
            yield row

# Create a data processing pipeline
raw_data = read_data('data.csv')
parsed_data = parse_csv(raw_data)
typed_data = convert_types(parsed_data)
filtered_data = filter_data(typed_data, 0.5)

# Process the data
for data_point in filtered_data:
    print(data_point)
```

#### Advanced Generator Features

**Example 1: Sending Values to Generators (`send()`)**

```python
def echo_generator():
    value = yield "Ready for input"
    while True:
        value = yield f"You said: {value}"

# Create the generator
echo = echo_generator()

# Start the generator
initial = next(echo)
print(initial)  # "Ready for input"

# Send values to the generator
print(echo.send("Hello"))  # "You said: Hello"
print(echo.send("World"))  # "You said: World"
```

**Example 2: Closing Generators (`close()`)**

```python
def resource_generator():
    try:
        print("Resource opened")
        for i in range(3):
            yield i
    finally:
        print("Resource closed")

gen = resource_generator()
print(next(gen))  # 0
print(next(gen))  # 1
gen.close()       # "Resource closed"
```

**Example 3: Exception Handling in Generators (`throw()`)**

```python
def exception_handling_generator():
    try:
        yield "Normal operation"
        yield "Still working"
    except ValueError:
        yield "Caught ValueError"
    yield "Continuing after exception"

gen = exception_handling_generator()
print(next(gen))             # "Normal operation"
print(gen.throw(ValueError))  # "Caught ValueError"
print(next(gen))             # "Continuing after exception"
```

#### Yield From: Delegating to Sub-generators

Python 3.3 introduced the `yield from` syntax, which allows a generator to delegate part of its operations to another generator, creating composable generator pipelines.

```python
def sub_generator():
    yield 1
    yield 2
    yield 3

def main_generator():
    yield 'A'
    yield from sub_generator()  # Delegates to sub_generator
    yield 'B'

# Output: A, 1, 2, 3, B
for item in main_generator():
    print(item)
```

**Example: Recursive Directory Traversal**

```python
import os

def all_files(directory):
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isdir(full_path):
            yield from all_files(full_path)  # Recurse into subdirectories
        else:
            yield full_path

# Find all Python files in the current directory and its subdirectories
python_files = (f for f in all_files('.') if f.endswith('.py'))
for file in python_files:
    print(file)
```

#### Best Practices for Generators

1. **Use generators for large datasets** to minimize memory usage
2. **Use generator expressions** instead of list comprehensions when you only need to iterate once
3. **Create data processing pipelines** by chaining generators together
4. **Close generators** that use resources when you're done with them
5. **Use `yield from`** to delegate to sub-generators and create more composable code
6. **Remember generator state**: Generators maintain state between calls, which can be both powerful and tricky
7. **Consider performance implications**: Generators trade some CPU efficiency for memory efficiency

### 2.3 Using `itertools` for Advanced Iteration Techniques

The `itertools` module provides a collection of fast, memory-efficient tools for creating and working with iterators. These tools are inspired by constructs from functional programming languages.

#### Core `itertools` Functions

##### Infinite Iterators

```python
import itertools

# count: count(start=0, step=1) -> start, start+step, start+2*step, ...
counter = itertools.count(10, 2)
print([next(counter) for _ in range(5)])  # [10, 12, 14, 16, 18]

# cycle: cycle(iterable) -> elements from iterable repeatedly
cycler = itertools.cycle(['A', 'B', 'C'])
print([next(cycler) for _ in range(7)])  # ['A', 'B', 'C', 'A', 'B', 'C', 'A']

# repeat: repeat(element, [times]) -> element repeated times or indefinitely
repeater = itertools.repeat('X', 5)
print(list(repeater))  # ['X', 'X', 'X', 'X', 'X']
```

##### Iterators Terminating on the Shortest Input Sequence

```python
# chain: chain(*iterables) -> elements from all iterables in sequence
combined = itertools.chain([1, 2, 3], ['a', 'b'], [7, 8, 9])
print(list(combined))  # [1, 2, 3, 'a', 'b', 7, 8, 9]

# compress: compress(data, selectors) -> elements from data where selector is True
data = ['A', 'B', 'C', 'D', 'E']
selectors = [1, 0, 1, 0, 1]
print(list(itertools.compress(data, selectors)))  # ['A', 'C', 'E']

# dropwhile: dropwhile(predicate, iterable) -> elements once predicate is False
result = itertools.dropwhile(lambda x: x < 5, [1, 3, 6, 2, 1, 9, 4])
print(list(result))  # [6, 2, 1, 9, 4] - drops initial elements until predicate is False

# takewhile: takewhile(predicate, iterable) -> elements until predicate is False
result = itertools.takewhile(lambda x: x < 5, [1, 3, 6, 2, 1, 9, 4])
print(list(result))  # [1, 3] - takes elements until predicate is False

# filterfalse: filterfalse(predicate, iterable) -> elements where predicate is False
result = itertools.filterfalse(lambda x: x % 2 == 0, range(10))
print(list(result))  # [1, 3, 5, 7, 9] - filter out even numbers

# islice: islice(iterable, [start], stop, [step]) -> elements from iterable[start:stop:step]
result = itertools.islice("ABCDEFGH", 2, 6, 2)
print(list(result))  # ['C', 'E'] - like slice but for iterators
```

##### Combinatoric Iterators

```python
# product: product(*iterables, repeat=1) -> Cartesian product
result = itertools.product('AB', [1, 2])
print(list(result))  # [('A', 1), ('A', 2), ('B', 1), ('B', 2)]

# permutations: permutations(iterable, r=None) -> r-length permutations
result = itertools.permutations('ABC', 2)
print(list(result))  # [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]

# combinations: combinations(iterable, r) -> r-length combinations
result = itertools.combinations('ABC', 2)
print(list(result))  # [('A', 'B'), ('A', 'C'), ('B', 'C')]

# combinations_with_replacement: combinations_with_replacement(iterable, r) -> r-length combinations allowing repeated elements
result = itertools.combinations_with_replacement('ABC', 2)
print(list(result))  # [('A', 'A'), ('A', 'B'), ('A', 'C'), ('B', 'B'), ('B', 'C'), ('C', 'C')]
```

#### Real-World Applications of `itertools`

**Example 1: Sliding Window Analysis**

```python
def sliding_window(iterable, n):
    """Collect data into fixed-length overlapping windows."""
    # sliding_window('ABCDEFG', 4) -> ABCD BCDE CDEF DEFG
    iterables = itertools.tee(iterable, n)
    for i, it in enumerate(iterables):
        for _ in range(i):
            next(it, None)
    return zip(*iterables)

# Example: Computing a moving average
data = [1, 5, 8, 4, 3, 9, 11, 15, 17, 3]
windows = sliding_window(data, 3)

moving_averages = [sum(window)/len(window) for window in windows]
print(list(moving_averages))
# [4.666..., 5.666..., 5.0, 5.333..., 7.666..., 11.666..., 14.333..., 11.666...]
```

**Example 2: Data Grouping and Aggregation**

```python
from itertools import groupby
import statistics

# Sample data: Student records with (name, subject, score)
data = [
    ('Alice', 'Math', 95),
    ('Bob', 'Math', 85),
    ('Alice', 'Science', 90),
    ('Charlie', 'Math', 82),
    ('Bob', 'Science', 88),
    ('Charlie', 'Science', 91)
]

# Sort by student name (required for groupby to work correctly)
data.sort(key=lambda x: x[0])

# Group by student name and calculate average score
for student, group in groupby(data, key=lambda x: x[0]):
    scores = [score for _, _, score in group]
    avg_score = statistics.mean(scores)
    print(f"{student}: Average score = {avg_score:.2f}")
```

**Example 3: Combinations for Machine Learning Feature Generation**

```python
import itertools
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Original features
X = np.array([[1, 2], [3, 4], [5, 6]])

# Using scikit-learn's PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_sklearn = poly.fit_transform(X)
print("Scikit-learn polynomial features:")
print(X_poly_sklearn)

# Implementing a custom polynomial feature generator using itertools
def polynomial_features(X, degree=2, include_bias=True):
    n_samples, n_features = X.shape
    feature_names = [f"x{i}" for i in range(n_features)]
    
    # List to store all combinations of features
    combos = []
    
    # Add bias term if requested
    if include_bias:
        combos.append(np.ones(n_samples).reshape(-1, 1))
    
    # Add original features
    for i in range(n_features):
        combos.append(X[:, i].reshape(-1, 1))
    
    # Add interaction terms
    for d in range(2, degree + 1):
        for combo in itertools.combinations_with_replacement(range(n_features), d):
            new_feature = np.ones(n_samples)
            for i in combo:
                new_feature *= X[:, i]
            combos.append(new_feature.reshape(-1, 1))
    
    return np.hstack(combos)

# Using our custom function
X_poly_custom = polynomial_features(X, degree=2, include_bias=False)
print("\nCustom polynomial features using itertools:")
print(X_poly_custom)
```

**Example 4: Efficient Data Processing Pipeline**

```python
import itertools
import csv
from collections import namedtuple

# Define a namedtuple for our data
DataPoint = namedtuple('DataPoint', ['id', 'value', 'category'])

def load_data(filename):
    """Load data from a CSV file."""
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            yield DataPoint(int(row[0]), float(row[1]), row[2])

def preprocess(data_points):
    """Preprocess data points by filtering and transforming."""
    # Filter out negative values
    non_negative = itertools.filterfalse(lambda x: x.value < 0, data_points)
    
    # Transform the data (e.g., normalize value)
    max_value = 100.0  # Assume maximum value is 100
    return (DataPoint(dp.id, dp.value / max_value, dp.category) for dp in non_negative)

def group_by_category(data_points):
    """Group data points by category."""
    # Sort by category (required for groupby)
    sorted_data = sorted(data_points, key=lambda x: x.category)
    
    # Group by category
    for category, group in itertools.groupby(sorted_data, key=lambda x: x.category):
        yield (category, list(group))

# Full data processing pipeline
def process_data(filename):
    raw_data = load_data(filename)
    preprocessed_data = preprocess(raw_data)
    grouped_data = group_by_category(preprocessed_data)
    
    # Process each group
    results = {}
    for category, group in grouped_data:
        # Calculate statistics for each group
        values = [dp.value for dp in group]
        if values:
            results[category] = {
                'count': len(values),
                'mean': sum(values) / len(values),
                'max': max(values)
            }
    
    return results

# Example usage (assuming 'data.csv' exists)
# results = process_data('data.csv')
# print(results)
```

#### Performance Considerations and Best Practices

1. **Memory Efficiency**: Most `itertools` functions return iterators, which are evaluated lazily and have minimal memory overhead.

2. **Speed**: The `itertools` functions are implemented in C, making them faster than equivalent Python loops.

3. **Chaining Functions**: Combine multiple `itertools` functions to create powerful data processing pipelines.

4. **Sort Before `groupby`**: Always sort data on the grouping key before using `groupby()`, as it only identifies consecutive elements with the same key.

5. **Infinite Iterators**: Be careful with infinite iterators like `count()` and `cycle()`. Always limit them with functions like `islice()` or control them in your code.

6. **Reuse with `tee()`**: Use `itertools.tee()` to create multiple independent copies of an iterator when you need to process the same data multiple times.

```python
import itertools

# Create an iterator
data = iter([1, 2, 3, 4, 5])

# Mistake: Trying to reuse the same iterator twice
# first_half = list(itertools.islice(data, 3))
# second_half = list(data)  # This will only have [4, 5] because data has been partially consumed

# Correct approach: Use tee to create independent copies
data = iter([1, 2, 3, 4, 5])
iter1, iter2 = itertools.tee(data)
first_three = list(itertools.islice(iter1, 3))
all_items = list(iter2)

print(first_three)  # [1, 2, 3]
print(all_items)    # [1, 2, 3, 4, 5]
```

7. **Avoid Materializing Large Iterators**: Only convert iterators to lists or other collections when necessary, especially for large datasets.

8. **Combine with Generator Expressions**: Use generator expressions alongside `itertools` functions for powerful, memory-efficient data processing.

```python
import itertools

# Large range of numbers
numbers = range(10**7)  # Would be expensive to convert to a list

# Find all pairs of adjacent numbers that sum to a prime
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# Memory-efficient pipeline using generator expressions and itertools
pairs = itertools.islice(
    ((a, b) for a, b in itertools.pairwise(numbers) if is_prime(a + b)),
    10  # Only take first 10 pairs
)

for a, b in pairs:
    print(f"{a} + {b} = {a + b} (prime)")
```

## Module 3: Decorators & Meta-programming

### 3.1 Writing Function Decorators

Decorators are a powerful feature in Python that allow you to modify the behavior of functions or classes without changing their code. They use the `@` syntax and are a form of metaprogramming.

#### How Decorators Work Internally

A decorator is a function that takes another function as an argument and returns a new function that usually extends or modifies the behavior of the original function. When you use the `@decorator` syntax, Python executes the decorator function at definition time and replaces the decorated function with the result.

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        # Code to execute before the function
        print("Before the function call")
        
        # Call the original function
        result = func(*args, **kwargs)
        
        # Code to execute after the function
        print("After the function call")
        
        # Return the result
        return result
    
    # Return the wrapper function
    return wrapper

# Using the decorator with @ syntax
@my_decorator
def say_hello():
    print("Hello!")

# This is equivalent to:
# say_hello = my_decorator(say_hello)

say_hello()
# Output:
# Before the function call
# Hello!
# After the function call
```

#### Basic Decorator Patterns

**Example 1: Timing Decorator**

```python
import time
import functools

def timer(func):
    @functools.wraps(func)  # Preserves the original function's metadata
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.6f} seconds to run")
        return result
    return wrapper

@timer
def slow_function():
    """This is a slow function that sleeps for 1 second."""
    time.sleep(1)
    return "Done!"

print(slow_function())
print(f"Function name: {slow_function.__name__}")  # 'slow_function', preserved by @wraps
print(f"Docstring: {slow_function.__doc__}")       # Docstring preserved by @wraps
```

**Example 2: Logging Decorator**

```python
import functools
import logging

logging.basicConfig(level=logging.INFO)

def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        
        logging.info(f"Calling {func.__name__}({signature})")
        try:
            result = func(*args, **kwargs)
            logging.info(f"{func.__name__} returned {result!r}")
            return result
        except Exception as e:
            logging.exception(f"{func.__name__} raised {type(e).__name__}: {e}")
            raise
            
    return wrapper

@log_function_call
def divide(a, b):
    return a / b

# Test successful call
print(divide(10, 2))

# Test exception
try:
    divide(5, 0)
except ZeroDivisionError:
    pass  # Exception already logged
```

**Example 3: Retry Decorator**

```python
import functools
import time
import random

def retry(max_tries=3, delay_seconds=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tries = 0
            while tries < max_tries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    tries += 1
                    if tries == max_tries:
                        raise e
                    print(f"Function {func.__name__} failed: {e}. Retrying in {delay_seconds} seconds...")
                    time.sleep(delay_seconds)
            return None  # This should never be reached
        return wrapper
    return decorator

@retry(max_tries=4, delay_seconds=0.5)
def unstable_network_call():
    # Simulate a network call that sometimes fails
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("Network error")
    return "Data received successfully"

# Test the retry mechanism
print(unstable_network_call())
```

#### Understanding `@staticmethod`, `@classmethod`, and `@property`

These built-in decorators modify how methods behave in classes.

**Example: Built-in Method Decorators**

```python
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        """Get temperature in Celsius."""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        """Set temperature in Celsius."""
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        """Get temperature in Fahrenheit."""
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        """Set temperature in Fahrenheit."""
        self.celsius = (value - 32) * 5/9
    
    @staticmethod
    def is_valid_temperature(value):
        """Check if temperature is physically possible."""
        return value >= -273.15
    
    @classmethod
    def from_fahrenheit(cls, value):
        """Create a Temperature instance from a Fahrenheit value."""
        celsius = (value - 32) * 5/9
        return cls(celsius)

# Using the class
temp = Temperature(25)
print(f"Celsius: {temp.celsius}, Fahrenheit: {temp.fahrenheit}")

# Use property setter
temp.fahrenheit = 68
print(f"After setting Fahrenheit: Celsius = {temp.celsius}")

# Use staticmethod
print(f"Is -300°C valid? {Temperature.is_valid_temperature(-300)}")

# Use classmethod factory
temp_f = Temperature.from_fahrenheit(98.6)
print(f"Body temperature: {temp_f.celsius:.1f}°C")
```

#### Decorators with Arguments

Decorators that take arguments require an additional layer of nesting.

```python
def repeat(times=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(times):
                results.append(func(*args, **kwargs))
            return results
        return wrapper
    return decorator

@repeat(times=3)
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))  # Calls greet 3 times with "Alice"
```

#### Stacking Decorators

Decorators can be stacked, with each decorator applying its transformation to the function in order from bottom to top.

```python
@timer
@log_function_call
@retry(max_tries=2)
def complex_operation(x, y):
    """Perform a complex operation that might fail."""
    # Simulating a potentially unstable operation
    if random.random() < 0.3:
        raise ValueError("Random failure")
    return x ** y
```

In this example, `retry` is applied first, then `log_function_call`, and finally `timer`. So the complete execution flow is:

1. `timer` starts timing
2. `log_function_call` logs the function call
3. `retry` attempts to run the function
4. If `complex_operation` fails, `retry` will try again
5. After success or max retries, `log_function_call` logs the result
6. `timer` records the total time and prints it

#### Creating a Decorator Factory with Customization

```python
def validate(validation_func=None, error_message=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the first argument (self for methods, first param for functions)
            value = args[0] if args else None
            
            # Use custom validation or default to truthy check
            validator = validation_func if validation_func else bool
            
            if not validator(value):
                # Use custom error or default message
                msg = error_message if error_message else f"Invalid input: {value}"
                raise ValueError(msg)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Using the validator with different configurations
@validate(validation_func=lambda x: x > 0, error_message="Value must be positive")
def calculate_square_root(x):
    return x ** 0.5

try:
    print(calculate_square_root(-5))
except ValueError as e:
    print(e)  # Value must be positive
```

#### Best Practices for Decorators

1. **Use `functools.wraps`**: Always use `@functools.wraps(func)` in your wrapper function to preserve the decorated function's metadata (name, docstring, etc.).

2. **Keep Decorators Simple**: Decorators should have a single, focused responsibility. Complex logic should be moved to helper functions.

3. **Handle Arguments Correctly**: Make sure your decorators properly handle any arguments and keyword arguments that might be passed to the decorated function.

4. **Error Handling**: Properly handle and propagate exceptions from the decorated function.

5. **Consider Performance**: Decorators add a function call overhead. For performance-critical functions, measure the impact.

6. **Documentation**: Document what your decorators do, especially if they modify behavior in non-obvious ways.

7. **Testing**: Write tests specifically for decorated functions to ensure the decorators work as expected.

### 3.2 How Closures and `nonlocal` Work Under the Hood

Closures are a fundamental concept in Python, especially when working with decorators and higher-order functions. They allow functions to "remember" the environment in which they were created.

#### What is a Closure?

A closure is a function object that has access to variables in its lexical scope, even when the function is called outside that scope. In other words, a closure "closes over" the free variables from its enclosing scope.

#### The Mechanics of Closures in Python

In Python, when a nested function refers to a value from its enclosing function, it creates a closure. The interpreter keeps track of the values in the parent function that the nested function needs.

```python
def outer_function(x):
    # x is a local variable in outer_function
    def inner_function(y):
        # inner_function has access to x from outer_function
        return x + y
    
    # Return the inner function, which forms a closure with x
    return inner_function

# Create a closure with x=10
add_10 = outer_function(10)

# Use the closure
print(add_10(5))  # 15
```

In this example, when `outer_function(10)` is called, it returns `inner_function` that "remembers" that `x` is 10. The value of `x` is stored with the function object, even though `outer_function` has completed execution.

#### How Python Implements Closures

Python implements closures using a special attribute called `__closure__` on function objects. This attribute contains cells, each of which references a variable from an enclosing scope.

```python
def counter_factory():
    count = 0
    
    def increment():
        nonlocal count
        count += 1
        return count
    
    return increment

counter = counter_factory()
print(counter())  # 1
print(counter())  # 2

# Inspect the closure
print(counter.__closure__)  # (<cell at 0x...: int object at 0x...>,)
print(counter.__closure__[0].cell_contents)  # 2
```

#### Understanding `nonlocal`

The `nonlocal` keyword was introduced in Python 3 to declare that a variable refers to a name in the nearest enclosing scope that is not global. It allows nested functions to modify variables from outer scopes.

```python
def outer():
    x = "outer"
    
    def inner():
        # Without nonlocal, this would create a new local variable x
        nonlocal x
        x = "inner"
        print("inner:", x)
    
    inner()
    print("outer:", x)

outer()
# Output:
# inner: inner
# outer: inner
```

##### How `nonlocal` Works Under the Hood

When a function is defined, Python analyzes the code to determine which variables are local and which are references to enclosing scopes. By default, when you assign to a variable inside a function, Python assumes it's a local variable.

The `nonlocal` statement changes this behavior by telling Python that a variable should be looked up in the nearest enclosing scope that isn't the global scope.

```python
def counter_without_nonlocal():
    count = 0
    
    def increment():
        # This would cause an UnboundLocalError
        # count += 1  # Error: local variable 'count' referenced before assignment
        return count + 1
    
    return increment

def counter_with_nonlocal():
    count = 0
    
    def increment():
        nonlocal count  # Tell Python that count is from the enclosing scope
        count += 1
        return count
    
    return increment
```

#### Advanced Uses of Closures and `nonlocal`

**Example 1: Memoization with Closures**

```python
def memoize(func):
    cache = {}
    
    def wrapper(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Now fibonacci results are cached
print(fibonacci(30))  # Fast calculation using memoization
```

**Example 2: Creating Stateful Functions**

```python
def create_counter(start=0, step=1):
    count = start
    
    def counter():
        nonlocal count
        current = count
        count += step
        return current
    
    # Add a method to reset the counter
    def reset(new_start=None):
        nonlocal count
        if new_start is not None:
            count = new_start
        else:
            count = start
    
    counter.reset = reset
    return counter

# Create a counter that starts at 100 and counts by 10
counter = create_counter(100, 10)
print(counter())  # 100
print(counter())  # 110
print(counter())  # 120

# Reset the counter
counter.reset()
print(counter())  # 100

# Reset to a specific value
counter.reset(50)
print(counter())  # 50
```

**Example 3: Function Factories with Different Behaviors**

```python
def power_function(exponent):
    def power(base):
        return base ** exponent
    
    return power

# Create specific power functions
square = power_function(2)
cube = power_function(3)
sqrt = power_function(0.5)

print(square(4))  # 16
print(cube(4))    # 64
print(sqrt(4))    # 2.0
```

**Example 4: Data Encapsulation with Closures**

```python
def create_account(initial_balance=0):
    balance = initial_balance
    transaction_count = 0
    
    def deposit(amount):
        nonlocal balance, transaction_count
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        balance += amount
        transaction_count += 1
        return balance
    
    def withdraw(amount):
        nonlocal balance, transaction_count
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > balance:
            raise ValueError("Insufficient funds")
        balance -= amount
        transaction_count += 1
        return balance
    
    def get_balance():
        return balance
    
    def get_transaction_count():
        return transaction_count
    
    # Return a dictionary of available operations
    return {
        'deposit': deposit,
        'withdraw': withdraw,
        'get_balance': get_balance,
        'get_transaction_count': get_transaction_count
    }

# Create a bank account
account = create_account(1000)

# Use the account
print(account['get_balance']())                   # 1000
account['deposit'](500)
print(account['get_balance']())                   # 1500
account['withdraw'](300)
print(account['get_balance']())                   # 1200
print(account['get_transaction_count']())         # 2
```

#### Common Pitfalls and Best Practices

**Pitfall 1: Late Binding in Loops**

```python
# Problematic code
def create_multipliers():
    multipliers = []
    for i in range(5):
        def multiplier(x):
            return i * x  # i is a free variable
        multipliers.append(multiplier)
    return multipliers

# All multipliers will use the final value of i (4)
functions = create_multipliers()
print([f(2) for f in functions])  # [8, 8, 8, 8, 8]

# Fixed version using default arguments to capture the current value of i
def create_multipliers_fixed():
    multipliers = []
    for i in range(5):
        def multiplier(x, i=i):  # i is now a default parameter
            return i * x
        multipliers.append(multiplier)
    return multipliers

functions = create_multipliers_fixed()
print([f(2) for f in functions])  # [0, 2, 4, 6, 8]
```

**Best Practices:**

1. **Use `nonlocal` sparingly**: While powerful, excessive use of `nonlocal` can make code harder to understand and maintain.

2. **Consider alternative patterns**: Sometimes using classes or generators can be clearer than complex closures.

3. **Be aware of the late binding issue**: When creating functions in a loop, use default arguments to capture the current value of loop variables.

4. **Document closures well**: Since closures capture their environment in non-obvious ways, good documentation is essential.

5. **Keep closures focused**: Try to minimize the number of variables a closure needs to access from its enclosing scope.

6. **Watch out for reference cycles**: Closures can cause memory leaks if they create circular references.

### 3.3 Implementing Class Decorators and Modifying Classes Dynamically

Class decorators are similar to function decorators but operate on classes. They provide a powerful way to modify or enhance classes without changing their source code. Additionally, Python allows for dynamic class modification, enabling runtime changes to class attributes and behavior.

#### Class Decorators Basics

A class decorator is a function that takes a class as its argument and returns the same class or a new class as its result. It's applied using the `@decorator` syntax above the class definition.

```python
def add_greeting(cls):
    """A class decorator that adds a greeting method."""
    
    def greeting(self):
        return f"Hello, I'm {self.name} from {cls.__name__} class"
    
    cls.greeting = greeting
    return cls

@add_greeting
class Person:
    def __init__(self, name):
        self.name = name

# Using the decorated class
p = Person("Alice")
print(p.greeting())  # "Hello, I'm Alice from Person class"
```

#### Real-World Examples of Class Decorators

**Example 1: Singleton Pattern**

```python
def singleton(cls):
    """Implement the Singleton pattern with a decorator."""
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    # Replace the class with the get_instance function
    return get_instance

@singleton
class DatabaseConnection:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        print(f"Connecting to database at {host}:{port}")
    
    def query(self, sql):
        print(f"Executing SQL: {sql}")
        # Actual database interaction would go here
        return ["result1", "result2"]

# First call - creates the instance
db1 = DatabaseConnection("localhost", 5432)

# Second call - returns the same instance
db2 = DatabaseConnection("localhost", 5432)

print(db1 is db2)  # True - both variables reference the same instance
```

**Example 2: Auto-registering Classes**

```python
# A registry for different types of handlers
HANDLERS = {}

def register_handler(cls):
    """Register a handler class by its name."""
    HANDLERS[cls.__name__] = cls
    return cls

@register_handler
class EmailHandler:
    def send(self, message):
        print(f"Sending email: {message}")

@register_handler
class SMSHandler:
    def send(self, message):
        print(f"Sending SMS: {message}")

# Now we can easily look up handlers by name
def send_notification(handler_name, message):
    if handler_name not in HANDLERS:
        raise ValueError(f"Unknown handler: {handler_name}")
    
    handler = HANDLERS[handler_name]()
    handler.send(message)

# Using the registered handlers
send_notification("EmailHandler", "Hello via email")
send_notification("SMSHandler", "Hello via SMS")

# See all registered handlers
print(HANDLERS)
```

**Example 3: Adding Validation to Classes**

```python
def validate_attributes(cls):
    """Add validation to class attribute setters."""
    original_setattr = cls.__setattr__
    
    def __setattr__(self, name, value):
        # Execute attribute-specific validation method if it exists
        validator_name = f"validate_{name}"
        if hasattr(self, validator_name):
            validator = getattr(self, validator_name)
            validator(value)
        
        # Call the original __setattr__
        original_setattr(self, name, value)
    
    cls.__setattr__ = __setattr__
    return cls

@validate_attributes
class User:
    def __init__(self, username, email, age):
        self.username = username
        self.email = email
        self.age = age
    
    def validate_username(self, username):
        if not isinstance(username, str) or len(username) < 3:
            raise ValueError("Username must be a string with at least 3 characters")
    
    def validate_email(self, email):
        if not isinstance(email, str) or '@' not in email:
            raise ValueError("Invalid email format")
    
    def validate_age(self, age):
        if not isinstance(age, int) or age < 0:
            raise ValueError("Age must be a positive integer")

# Create a valid user
user1 = User("alice", "alice@example.com", 30)

# Try to create an invalid user
try:
    user2 = User("bob", "invalid_email", 25)
except ValueError as e:
    print(f"Validation error: {e}")
```

#### Dynamically Modifying Classes

Python allows you to modify classes at runtime by adding, changing, or removing attributes and methods.

**Example 1: Adding Methods and Attributes Dynamically**

```python
class DynamicClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Add an instance method dynamically
def add_method(instance):
    return instance.x + instance.y

DynamicClass.add = add_method

# Add a class method dynamically
@classmethod
def create_origin(cls):
    return cls(0, 0)

DynamicClass.create_origin = create_origin

# Add a property dynamically
def get_sum(self):
    return self.x + self.y

DynamicClass.sum = property(get_sum)

# Use the dynamically modified class
d = DynamicClass(5, 10)
print(d.add())  # 15
print(d.sum)    # 15

origin = DynamicClass.create_origin()
print(origin.x, origin.y)  # 0, 0
```

**Example 2: Method Injection with Monkeypatching**

```python
# Original class that we want to extend
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

# Add methods that weren't in the original class
def perimeter(self):
    return 2 * (self.width + self.height)

def is_square(self):
    return self.width == self.height

# Monkey-patch the class
Rectangle.perimeter = perimeter
Rectangle.is_square = is_square

# Use the enhanced class
rect = Rectangle(5, 10)
print(f"Area: {rect.area()}")         # 50
print(f"Perimeter: {rect.perimeter()}")  # 30
print(f"Is square? {rect.is_square()}")  # False

square = Rectangle(5, 5)
print(f"Is square? {square.is_square()}")  # True
```

**Example 3: Creating Classes Dynamically with `type`**

The `type` function can be used to create classes dynamically at runtime.

```python
# Create a class dynamically
def create_model_class(name, fields):
    def __init__(self, **kwargs):
        for field, value in kwargs.items():
            if field not in fields:
                raise AttributeError(f"Unknown field: {field}")
            setattr(self, field, value)
    
    def __repr__(self):
        field_values = ", ".join(f"{field}={getattr(self, field, None)!r}" 
                               for field in fields)
        return f"{name}({field_values})"
    
    # Create class attributes and methods
    attrs = {
        '__init__': __init__,
        '__repr__': __repr__,
        'fields': fields
    }
    
    # Create the class using type
    return type(name, (object,), attrs)

# Create a User model class with specific fields
User = create_model_class('User', ['name', 'email', 'age'])

# Create instances of the dynamic class
user1 = User(name="Alice", email="alice@example.com", age=30)
print(user1)  # User(name='Alice', email='alice@example.com', age=30)

# Create another model class
Product = create_model_class('Product', ['name', 'price', 'category'])
product1 = Product(name="Laptop", price=999.99, category="Electronics")
print(product1)  # Product(name='Laptop', price=999.99, category='Electronics')
```

#### Advanced Dynamic Class Modification with Metaclasses

Metaclasses are the "classes of classes" that control how classes are created. They provide the most powerful way to modify class behavior.

```python
class LoggingMeta(type):
    def __new__(mcs, name, bases, attrs):
        # Add logging to each method
        for attr_name, attr_value in attrs.items():
            if callable(attr_value) and not attr_name.startswith('__'):
                attrs[attr_name] = LoggingMeta.add_logging(attr_value)
        
        # Create the class
        return super().__new__(mcs, name, bases, attrs)
    
    @staticmethod
    def add_logging(method):
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            cls_name = args[0].__class__.__name__
            print(f"Calling {cls_name}.{method.__name__}")
            result = method(*args, **kwargs)
            print(f"Finished {cls_name}.{method.__name__}")
            return result
        return wrapper

# Use the metaclass
class Service(metaclass=LoggingMeta):
    def process(self, data):
        print(f"Processing {data}")
        return data.upper()
    
    def analyze(self, data):
        print(f"Analyzing {data}")
        return len(data)

# Each method call is automatically logged
service = Service()
service.process("hello")
service.analyze("world")
```

#### Best Practices for Class Decorators and Dynamic Modifications

1. **Document thoroughly**: Since class decorators and dynamic modifications can significantly change behavior, good documentation is crucial.

2. **Preserve metadata**: When replacing methods or classes, use `functools.wraps` to preserve docstrings and other metadata.

3. **Respect encapsulation**: Be careful about modifying private attributes or methods (those with names starting with underscore).

4. **Consider alternatives**: Sometimes inheritance or composition provide cleaner solutions than class decorators or dynamic modifications.

5. **Test exhaustively**: Dynamic modifications can have subtle and unexpected effects, so thorough testing is essential.

6. **Be mindful of performance**: Especially with metaclasses, which run at class definition time, not instance creation time.

7. **Use clear names**: Give decorators and metaclasses names that clearly describe their purpose.

## Module 4: Type Checking & Dynamic Typing

### 4.1 Using `isinstance()` Properly for Type Checking

While Python is dynamically typed, type checking is sometimes necessary to ensure code correctness. The `isinstance()` function provides a Pythonic way to check types at runtime.

#### Why and When to Use Type Checking

Type checking can be useful in several scenarios:

1. **Input validation**: Ensuring arguments to functions are of expected types
2. **Polymorphic behavior**: Implementing different behavior based on the type of an object
3. **Defensive programming**: Preventing type-related errors in code with many dependencies
4. **Working with external data**: Validating data from external sources (files, APIs)

However, Python's philosophy generally favors "duck typing" (if it walks like a duck and quacks like a duck, it's a duck) over explicit type checking. When possible, focus on what an object can do (its behavior) rather than what it is (its type).

#### Best Practices for Using `isinstance()`

**1. Check Against Abstract Base Classes (ABCs)**

Instead of checking against concrete implementations, check against the abstract interfaces that define the behavior you need.

```python
from collections.abc import Mapping, Sequence, Iterable

# Bad - checks for a specific implementation
def process_data(data):
    if isinstance(data, dict):
        # Process dictionary
        pass
    elif isinstance(data, list):
        # Process list
        pass

# Good - checks for the interface/behavior
def process_data_better(data):
    if isinstance(data, Mapping):
        # Process any mapping type (dict, defaultdict, OrderedDict, etc.)
        pass
    elif isinstance(data, Sequence) and not isinstance(data, str):
        # Process any sequence type except strings
        pass
```

**2. Check Multiple Types at Once**

```python
# Check if value is any numeric type
def is_number(value):
    return isinstance(value, (int, float, complex))

# Process different types of sequences
def process_sequence(seq):
    if not isinstance(seq, (list, tuple, set)):
        raise TypeError("Expected a sequence")
    # Process the sequence
```

**3. Using Type Checking for Parameter Validation**

```python
def calculate_statistics(numbers):
    """Calculate basic statistics for a list of numbers.
    
    Args:
        numbers: A sequence of numeric values
        
    Returns:
        A dictionary with min, max, sum, and average
    """
    # Type validation
    if not isinstance(numbers, Iterable):
        raise TypeError("numbers must be an iterable")
    
    # Convert to list (in case it's a generator or other iterable)
    numbers = list(numbers)
    
    # Further validation
    if not numbers:
        raise ValueError("numbers cannot be empty")
        
    if not all(isinstance(x, (int, float)) for x in numbers):
        raise TypeError("all elements must be numeric")
    
    # Calculate statistics
    return {
        'min': min(numbers),
        'max': max(numbers),
        'sum': sum(numbers),
        'average': sum(numbers) / len(numbers)
    }
```

**4. Checking for String Types**

Be careful with string type checking, as there are differences between Python 2 and 3:

```python
# Python 3 way (recommended)
def is_string(value):
    return isinstance(value, str)

# Python 2 and 3 compatible way
def is_string_py2_py3(value):
    try:
        return isinstance(value, basestring)  # Python 2
    except NameError:
        return isinstance(value, str)  # Python 3
```

**5. Type Checking in OOP**

```python
class Shape:
    def area(self):
        raise NotImplementedError("Subclasses must implement area()")

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        import math
        return math.pi * self.radius ** 2

def get_area(shape):
    """Calculate the area of a shape.
    
    Args:
        shape: An instance of Shape
        
    Returns:
        The area of the shape
    """
    if not isinstance(shape, Shape):
        raise TypeError("Expected a Shape instance")
    
    return shape.area()
```

#### Common Anti-Patterns and How to Avoid Them

**Anti-Pattern 1: Overusing `isinstance()`**

```python
# Bad: Excessive type checking
def process_value(value):
    if isinstance(value, int):
        return value * 2
    elif isinstance(value, float):
        return value * 2.0
    elif isinstance(value, str):
        try:
            return float(value) * 2
        except ValueError:
            return value + value
    elif isinstance(value, list):
        return [x * 2 for x in value]
    # And so on...

# Better: Using duck typing
def process_value_better(value):
    try:
        # Works for any numeric type or string that converts to number
        return value * 2
    except TypeError:
        # Handle the case where multiplication doesn't work
        try:
            # Check if it's iterable (except strings)
            if hasattr(value, '__iter__') and not isinstance(value, str):
                return [process_value_better(x) for x in value]
            # Default string case
            return value + value
        except (TypeError, AttributeError):
            raise TypeError(f"Cannot process value of type {type(value).__name__}")
```

**Anti-Pattern 2: Using `type()` Instead of `isinstance()`**

```python
# Bad: Using type() for type checking
def is_integer(value):
    return type(value) is int

# Good: Using isinstance() for type checking
def is_integer_better(value):
    return isinstance(value, int)

# Why it matters:
class MyInt(int):
    pass

my_int = MyInt(5)
print(is_integer(my_int))        # False - doesn't consider inheritance
print(is_integer_better(my_int))  # True - respects inheritance
```

**Anti-Pattern 3: Checking Types Instead of Capabilities**

```python
# Bad: Checking specific types
def save_data(data, file):
    if isinstance(file, str):
        with open(file, 'w') as f:
            f.write(data)
    elif isinstance(file, io.TextIOBase):
        file.write(data)
    else:
        raise TypeError("Expected a filename or file-like object")

# Better: Checking for the capability (duck typing)
def save_data_better(data, file):
    # If it's a string, treat it as a filename
    if isinstance(file, str):
        with open(file, 'w') as f:
            f.write(data)
    # If it has a write method, use that
    elif hasattr(file, 'write'):
        file.write(data)
    else:
        raise TypeError("Expected a filename or file-like object")
```

**Anti-Pattern 4: Ignoring Inheritance in Type Checks**

```python
# Bad: Not considering inheritance
def is_safe_shape(shape):
    # This only checks for direct instances of Shape, not subclasses
    return type(shape) is Shape

# Good: Considering inheritance
def is_safe_shape_better(shape):
    return isinstance(shape, Shape)
```

#### Advanced `isinstance()` Techniques

**1. Creating Custom Type Checking for Duck Typing**

```python
# Define protocols for duck typing
class Quackable:
    """Protocol class for objects that can quack."""
    
    @classmethod
    def __subclasshook__(cls, C):
        if cls is Quackable:
            if any("quack" in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented

class Duck:
    def quack(self):
        return "Quack!"

class Person:
    def quack(self):
        return "I'm pretending to be a duck!"

class Dog:
    def bark(self):
        return "Woof!"

# Check if objects follow the Quackable protocol
print(isinstance(Duck(), Quackable))    # True
print(isinstance(Person(), Quackable))  # True
print(isinstance(Dog(), Quackable))     # False
```

**2. Using `__instancecheck__` for Custom Type Checking**

```python
class NumericType(type):
    def __instancecheck__(cls, instance):
        return (
            hasattr(instance, '__int__') and
            hasattr(instance, '__float__') and
            hasattr(instance, '__add__')
        )

class Numeric(metaclass=NumericType):
    """Represents any numeric-like object."""
    pass

# Test with various types
print(isinstance(42, Numeric))              # True
print(isinstance(3.14, Numeric))            # True
print(isinstance(complex(1, 1), Numeric))   # True
print(isinstance("not a number", Numeric))  # False

# Custom numeric class
class MyNumber:
    def __init__(self, value):
        self.value = value
    
    def __int__(self):
        return int(self.value)
        
    def __float__(self):
        return float(self.value)
        
    def __add__(self, other):
        return MyNumber(self.value + other)

print(isinstance(MyNumber(10), Numeric))    # True
```

#### Type Checking in the Context of Python's Type Hints

Python 3.5+ introduced type hints, providing a way to indicate expected types in your code. While type hints are not enforced at runtime, they can be used in combination with runtime type checking for more robust code.

```python
from typing import List, Dict, Union, Optional

def process_data(items: List[int], config: Optional[Dict[str, str]] = None) -> List[int]:
    """Process a list of integers based on optional configuration."""
    if config is not None and not isinstance(config, dict):
        raise TypeError("config must be a dictionary or None")
    
    if not isinstance(items, list) or not all(isinstance(x, int) for x in items):
        raise TypeError("items must be a list of integers")
    
    # Process the data
    result = [x * 2 for x in items]
    return result
```

This approach combines the benefits of static type hints (for documentation and static analysis tools) with runtime type checking (for robustness).

### 4.2 Understanding `type()` vs `id()` vs `hash()`

Python provides several built-in functions that help inspect and identify objects: `type()`, `id()`, and `hash()`. Each serves a distinct purpose and understanding their differences is crucial for advanced Python programming.

#### The `type()` Function

The `type()` function has two main uses:
1. Getting the type of an object
2. Creating new types (classes) dynamically

##### Getting an Object's Type

```python
# Basic types
print(type(42))         # <class 'int'>
print(type("hello"))    # <class 'str'>
print(type([1, 2, 3]))  # <class 'list'>

# Custom classes
class Person:
    pass

p = Person()
print(type(p))          # <class '__main__.Person'>
print(type(Person))     # <class 'type'> - classes are instances of 'type'
```

##### How `type()` Works Internally

The `type()` function returns the object's `__class__` attribute, which points to the class that created the object. Every class in Python is an instance of the metaclass `type`, which is why `type(Person)` returns `<class 'type'>`.

```python
x = 42
print(x.__class__)      # <class 'int'>
print(type(x))          # <class 'int'>
print(x.__class__ is type(x))  # True
```

##### Creating Classes with `type()`

The three-argument form of `type()` creates a new class dynamically:

```python
# Syntax: type(name, bases, namespace)
# - name: string for the class name
# - bases: tuple of base classes
# - namespace: dictionary of attributes and methods

# Creating a class dynamically
Person = type('Person', (), {
    'greeting': 'Hello',
    'say_hello': lambda self: f"{self.greeting}, I'm a person!"
})

# Using the dynamically created class
p = Person()
print(p.say_hello())  # "Hello, I'm a person!"

# Creating a subclass dynamically
Employee = type('Employee', (Person,), {
    'greeting': 'Hi',
    'job': 'developer',
    'describe_job': lambda self: f"I work as a {self.job}"
})

e = Employee()
print(e.say_hello())     # "Hi, I'm a person!"
print(e.describe_job())  # "I work as a developer"
```

##### When to Use `type()`

1. **Type checking**: When you absolutely need to check for an exact type (though `isinstance()` is usually preferred)
2. **Metaprogramming**: Creating classes dynamically or implementing metaclasses
3. **Introspection**: Examining the types of objects in a program

#### The `id()` Function

The `id()` function returns an integer that uniquely identifies an object during its lifetime. This ID is guaranteed to be unique among simultaneously existing objects, but may be reused after an object is garbage collected.

##### How `id()` Works

In CPython (the standard Python implementation), `id()` returns the memory address of the object. However, this is an implementation detail, and other Python implementations might use different mechanisms.

```python
x = [1, 2, 3]
y = [1, 2, 3]
z = x

print(id(x))      # e.g., 140241839075272
print(id(y))      # e.g., 140241839075656 (different from x)
print(id(z))      # e.g., 140241839075272 (same as x)

# Comparing objects with is uses id() internally
print(x is y)     # False - different objects
print(x is z)     # True - same object
```

##### Object Identity vs Equality

Python distinguishes between identity (is the same object) and equality (has the same value):

```python
a = [1, 2, 3]
b = [1, 2, 3]

print(a == b)    # True - equal values
print(a is b)    # False - different objects

# Small integers and strings are interned
x = 5
y = 5
print(x is y)    # True - Python interns small integers

s1 = "hello"
s2 = "hello"
print(s1 is s2)  # True - Python interns strings (implementation-dependent)
```

##### When to Use `id()`

1. **Debugging references**: Tracking whether variables reference the same object
2. **Implementing identity-based collections**: When you need to use objects as dictionary keys based on identity
3. **Understanding caching behavior**: Investigating when Python reuses or creates new objects

#### The `hash()` Function

The `hash()` function returns the hash value of an object, used for quick comparisons in dictionaries and sets. Only immutable objects (like strings, numbers, and tuples of immutable objects) can be hashed.

##### How `hash()` Works

The hash function takes an object and returns an integer that remains constant during the object's lifetime. Objects that compare equal should have the same hash value.

```python
# Hash values for immutable types
print(hash(42))           # e.g., 42
print(hash("hello"))      # e.g., -6222589211343679025
print(hash((1, 2, 3)))    # e.g., 2528502973977326415

# Mutable objects are not hashable
try:
    hash([1, 2, 3])
except TypeError as e:
    print(e)  # "unhashable type: 'list'"
```

##### Custom Hash Implementation

You can define custom hash behavior by implementing `__hash__` and `__eq__` methods in your classes:

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
        return hash((self.x, self.y))

# Using hashable objects in sets and dictionaries
p1 = Point(1, 2)
p2 = Point(1, 2)
p3 = Point(3, 4)

point_set = {p1, p2, p3}
print(len(point_set))  # 2 - p1 and p2 are considered equal

point_dict = {p1: "first point", p3: "third point"}
print(point_dict[p2])  # "first point" - p2 hashes to the same value as p1
```

##### Implications of Hashing in Collections

Hashable objects can be used as dictionary keys and set elements because their hash values allow for O(1) lookup time:

```python
# Dictionary lookups use hash() internally
d = {'a': 1, 'b': 2}
print(d['a'])  # 1

# Set membership tests use hash() internally
s = {1, 2, 3}
print(2 in s)  # True
```

##### When to Use `hash()`

1. **Implementing hashable classes**: When you need instances to work as dictionary keys or set elements
2. **Caching results**: Implementing memorization patterns based on input hash values
3. **Building hash-based data structures**: When implementing specialized collections

#### Practical Comparisons of `type()`, `id()`, and `hash()`

**Example: Understanding Object Identity and Equality**

```python
def explore_object(obj):
    """Explore an object's type, id, and hash (if available)."""
    print(f"Object: {obj}")
    print(f"Type: {type(obj)}")
    print(f"ID: {id(obj)}")
    
    try:
        print(f"Hash: {hash(obj)}")
    except TypeError:
        print("Hash: Not hashable")
    
    print()

# Explore different objects
explore_object(42)
explore_object("hello")
explore_object([1, 2, 3])

# Explore equal but distinct objects
a = "hello"
b = "hello"
print(f"a is b: {a is b}")  # Often True for strings (due to interning)
print(f"a == b: {a == b}")  # True
print(f"id(a): {id(a)}")
print(f"id(b): {id(b)}")
print()

# Integers are also interned
x = 500
y = 500
print(f"x is y: {x is y}")  # Implementation dependent
print(f"x == y: {x == y}")  # True
print(f"id(x): {id(x)}")
print(f"id(y): {id(y)}")
```

**Example: Mutable vs. Immutable Objects**

```python
# Mutable object
list1 = [1, 2, 3]
original_id = id(list1)
list1.append(4)
print(f"After modification, same id? {id(list1) == original_id}")  # True

# Immutable object
tuple1 = (1, 2, 3)
original_id = id(tuple1)
# Creating a new tuple
tuple1 = tuple1 + (4,)
print(f"After 'modification', same id? {id(tuple1) == original_id}")  # False
```

**Example: Practical Applications of `hash()`**

```python
import functools

# Using hash for memoization
@functools.lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# This function uses hash() internally to cache results based on the argument

# Custom caching function using hash
def memoize(func):
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a key using the function arguments
        key = hash(args + tuple(sorted(kwargs.items())))
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return wrapper

@memoize
def compute_value(x, y, z):
    print(f"Computing for {x}, {y}, {z}")
    return x * y * z

# First call computes the value
result1 = compute_value(2, 3, 4)
# Second call returns cached result
result2 = compute_value(2, 3, 4)
```

### 4.3 Working with the `typing` Module

The `typing` module was introduced in Python 3.5 to support type hints, allowing developers to indicate expected types in their code. While Python remains dynamically typed, type hints serve as documentation and enable static type checking with external tools like mypy.

#### Basic Type Hints

```python
from typing import List, Dict, Tuple, Set, Optional, Union, Any

# Basic function with type hints
def greet(name: str) -> str:
    return f"Hello, {name}!"

# Function with multiple parameter types
def process_item(item_id: int, item_data: Dict[str, Any]) -> bool:
    # Process the item
    return True

# Function with container types
def get_user_scores(user_ids: List[int]) -> Dict[int, float]:
    scores = {}
    for user_id in user_ids:
        scores[user_id] = 9.5  # Simplified example
    return scores

# Optional parameters
def connect_to_database(host: str, port: Optional[int] = None) -> bool:
    if port is None:
        port = 5432  # Default port
    # Connect to database
    return True

# Union types for multiple possibilities
def process_data(data: Union[str, bytes, List[int]]) -> None:
    if isinstance(data, str):
        print(f"Processing string: {data}")
    elif isinstance(data, bytes):
        print(f"Processing bytes of length: {len(data)}")
    else:
        print(f"Processing list of integers: {data}")
```

#### Generic Types and Type Variables

Type variables allow you to create generic functions and classes that preserve type information.

```python
from typing import TypeVar, Generic, List, Dict, Optional

# Define a type variable
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

# Generic function
def first_element(items: List[T]) -> Optional[T]:
    return items[0] if items else None

# Usage
numbers = [1, 2, 3]
first_num = first_element(numbers)  # Type: Optional[int]

names = ["Alice", "Bob", "Charlie"]
first_name = first_element(names)  # Type: Optional[str]

# Generic class
class Box(Generic[T]):
    def __init__(self, content: T):
        self.content = content
    
    def get_content(self) -> T:
        return self.content
    
    def set_content(self, content: T) -> None:
        self.content = content

# Usage
int_box = Box[int](42)
str_box = Box[str]("Hello")

# int_box.set_content("string")  # Type checker would flag this as an error

# Generic dictionary-like class
class KeyValueStore(Generic[K, V]):
    def __init__(self):
        self.store: Dict[K, V] = {}
    
    def set(self, key: K, value: V) -> None:
        self.store[key] = value
    
    def get(self, key: K) -> Optional[V]:
        return self.store.get(key)
```

#### Callable Types

The `Callable` type hint is used for functions and other callable objects.

```python
from typing import Callable, List, Dict, Any, TypeVar

# Function that takes a callback
def process_items(items: List[Any], 
                  processor: Callable[[Any], Any]) -> List[Any]:
    return [processor(item) for item in items]

# More specific callable with multiple arguments
def apply_operation(a: int, b: int, 
                   operation: Callable[[int, int], int]) -> int:
    return operation(a, b)

# Usage
result = apply_operation(5, 3, lambda x, y: x + y)  # 8

# Higher-order function with generic types
T = TypeVar('T')
R = TypeVar('R')

def map_values(items: List[T], 
               func: Callable[[T], R]) -> List[R]:
    return [func(item) for item in items]

# Usage preserves type information
numbers = [1, 2, 3, 4]
squared = map_values(numbers, lambda x: x ** 2)  # List[int]
names = ["alice", "bob", "charlie"]
upper_names = map_values(names, str.upper)  # List[str]
```

#### Protocol Classes for Structural Subtyping

The `Protocol` class, introduced in Python 3.8, enables structural subtyping – focusing on what an object can do rather than its specific type.

```python
from typing import Protocol, List, Iterator, runtime_checkable

# Define a protocol
class Drawable(Protocol):
    def draw(self) -> None:
        ...

# Classes that implement the protocol without explicit inheritance
class Circle:
    def __init__(self, radius: float):
        self.radius = radius
    
    def draw(self) -> None:
        print(f"Drawing a circle with radius {self.radius}")

class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    
    def draw(self) -> None:
        print(f"Drawing a rectangle {self.width}x{self.height}")

# Function that accepts any Drawable
def render(item: Drawable) -> None:
    item.draw()

# Usage - no explicit inheritance needed
render(Circle(5.0))
render(Rectangle(10.0, 20.0))

# Runtime-checkable protocol
@runtime_checkable
class Sized(Protocol):
    def __len__(self) -> int:
        ...

# Check if objects implement the protocol
print(isinstance([], Sized))       # True
print(isinstance("hello", Sized))  # True
print(isinstance(42, Sized))       # False
```

#### Type Aliases and NewType

Type aliases provide convenient shorthand for complex types, while `NewType` creates distinct types for added type safety.

```python
from typing import List, Dict, Tuple, NewType, Union

# Type aliases
UserId = int
UserName = str
UserScore = float
UserRecord = Dict[str, Union[UserId, UserName, UserScore]]

# Function using type aliases
def get_user_data(user_id: UserId) -> UserRecord:
    return {
        "id": user_id,
        "name": f"User {user_id}",
        "score": 95.5
    }

# NewType creates a distinct type
AdminId = NewType('AdminId', int)

# Functions specific to each type
def get_user(user_id: UserId) -> UserRecord:
    return get_user_data(user_id)

def get_admin(admin_id: AdminId) -> UserRecord:
    # Additional admin-specific logic
    record = get_user_data(admin_id)
    record["is_admin"] = True
    return record

# Usage
user1 = get_user(42)           # OK
admin1 = get_admin(AdminId(100))  # OK

# Type checker would flag these as errors:
# admin2 = get_admin(42)       # Error: Expected AdminId, got int
# user2 = get_user(AdminId(100))  # Error: Expected UserId, got AdminId
```

#### Working with Literal and Final Types

`Literal` and `Final` types (Python 3.8+) provide more specific type constraints.

```python
from typing import Literal, Final, Dict, List, Union

# Literal type for specific values
Direction = Literal['north', 'south', 'east', 'west']

def move(direction: Direction, steps: int) -> None:
    print(f"Moving {steps} steps {direction}")

# Usage
move('north', 3)  # OK
# move('up', 3)   # Type checker error: 'up' is not a valid Direction

# Constants with Final
MAX_USERS: Final = 100
API_KEY: Final[str] = "abc123"

# Cannot reassign Final variables
# MAX_USERS = 200  # Type checker error

# Function overloads
from typing import overload

@overload
def process_data(data: str) -> str: ...

@overload
def process_data(data: List[int]) -> int: ...

def process_data(data: Union[str, List[int]]) -> Union[str, int]:
    if isinstance(data, str):
        return data.upper()
    else:
        return sum(data)

# Usage
result1 = process_data("hello")  # Type: str
result2 = process_data([1, 2, 3])  # Type: int
```

#### Advanced Type Checking with Custom Types

```python
from typing import TypeVar, Generic, TypedDict, List, Dict, Any, cast

# TypedDict for structured dictionaries
class MovieInfo(TypedDict):
    title: str
    year: int
    director: str
    rating: float
    genres: List[str]

# Function that uses TypedDict
def print_movie(movie: MovieInfo) -> None:
    print(f"{movie['title']} ({movie['year']}) - {movie['rating']}/10")
    print(f"Directed by {movie['director']}")
    print(f"Genres: {', '.join(movie['genres'])}")

# Usage
inception: MovieInfo = {
    "title": "Inception",
    "year": 2010,
    "director": "Christopher Nolan",
    "rating": 8.8,
    "genres": ["Action", "Sci-Fi", "Thriller"]
}

print_movie(inception)

# Type casting
def get_data() -> Dict[str, Any]:
    return {"name": "Alice", "age": 30, "scores": [95, 88, 92]}

# Using cast to tell the type checker this is a specific type
user_data = cast(MovieInfo, get_data())  # Type checker will trust this assertion
```

#### Best Practices for Using the `typing` Module

1. **Start Gradually**: Add type hints to new code or critical parts first, then expand coverage.

2. **Use Type Checkers**: Tools like mypy, pyright, or Pytype can analyze your code for type errors.

3. **Type Hint Comments for Python 3.5 or Earlier**:
```python
def greet(name):  # type: (str) -> str
    return f"Hello, {name}!"
```

4. **Be Cautious with Dynamic Features**: Some Python patterns are hard to type correctly. Use `Any` when necessary, but try to be as specific as possible.

5. **Document Type Expectations**: Add type information in docstrings to complement type hints.

6. **Use Protocols for Duck Typing**: Prefer structural subtyping (Protocols) over nominal typing when emphasizing behavior.

7. **TypeVar Constraints**: Use bounds and constraints with TypeVar for more specific generic types.
```python
from typing import TypeVar, List

# TypeVar with constraints
NumericT = TypeVar('NumericT', int, float, complex)

def square(x: NumericT) -> NumericT:
    return x * x
```

8. **Import Types from `__future__`**: In Python 3.7+, use string literals for forward references.
```python
from __future__ import annotations

class Tree:
    def __init__(self, value: int, left: Tree = None, right: Tree = None):
        self.value = value
        self.left = left
        self.right = right
```

9. **Avoid Circular Imports**: Use string literals for type hints that would cause circular imports.

10. **Type Stubs for Libraries**: For libraries without type hints, create or use stub files (.pyi).

## Module 5: Memory Management & Optimization

### 5.1 Understanding How Python Manages Memory

Python's memory management is automatic, handling allocation and deallocation for you. Understanding how it works can help you write more efficient code and avoid memory leaks.

#### Python's Memory Architecture

Python's memory management involves several layers:

1. **The Memory Manager**: Handles object allocation and deallocation
2. **Private Heap Space**: All Python objects and data structures are stored in a private heap
3. **Object Allocator**: Handles small object allocation
4. **Garbage Collector**: Reclaims memory from objects that are no longer reachable

##### Memory Allocation

When you create objects in Python, memory is allocated from the private heap:

```python
# Each of these allocates memory
x = 42          # Integer object
name = "Alice"  # String object
numbers = [1, 2, 3]  # List object with three integer objects
```

##### Object References and Reference Counting

Python uses reference counting to track how many references point to each object. When the count reaches zero, the object is deallocated.

```python
import sys

# Create an object and check its reference count
x = [1, 2, 3]
print(sys.getrefcount(x))  # 2 (one for x, one for the argument to getrefcount)

# Create another reference to the same object
y = x
print(sys.getrefcount(x))  # 3

# Remove a reference
y = None
print(sys.getrefcount(x))  # 2

# Function that demonstrates reference counting
def ref_count_demo():
    numbers = [1, 2, 3]
    
    # Local reference to numbers
    print(f"Inside function: {sys.getrefcount(numbers)}")
    
    # Return the object, passing a reference to the caller
    return numbers

result = ref_count_demo()
print(f"After function: {sys.getrefcount(result)}")
```

##### The Garbage Collector and Cyclic References

Reference counting alone can't handle circular references (objects that reference each other). Python's garbage collector identifies and cleans up these cycles.

```python
import gc

# Create objects with circular references
def create_cycle():
    list1 = []
    list2 = []
    
    # Create a cycle
    list1.append(list2)
    list2.append(list1)
    
    # Local variables list1 and list2 go out of scope here
    # but the objects remain in memory due to circular references

# Create and discard cycles
for _ in range(10):
    create_cycle()

# Force garbage collection
collected = gc.collect()
print(f"Garbage collector: collected {collected} objects.")
```

#### Memory Profiling Tools

Python provides several modules for inspecting memory usage:

##### The `sys` Module

```python
import sys

# Get the size of an object in bytes
x = 42
print(f"Size of int: {sys.getsizeof(x)} bytes")

y = "Hello, world!"
print(f"Size of string: {sys.getsizeof(y)} bytes")

z = [1, 2, 3, 4, 5]
print(f"Size of list: {sys.getsizeof(z)} bytes")

# Note: getsizeof only shows the direct memory usage of the container,
# not the memory used by contained objects
```

##### The `tracemalloc` Module

`tracemalloc`, introduced in Python 3.4, provides detailed memory allocation tracking.

```python
import tracemalloc

# Start tracking memory allocations
tracemalloc.start()

# Create some objects
x = [1] * 1000000  # A list with 1 million items
y = "a" * 1000000  # A string with 1 million characters
z = {i: i*i for i in range(100000)}  # A dictionary with 100,000 items

# Get current memory usage
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 10 memory usages ]")
for stat in top_stats[:10]:
    print(stat)
```

##### Memory Profilers

Third-party libraries like `memory_profiler` provide more advanced memory profiling.

```python
# Install with: pip install memory_profiler
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Create a large list
    large_list = [i for i in range(10000000)]
    
    # Create a large dictionary
    large_dict = {i: i*i for i in range(1000000)}
    
    # Process the data
    result = sum(large_list) + sum(large_dict.values())
    
    return result

# Running this function with @profile will show memory usage
result = memory_intensive_function()
```

#### Memory Optimization Techniques

##### 1. Using Generators Instead of Lists

```python
import sys

# Memory-intensive approach
def sum_large_list_memory_intensive(n):
    # Creates a full list in memory
    numbers = [i for i in range(n)]
    return sum(numbers)

# Memory-efficient approach
def sum_large_list_memory_efficient(n):
    # Uses a generator expression - no full list in memory
    return sum(i for i in range(n))

# Compare memory usage
n = 10**7

# With list comprehension
large_list = [i for i in range(n)]
print(f"List size: {sys.getsizeof(large_list)} bytes")

# With generator expression
large_gen = (i for i in range(n))
print(f"Generator size: {sys.getsizeof(large_gen)} bytes")
```

##### 2. Using `__slots__` to Reduce Memory Usage

```python
import sys

# Regular class without __slots__
class PersonRegular:
    def __init__(self, name, age, email):
        self.name = name
        self.age = age
        self.email = email

# Class with __slots__
class PersonSlots:
    __slots__ = ['name', 'age', 'email']
    
    def __init__(self, name, age, email):
        self.name = name
        self.age = age
        self.email = email

# Compare memory usage
regular_person = PersonRegular("Alice", 30, "alice@example.com")
slots_person = PersonSlots("Alice", 30, "alice@example.com")

print(f"Size of regular person: {sys.getsizeof(regular_person)} bytes")
print(f"Size of slots person: {sys.getsizeof(slots_person)} bytes")

# Create many instances to see the cumulative effect
regular_people = [PersonRegular(f"Person {i}", i, f"person{i}@example.com") 
                  for i in range(10000)]
slots_people = [PersonSlots(f"Person {i}", i, f"person{i}@example.com") 
                for i in range(10000)]

# Note: sys.getsizeof doesn't account for the size of all attributes,
# so the difference is actually larger than what is reported
```

##### 3. Object Pooling and Reuse

```python
# Simple object pool implementation
class ObjectPool:
    def __init__(self, create_func, pool_size=10):
        self.create_func = create_func
        self.pool = [create_func() for _ in range(pool_size)]
    
    def get(self):
        if self.pool:
            return self.pool.pop()
        return self.create_func()
    
    def release(self, obj):
        self.pool.append(obj)

# Example usage with database connections
class DatabaseConnection:
    def __init__(self):
        print("Creating new database connection")
        # In a real scenario, this would connect to a database
    
    def query(self, sql):
        print(f"Executing: {sql}")
        # In a real scenario, this would execute a query
    
    def close(self):
        print("Closing connection")
        # In a real scenario, this would close the connection

# Create a connection pool
connection_pool = ObjectPool(DatabaseConnection)

# Use a connection
conn = connection_pool.get()
conn.query("SELECT * FROM users")
connection_pool.release(conn)  # Return to the pool instead of destroying

# Use another connection (reuses the one we just released)
conn2 = connection_pool.get()
conn2.query("SELECT * FROM products")
connection_pool.release(conn2)
```

##### 4. Using More Efficient Data Structures

```python
import sys
from array import array

# List of integers (flexible but uses more memory)
int_list = [i for i in range(10000)]
print(f"List size: {sys.getsizeof(int_list)} bytes")

# Array of integers (less flexible but more memory-efficient)
int_array = array('i', [i for i in range(10000)])
print(f"Array size: {sys.getsizeof(int_array)} bytes")

# For dictionaries with string keys, consider using ids instead
string_keys = {f"user_{i}": i for i in range(10000)}
integer_keys = {i: i for i in range(10000)}

print(f"Dict with string keys: {sys.getsizeof(string_keys)} bytes")
print(f"Dict with integer keys: {sys.getsizeof(integer_keys)} bytes")
```

##### 5. Using Weak References

Weak references allow you to reference objects without preventing garbage collection.

```python
import weakref
import gc

class LargeObject:
    def __init__(self, name):
        self.name = name
        # Imagine this is a large object
        self.data = [0] * 1000000
    
    def __del__(self):
        print(f"{self.name} is being deleted")

# Create an object
large_obj = LargeObject("Large Object")

# Create a weak reference to it
weak_ref = weakref.ref(large_obj)

# Accessing the object through the weak reference
if weak_ref() is not None:
    print(f"Object name: {weak_ref().name}")

# Remove the strong reference
large_obj = None

# Force garbage collection
gc.collect()

# The weak reference now returns None
print(f"After deletion: {weak_ref()}")
```

### 5.2 Using `__slots__` to Reduce Memory Usage in Large Objects

The `__slots__` attribute is a powerful mechanism for reducing the memory footprint of Python classes, especially when you need to create many instances.

#### How Python Classes Store Attributes

By default, Python uses a dictionary (`__dict__`) to store object attributes, providing flexibility but consuming more memory.

```python
class RegularClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

obj = RegularClass(1, 2)
print(obj.__dict__)  # {'x': 1, 'y': 2}

# Add a new attribute at runtime
obj.z = 3
print(obj.__dict__)  # {'x': 1, 'y': 2, 'z': 3}
```

#### How `__slots__` Works

When you define `__slots__` in a class, Python uses a more efficient storage mechanism instead of a dictionary, significantly reducing memory usage.

```python
class SlottedClass:
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

obj = SlottedClass(1, 2)

# No __dict__ attribute
try:
    print(obj.__dict__)
except AttributeError as e:
    print(f"AttributeError: {e}")

# Cannot add new attributes
try:
    obj.z = 3
except AttributeError as e:
    print(f"AttributeError: {e}")
```

#### Memory Comparison

```python
import sys

# Regular class
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Class with __slots__
class PointSlots:
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Create instances
p1 = Point(3, 4)
p2 = PointSlots(3, 4)

# Compare memory usage
print(f"Memory used by Point instance: {sys.getsizeof(p1) + sys.getsizeof(p1.__dict__)} bytes")
print(f"Memory used by PointSlots instance: {sys.getsizeof(p2)} bytes")

# Create many instances to see the cumulative effect
n = 1_000_000
points = [Point(i, i) for i in range(n)]
points_slots = [PointSlots(i, i) for i in range(n)]

# Note: This is just an illustrative example
# In a real program, you'd use memory_profiler or similar tools
```

#### Advanced `__slots__` Usage

##### Inheritance with `__slots__`

```python
class Parent:
    __slots__ = ['a', 'b']
    
    def __init__(self, a, b):
        self.a = a
        self.b = b

# Child inherits slots from parent
class Child(Parent):
    __slots__ = ['c']
    
    def __init__(self, a, b, c):
        super().__init__(a, b)
        self.c = c

# Child has slots but also uses __dict__
class ChildWithDict(Parent):
    def __init__(self, a, b, z):
        super().__init__(a, b)
        self.z = z  # Uses __dict__ for z

# Test slot inheritance
child = Child(1, 2, 3)
print(f"Child attributes: a={child.a}, b={child.b}, c={child.c}")

child_with_dict = ChildWithDict(1, 2, 99)
print(f"ChildWithDict attributes: a={child_with_dict.a}, b={child_with_dict.b}, z={child_with_dict.z}")
print(f"ChildWithDict __dict__: {child_with_dict.__dict__}")  # Only contains 'z'
```

##### Including `__dict__` in `__slots__`

```python
class FlexibleSlots:
    __slots__ = ['x', 'y', '__dict__']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

# This class has the memory benefits of slots for x and y,
# but also allows dynamic attribute addition
obj = FlexibleSlots(1, 2)
obj.z = 3  # This goes in __dict__

print(f"Attributes in slots: x={obj.x}, y={obj.y}")
print(f"Attributes in __dict__: {obj.__dict__}")
```

##### Using `__slots__` with Properties and Methods

```python
class Vector:
    __slots__ = ['_x', '_y']
    
    def __init__(self, x, y):
        self._x = x
        self._y = y
    
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("x must be a number")
        self._x = value
    
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("y must be a number")
        self._y = value
    
    @property
    def magnitude(self):
        return (self._x ** 2 + self._y ** 2) ** 0.5
    
    def __repr__(self):
        return f"Vector({self._x}, {self._y})"

# Test the class
v = Vector(3, 4)
print(v)
print(f"Magnitude: {v.magnitude}")

v.x = 5
print(v)
print(f"New magnitude: {v.magnitude}")
```

#### Best Practices and Considerations

1. **When to Use `__slots__`**:
   - When you need to create many instances of a class
   - When memory usage is a concern
   - When you know in advance all the attributes an instance will need

2. **When Not to Use `__slots__`**:
   - When you need dynamic attribute assignment
   - When working with metaclasses or certain types of introspection
   - In small applications where memory optimization is not critical

3. **Pitfalls to Avoid**:
   - Don't use `__slots__` in a parent class without considering the impact on child classes
   - Be aware that `__slots__` affects pickling behavior
   - Remember that descriptors like `@property` still work with `__slots__`

4. **Measuring Impact**:
   Always measure the actual memory impact in your specific application, as the benefit varies depending on the number of instances and attributes.

### 5.3 Difference Between Shallow Copy vs Deep Copy

Python provides two ways to copy objects: shallow copying and deep copying. Understanding the difference is crucial for avoiding unexpected behavior and bugs related to object references.

#### Basic Copy Operations

```python
original_list = [1, [2, 3], 4]

# Simple assignment - creates a reference, not a copy
reference = original_list

# Modify through the reference
reference[0] = 99
print(f"Original after reference modification: {original_list}")  # [99, [2, 3], 4]
```

#### Shallow Copy with `copy.copy()`

A shallow copy creates a new object but inserts references to the same objects contained in the original.

```python
import copy

original_list = [1, [2, 3], 4]

# Create a shallow copy
shallow_copy = copy.copy(original_list)

# Modify the shallow copy at the top level
shallow_copy[0] = 99
print(f"Original after shallow top-level mod: {original_list}")  # [1, [2, 3], 4] - unchanged

# Modify a nested object in the shallow copy
shallow_copy[1][0] = 99
print(f"Original after shallow nested mod: {original_list}")  # [1, [99, 3], 4] - changed!
```

#### Deep Copy with `copy.deepcopy()`

A deep copy creates a new object and recursively inserts copies of all objects contained in the original.

```python
import copy

original_list = [1, [2, 3], 4]

# Create a deep copy
deep_copy = copy.deepcopy(original_list)

# Modify the deep copy at the top level
deep_copy[0] = 99
print(f"Original after deep top-level mod: {original_list}")  # [1, [2, 3], 4] - unchanged

# Modify a nested object in the deep copy
deep_copy[1][0] = 99
print(f"Original after deep nested mod: {original_list}")  # [1, [2, 3], 4] - still unchanged
```

#### How Copy Operations Work Internally

Python's copying mechanisms rely on special methods that objects can implement:

- `__copy__`: For shallow copying
- `__deepcopy__`: For deep copying

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"
    
    def __copy__(self):
        print("__copy__ called")
        return Point(self.x, self.y)
    
    def __deepcopy__(self, memo):
        print("__deepcopy__ called")
        # memo is a dictionary that keeps track of objects already copied
        # to avoid infinite recursion with circular references
        return Point(self.x, self.y)

# Test copying
p1 = Point(1, 2)
p1_shallow = copy.copy(p1)
p1_deep = copy.deepcopy(p1)

print(f"Original: {p1}")
print(f"Shallow copy: {p1_shallow}")
print(f"Deep copy: {p1_deep}")
```

#### Copying Complex Objects

```python
class Person:
    def __init__(self, name, age, friends=None):
        self.name = name
        self.age = age
        self.friends = friends if friends is not None else []
    
    def __repr__(self):
        return f"Person({self.name}, {self.age}, {self.friends})"

# Create some people
alice = Person("Alice", 30)
bob = Person("Bob", 25)
charlie = Person("Charlie", 35)

# Add friends
alice.friends = [bob, charlie]
bob.friends = [alice]
charlie.friends = [alice, bob]

# Shallow copy
alice_shallow = copy.copy(alice)
alice_shallow.age = 31  # Modify the shallow copy

print(f"Original Alice: {alice}")
print(f"Shallow copy: {alice_shallow}")

# The age changed, but friends are the same objects
print(f"Original Alice's friend Bob: {id(alice.friends[0])}")
print(f"Shallow copy's friend Bob: {id(alice_shallow.friends[0])}")

# Deep copy
alice_deep = copy.deepcopy(alice)
alice_deep.age = 32  # Modify the deep copy
alice_deep.friends[0].name = "Robert"  # Modify a friend in the deep copy

print(f"Original Alice: {alice}")
print(f"Deep copy: {alice_deep}")

# The friends are completely different objects
print(f"Original Alice's friend: {alice.friends[0].name}")  # Still "Bob"
print(f"Deep copy's friend: {alice_deep.friends[0].name}")  # "Robert"
```

#### Copy Methods for Built-in Types

Different built-in types have different ways of creating copies:

```python
# List copying
original_list = [1, 2, 3]
# Methods:
list_copy_1 = original_list.copy()  # list method
list_copy_2 = list(original_list)   # constructor
list_copy_3 = original_list[:]      # slicing

# Dictionary copying
original_dict = {'a': 1, 'b': 2}
# Methods:
dict_copy_1 = original_dict.copy()  # dict method
dict_copy_2 = dict(original_dict)   # constructor

# Set copying
original_set = {1, 2, 3}
# Methods:
set_copy_1 = original_set.copy()    # set method
set_copy_2 = set(original_set)      # constructor
```

#### Performance Considerations

```python
import timeit
import copy

# Setup a complex nested structure
setup = """
data = [
    [1, 2, 3, [4, 5]],
    {'a': 1, 'b': [2, 3], 'c': {'d': 4}},
    (1, 2, [3, 4])
]
import copy
"""

# Compare different copy methods
print("Shallow copy time:")
print(timeit.timeit("copy.copy(data)", setup=setup, number=100000))

print("Deep copy time:")
print(timeit.timeit("copy.deepcopy(data)", setup=setup, number=100000))
```

#### Common Pitfalls and Best Practices

1. **Forgetting Nested Structures**:
   Remember that shallow copies only create new top-level containers. Nested mutable objects are shared references.

2. **Circular References**:
   Be careful with circular references (objects that reference each other). Deep copying handles them correctly, but they can cause issues with custom `__deepcopy__` implementations.

3. **Performance Impact**:
   Deep copying can be expensive for large structures. Use shallow copies when possible, and deep copies only when necessary.

4. **Custom Objects**:
   Implement `__copy__` and `__deepcopy__` methods for custom objects to control their copy behavior.

5. **Immutable vs. Mutable**:
   Shallow copying is often sufficient for objects containing only immutable values (numbers, strings, tuples of immutables).

**Example: Working with Circular References**

```python
import copy

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
        self.prev = None
    
    def __repr__(self):
        return f"Node({self.value})"

# Create a circular linked list
node1 = Node(1)
node2 = Node(2)
node3 = Node(3)

node1.next = node2
node2.next = node3
node3.next = node1  # Circular reference

node3.prev = node2
node2.prev = node1
node1.prev = node3  # Another circular reference

# Deep copy handles circular references correctly
nodes_copy = copy.deepcopy(node1)

print(f"Original: {node1.value} -> {node1.next.value} -> {node1.next.next.value} -> {node1.next.next.next.value}")
print(f"Copy: {nodes_copy.value} -> {nodes_copy.next.value} -> {nodes_copy.next.next.value} -> {nodes_copy.next.next.next.value}")

# Verify they're different objects
print(f"Original object ID: {id(node1)}")
print(f"Copy object ID: {id(nodes_copy)}")
```

## Module 6: Error Handling & Custom Exceptions

### 6.1 Best Practices for `try/except/else/finally`

Error handling is a critical aspect of writing robust Python code. The `try/except/else/finally` blocks provide a comprehensive mechanism for handling exceptions and ensuring proper resource management.

#### Basic Exception Handling Structure

```python
try:
    # Code that might raise an exception
    result = 10 / 0
except ZeroDivisionError:
    # Code to handle the specific exception
    print("Cannot divide by zero!")
except (TypeError, ValueError) as e:
    # Handling multiple exception types with an error variable
    print(f"Input error: {e}")
except Exception as e:
    # Fallback for other exceptions
    print(f"Unexpected error: {e}")
else:
    # Code to run if no exceptions were raised
    print(f"Result: {result}")
finally:
    # Code that always runs, regardless of exceptions
    print("Execution completed")
```

#### The Execution Flow of `try/except/else/finally`

1. The code in the `try` block is executed.
2. If an exception occurs, Python looks for a matching `except` block.
3. If the exception is caught, the corresponding `except` block runs.
4. If no exception occurs, the `else` block runs (if present).
5. The `finally` block always runs, regardless of whether an exception occurred.

```python
def demonstrate_flow(divisor):
    print(f"\nTrying with divisor: {divisor}")
    resource = None
    try:
        print("Entering try block")
        resource = "Resource allocated"
        result = 100 / divisor
        print(f"Division successful: 100 / {divisor} = {result}")
    except ZeroDivisionError:
        print("Caught ZeroDivisionError")
        return "Error result"  # Note: finally still runs after return
    else:
        print("Entering else block")
        return result  # Note: finally still runs after return
    finally:
        print(f"Entering finally block, cleaning up: {resource}")
        resource = None
        print("Resource released")

# Test with valid input
print(f"Result: {demonstrate_flow(5)}")

# Test with zero (causes exception)
print(f"Result: {demonstrate_flow(0)}")
```

#### Exception Hierarchy and Exception Types

Python's exceptions form a hierarchy, with `BaseException` at the top. Understanding this hierarchy helps you catch exceptions at the appropriate level of specificity.

```python
# Common exception hierarchy
# BaseException
#  +-- SystemExit, KeyboardInterrupt, GeneratorExit
#  +-- Exception
#       +-- StopIteration, StopAsyncIteration
#       +-- ArithmeticError
#       |    +-- FloatingPointError
#       |    +-- OverflowError
#       |    +-- ZeroDivisionError
#       +-- AssertionError
#       +-- AttributeError
#       +-- BufferError
#       +-- EOFError
#       +-- ImportError
#       |    +-- ModuleNotFoundError
#       +-- LookupError
#       |    +-- IndexError
#       |    +-- KeyError
#       +-- MemoryError
#       +-- NameError
#       +-- OSError
#       |    +-- BlockingIOError, ChildProcessError, ConnectionError, ...
#       +-- ReferenceError
#       +-- RuntimeError
#       |    +-- NotImplementedError, RecursionError
#       +-- SyntaxError
#       |    +-- IndentationError
#       |         +-- TabError
#       +-- SystemError
#       +-- TypeError
#       +-- ValueError
#       |    +-- UnicodeError
#       |         +-- UnicodeDecodeError, UnicodeEncodeError, ...
#       +-- Warning
#            +-- DeprecationWarning, FutureWarning, UserWarning, ...
```

##### Choosing the Right Exception Type

```python
def process_data(data):
    if not data:
        raise ValueError("Empty data cannot be processed")
    
    if not isinstance(data, list):
        raise TypeError("Data must be a list")
    
    try:
        return [float(item) for item in data]
    except ValueError as e:
        raise ValueError(f"Invalid numeric data: {e}") from e

# Test with different inputs
try:
    process_data([])
except Exception as e:
    print(f"Test 1: {type(e).__name__}: {e}")

try:
    process_data("not a list")
except Exception as e:
    print(f"Test 2: {type(e).__name__}: {e}")

try:
    process_data([1, 2, "three"])
except Exception as e:
    print(f"Test 3: {type(e).__name__}: {e}")
```

#### Best Practices for Exception Handling

##### 1. Be Specific About What Exceptions You Catch

```python
# Bad practice - catching all exceptions
try:
    value = int(input("Enter a number: "))
    result = 100 / value
except Exception as e:  # Too broad
    print(f"Error: {e}")

# Good practice - catching specific exceptions
try:
    value = int(input("Enter a number: "))
    result = 100 / value
except ValueError:
    print("Please enter a valid integer")
except ZeroDivisionError:
    print("Cannot divide by zero")
```

##### 2. Use `else` for Code That Should Run Only If No Exception Occurs

```python
def read_data(filename):
    try:
        with open(filename, 'r') as file:
            data = file.read()
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None
    except PermissionError:
        print(f"No permission to read {filename}")
        return None
    else:
        # This only runs if no exception was raised
        print(f"Successfully read {len(data)} bytes from {filename}")
        return data
```

##### 3. Use `finally` for Cleanup Code That Must Always Run

```python
def process_file(filename):
    file = None
    try:
        file = open(filename, 'r')
        data = file.read()
        return data
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None
    finally:
        if file:
            file.close()
            print(f"File {filename} closed")
```

##### 4. Using Context Managers (`with` Statements) for Resource Management

```python
# Better approach than the previous example - context manager handles closing
def process_file_with_context(filename):
    try:
        with open(filename, 'r') as file:
            data = file.read()
            return data
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None
    # No need for finally to close the file - the context manager does it
```

##### 5. Don't Silence Exceptions Without Good Reason

```python
# Bad practice - silencing exceptions
try:
    process_data()
except Exception:
    pass  # Silently ignore all errors

# Good practice - log exceptions even if you handle them
import logging

try:
    process_data()
except Exception as e:
    logging.error(f"Error processing data: {e}")
    # Then handle or re-raise as appropriate
```

##### 6. Re-raising Exceptions

```python
def validate_age(age):
    try:
        age = int(age)
        if age < 0 or age > 120:
            raise ValueError("Age must be between 0 and 120")
        return age
    except ValueError as e:
        # Add context to the exception
        raise ValueError(f"Invalid age format: {e}") from e

# Using exception chaining
try:
    validate_age("not a number")
except ValueError as e:
    print(f"Error: {e}")
    if e.__cause__:
        print(f"Caused by: {e.__cause__}")
```

##### 7. Using `try/except` as a Conditional

```python
# Sometimes more readable than explicit checking
def get_value(dictionary, key):
    try:
        return dictionary[key]
    except KeyError:
        return None

# Using get() is better in this specific case
def get_value_better(dictionary, key):
    return dictionary.get(key)
```

##### 8. Avoid Using Exceptions for Flow Control

```python
# Bad practice - using exceptions for normal flow control
def find_index(items, target):
    try:
        return items.index(target)
    except ValueError:
        return -1

# Better practice - using normal flow control
def find_index_better(items, target):
    if target in items:
        return items.index(target)
    return -1
```

#### Advanced Exception Handling Techniques

##### 1. Custom Exception Behavior with `__context__` and `__cause__`

```python
def process_user_input():
    try:
        # First level try/except
        try:
            value = int(input("Enter a positive number: "))
            if value <= 0:
                raise ValueError("Number must be positive")
            return value
        except ValueError as e:
            # This becomes e.__context__ in the outer exception
            print("Invalid input, using default value")
            return 42
    except Exception as outer_e:
        # Access the inner exception's context
        print(f"Outer exception: {outer_e}")
        if outer_e.__context__:
            print(f"Inner exception context: {outer_e.__context__}")
        
        # Explicitly setting cause with 'raise ... from'
        raise RuntimeError("Processing failed") from outer_e
```

##### 2. Exception Handling in Generators and Context Managers

```python
def line_processor(filename):
    try:
        with open(filename, 'r') as file:
            for i, line in enumerate(file, 1):
                try:
                    # Process each line
                    yield line.strip().upper()
                except Exception as e:
                    print(f"Error processing line {i}: {e}")
                    # Skip this line but continue processing
                    continue
    except FileNotFoundError:
        print(f"File {filename} not found")
        yield f"ERROR: {filename} not found"
    except Exception as e:
        print(f"Error reading file: {e}")
        yield f"ERROR: {str(e)}"

# Using the generator with exception handling
for processed_line in line_processor("sample.txt"):
    print(processed_line)
```

##### 3. Cleanup with `finally` and Context Managers

```python
class Resource:
    def __init__(self, name):
        self.name = name
        print(f"Resource {name} initialized")
    
    def use(self):
        print(f"Using resource {self.name}")
    
    def cleanup(self):
        print(f"Resource {self.name} cleaned up")
    
    # Context manager protocol
    def __enter__(self):
        print(f"Entering context for {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Exiting context for {self.name}")
        self.cleanup()
        # Return False to propagate exceptions, True to suppress them
        return False

# Manual cleanup with try/finally
def use_resource_manually():
    resource = Resource("Manual")
    try:
        resource.use()
        # Simulate an error
        raise ValueError("Simulated error")
    finally:
        resource.cleanup()

# Automatic cleanup with context manager
def use_resource_context():
    with Resource("Context") as resource:
        resource.use()
        # Simulate an error
        raise ValueError("Simulated error")

# Test both approaches
try:
    use_resource_manually()
except ValueError as e:
    print(f"Caught in manual: {e}")

try:
    use_resource_context()
except ValueError as e:
    print(f"Caught in context: {e}")
```

### 6.2 Creating Custom Exceptions for Better Debugging

Creating custom exceptions allows you to provide more specific error information and improve debugging. Well-designed custom exceptions make your code more maintainable and help users of your code understand what went wrong.

#### Why Create Custom Exceptions?

1. **Specificity**: They provide more specific information about what went wrong
2. **Semantics**: They communicate the meaning of the error, not just the mechanism
3. **Abstraction**: They hide implementation details while exposing appropriate error information
4. **Organization**: They help organize error handling logic
5. **Documentation**: They serve as self-documenting code for error cases

#### Creating a Basic Custom Exception

```python
# Basic custom exception
class MyCustomError(Exception):
    """Base class for exceptions in this module."""
    pass

# More specific custom exceptions
class ValueTooLargeError(MyCustomError):
    """Raised when the input value is too large."""
    pass

class ValueTooSmallError(MyCustomError):
    """Raised when the input value is too small."""
    pass

# Using custom exceptions
def process_value(value):
    if value > 100:
        raise ValueTooLargeError("Value cannot exceed 100")
    if value < 0:
        raise ValueTooSmallError("Value cannot be negative")
    return value * 2

# Handling custom exceptions
try:
    result = process_value(150)
except ValueTooLargeError as e:
    print(f"Large value error: {e}")
except ValueTooSmallError as e:
    print(f"Small value error: {e}")
except MyCustomError as e:
    # Catch any other custom errors
    print(f"Other custom error: {e}")
except Exception as e:
    # Catch all other exceptions
    print(f"Unexpected error: {e}")
```

#### Designing an Exception Hierarchy

A well-designed exception hierarchy helps organize and structure error handling.

```python
# Base exception for the application/module
class AppError(Exception):
    """Base class for all exceptions in this application."""
    pass

# Category of exceptions
class ConfigError(AppError):
    """Base class for configuration-related exceptions."""
    pass

# Specific exceptions
class ConfigFileNotFoundError(ConfigError):
    """Raised when the configuration file is not found."""
    pass

class ConfigFormatError(ConfigError):
    """Raised when the configuration format is invalid."""
    def __init__(self, message, line_number=None, line_content=None):
        self.line_number = line_number
        self.line_content = line_content
        if line_number is not None and line_content is not None:
            message = f"{message} at line {line_number}: {line_content}"
        super().__init__(message)

# Another category
class DataError(AppError):
    """Base class for data-related exceptions."""
    pass

class DataValidationError(DataError):
    """Raised when data validation fails."""
    def __init__(self, message, field_name=None, field_value=None):
        self.field_name = field_name
        self.field_value = field_value
        if field_name is not None:
            message = f"{message} for field '{field_name}'"
            if field_value is not None:
                message = f"{message} with value '{field_value}'"
        super().__init__(message)

# Example usage
def validate_user_data(user_data):
    if not isinstance(user_data, dict):
        raise DataError("User data must be a dictionary")
    
    if "name" not in user_data:
        raise DataValidationError("Missing required field", "name")
    
    if "age" in user_data and not isinstance(user_data["age"], int):
        raise DataValidationError("Field must be an integer", "age", user_data["age"])
    
    return True

# Handling the exceptions
try:
    user = {"name": "Alice", "age": "thirty"}
    validate_user_data(user)
except DataValidationError as e:
    print(f"Validation error: {e}")
    if hasattr(e, "field_name"):
        print(f"Field name: {e.field_name}")
    if hasattr(e, "field_value"):
        print(f"Field value: {e.field_value}")
except DataError as e:
    print(f"Data error: {e}")
except AppError as e:
    print(f"Application error: {e}")
```

#### Adding Context and Metadata to Exceptions

Custom exceptions can carry additional information that helps with debugging.

```python
class DatabaseError(Exception):
    """Base exception for database errors."""
    def __init__(self, message, query=None, params=None, error_code=None):
        self.query = query
        self.params = params
        self.error_code = error_code
        
        # Build a detailed message
        details = []
        if error_code is not None:
            details.append(f"Error code: {error_code}")
        if query is not None:
            details.append(f"Query: {query}")
        if params is not None:
            details.append(f"Parameters: {params}")
        
        if details:
            detailed_message = f"{message} ({'; '.join(details)})"
        else:
            detailed_message = message
            
        super().__init__(detailed_message)

# Derived exceptions
class QueryError(DatabaseError):
    """Raised when a query is malformed."""
    pass

class ConnectionError(DatabaseError):
    """Raised when a database connection fails."""
    pass

# Example function that simulates database operations
def execute_query(query, params=None):
    if "SELECT" not in query.upper():
        raise QueryError("Invalid SELECT query", query, params, 1001)
    
    # Simulate a connection error
    if "users" in query and params and params.get("id") == 0:
        raise ConnectionError("Database connection lost", query, params, 2005)
    
    # Simulate successful query
    return [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

# Using the exceptions
try:
    # Try an invalid query
    data = execute_query("UPDATE users SET active = true", {"id": 1})
except QueryError as e:
    print(f"Query error: {e}")
    print(f"Original query: {e.query}")
    print(f"Error code: {e.error_code}")
except ConnectionError as e:
    print(f"Connection error: {e}")
except DatabaseError as e:
    print(f"General database error: {e}")
```

#### Exception Chaining and Contextualization

Exception chaining helps preserve the original exception while adding context.

```python
class ServiceError(Exception):
    """Base exception for service errors."""
    pass

class APIError(ServiceError):
    """Raised when an API request fails."""
    def __init__(self, message, endpoint=None, status_code=None):
        self.endpoint = endpoint
        self.status_code = status_code
        super().__init__(message)

def call_external_api(endpoint):
    try:
        # Simulate an HTTP request that fails
        if "users" in endpoint:
            raise ConnectionError("Connection refused")
        
        # Simulate a successful request
        return {"data": "some data"}
    except Exception as e:
        # Add context while preserving the original exception
        raise APIError(f"API call failed", endpoint=endpoint) from e

# Function that uses the API
def get_user_data(user_id):
    try:
        return call_external_api(f"/users/{user_id}")
    except APIError as e:
        # Add even more context
        raise ServiceError(f"Could not get data for user {user_id}") from e

# Using the function and handling exceptions
try:
    user_data = get_user_data(123)
except ServiceError as e:
    print(f"Service error: {e}")
    
    # Access the nested exception (APIError)
    if e.__cause__:
        print(f"Caused by: {e.__cause__}")
        
        # Access the original exception (ConnectionError)
        if e.__cause__.__cause__:
            print(f"Original error: {e.__cause__.__cause__}")
```

#### Enhancing Custom Exceptions with `__str__` and `__repr__`

Customizing string representation improves debugging and logging.

```python
class HTTPError(Exception):
    """Exception raised for HTTP errors."""
    
    def __init__(self, status_code, message=None, url=None, response=None):
        self.status_code = status_code
        self.message = message
        self.url = url
        self.response = response
        super().__init__(self.message)
    
    def __str__(self):
        """String representation for general use."""
        parts = [f"HTTP Error {self.status_code}"]
        if self.message:
            parts.append(self.message)
        if self.url:
            parts.append(f"URL: {self.url}")
        return " - ".join(parts)
    
    def __repr__(self):
        """Detailed representation for debugging."""
        cls_name = self.__class__.__name__
        attributes = []
        if self.status_code is not None:
            attributes.append(f"status_code={self.status_code}")
        if self.message is not None:
            attributes.append(f"message={repr(self.message)}")
        if self.url is not None:
            attributes.append(f"url={repr(self.url)}")
        if self.response is not None:
            attributes.append("response=<Response object>")
        
        return f"{cls_name}({', '.join(attributes)})"

# Using the enhanced exception
error = HTTPError(404, "Page not found", "https://example.com/missing")
print(str(error))   # More user-friendly
print(repr(error))  # More technical for debugging
```

#### Best Practices for Custom Exceptions

1. **Inherit from Appropriate Base Exceptions**:
   - Inherit from `Exception` for standard errors
   - Inherit from specialized exceptions like `ValueError` or `TypeError` when appropriate
   - Create your own base exception for your module/application

2. **Add Meaningful Information**:
   - Include all relevant context in the exception
   - Store additional data as attributes for programmatic access
   - Consider what information would be useful for debugging

3. **Document Your Exceptions**:
   - Add docstrings to exception classes
   - Explain when and why each exception is raised
   - Document the attributes and additional information

4. **Organize with a Hierarchy**:
   - Create a base exception for your module
   - Group related exceptions under intermediate base classes
   - Make the hierarchy reflect your application's domain

5. **Use Exception Chaining**:
   - Preserve original exceptions with `raise ... from`
   - Add context to help understand the error's origin
   - Don't lose valuable debugging information

### 6.3 Formatting Error Messages with Colors for Better Visibility

Colorized error messages can significantly improve readability and help distinguish different types of errors. This is especially useful in terminals and command-line applications.

#### Using ANSI Escape Sequences for Colors

ANSI escape sequences are a standard way to add colors to terminal output.

```python
# Basic ANSI color codes
class Colors:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    
    # Bright variants
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    
    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    
    # Styles
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

# Example usage
def print_colored(message, color):
    print(f"{color}{message}{Colors.RESET}")

print_colored("Success!", Colors.GREEN)
print_colored("Warning!", Colors.YELLOW)
print_colored("Error!", Colors.RED)
print_colored("Critical Error!", Colors.BOLD + Colors.BRIGHT_RED)
```

#### Creating a Colorized Error Formatter

```python
import sys
import traceback

class ErrorFormatter:
    """Format error messages with colors for better visibility."""
    
    # Color constants
    RESET = "\033[0m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    BRIGHT_RED = "\033[91m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    BOLD = "\033[1m"
    
    @staticmethod
    def format_exception(e, show_traceback=False):
        """Format an exception with colors."""
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Format the error message
        formatted = (
            f"{ErrorFormatter.BOLD}{ErrorFormatter.BRIGHT_RED}Error: "
            f"{error_type}{ErrorFormatter.RESET}{ErrorFormatter.RED} - "
            f"{error_msg}{ErrorFormatter.RESET}"
        )
        
        # Add traceback if requested
        if show_traceback:
            tb = traceback.format_exc()
            formatted += f"\n{ErrorFormatter.YELLOW}{tb}{ErrorFormatter.RESET}"
        
        return formatted
    
    @staticmethod
    def print_exception(e, show_traceback=False, file=sys.stderr):
        """Print a formatted exception."""
        print(ErrorFormatter.format_exception(e, show_traceback), file=file)
    
    @staticmethod
    def format_warning(message):
        """Format a warning message with colors."""
        return f"{ErrorFormatter.BOLD}{ErrorFormatter.YELLOW}Warning: {message}{ErrorFormatter.RESET}"
    
    @staticmethod
    def format_info(message):
        """Format an info message with colors."""
        return f"{ErrorFormatter.BLUE}Info: {message}{ErrorFormatter.RESET}"
    
    @staticmethod
    def format_success(message):
        """Format a success message with colors."""
        return f"{ErrorFormatter.BOLD}\033[32mSuccess: {message}{ErrorFormatter.RESET}"

# Example usage
try:
    result = 10 / 0
except Exception as e:
    ErrorFormatter.print_exception(e, show_traceback=True)

print(ErrorFormatter.format_warning("This operation is deprecated"))
print(ErrorFormatter.format_info("Processing file: data.csv"))
print(ErrorFormatter.format_success("All tests passed"))
```

#### Integrating with Custom Exceptions

```python
class ColoredError(Exception):
    """Base class for exceptions with colored output."""
    
    # Color constants
    RESET = "\033[0m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    BOLD = "\033[1m"
    
    # Default color for this exception
    COLOR = RED
    
    def __str__(self):
        """Return a colored string representation of the error."""
        return f"{self.COLOR}{super().__str__()}{self.RESET}"

class ValidationError(ColoredError):
    """Validation errors with customized color output."""
    
    # Override the color
    COLOR = YELLOW + BOLD
    
    def __init__(self, message, field=None):
        self.field = field
        if field:
            message = f"Invalid value for field '{field}': {message}"
        super().__init__(message)

class CriticalError(ColoredError):
    """Critical errors with bright red color."""
    
    # Override with a different color
    COLOR = "\033[91m" + BOLD

# Using the colored exceptions
try:
    raise ValidationError("Value must be a number", "age")
except Exception as e:
    print(f"Caught: {e}")

try:
    raise CriticalError("Database connection failed")
except Exception as e:
    print(f"Caught: {e}")
```

#### Using Third-Party Libraries for Advanced Coloring

Libraries like `colorama` provide cross-platform color support, while `rich` offers advanced formatting.

```python
# Using colorama (cross-platform)
# pip install colorama
from colorama import init, Fore, Back, Style

# Initialize colorama
init()

def print_colorama_error(message):
    print(f"{Fore.RED}{Style.BRIGHT}Error: {message}{Style.RESET_ALL}")

def print_colorama_warning(message):
    print(f"{Fore.YELLOW}Warning: {message}{Style.RESET_ALL}")

print_colorama_error("Failed to connect to the server")
print_colorama_warning("Configuration file not found, using defaults")

# Using rich for advanced formatting
# pip install rich
from rich import print as rprint
from rich.console import Console
from rich.traceback import install

# Install rich traceback handler
install()

console = Console()

def print_rich_error(message):
    console.print(f"[bold red]Error:[/] {message}")

def print_rich_traceback(e):
    console.print_exception()

try:
    x = 1 / 0
except Exception as e:
    print_rich_error("Division by zero")
    print_rich_traceback(e)
```

#### Creating a Logging Handler with Colors

Integrating color formatting with the standard logging module.

```python
import logging
import sys

# Custom logging handler for colored output
class ColoredConsoleHandler(logging.StreamHandler):
    """A logging handler that outputs colored messages to the console."""
    
    # Color mapping
    COLORS = {
        logging.DEBUG: "\033[94m",      # Light blue
        logging.INFO: "\033[92m",       # Light green
        logging.WARNING: "\033[93m",    # Light yellow
        logging.ERROR: "\033[91m",      # Light red
        logging.CRITICAL: "\033[97;41m" # White text on red background
    }
    
    RESET = "\033[0m"
    
    def emit(self, record):
        # Get the appropriate color for the log level
        color = self.COLORS.get(record.levelno, self.RESET)
        
        # Format the message
        formatter = self.formatter or logging.Formatter("%(levelname)s: %(message)s")
        formatted_message = formatter.format(record)
        
        # Add color and reset codes
        colored_message = f"{color}{formatted_message}{self.RESET}"
        
        # Write to the stream
        self.stream.write(colored_message + "\n")
        self.stream.flush()

# Configure logging with our custom handler
def setup_colored_logging(level=logging.INFO):
    """Set up logging with colored output."""
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add our custom handler
    handler = ColoredConsoleHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    
    return logger

# Example usage
logger = setup_colored_logging(logging.DEBUG)

logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")
```

#### Best Practices for Colored Error Messages

1. **Consider Terminal Support**:
   - Not all terminals support ANSI colors
   - Use libraries like `colorama` for cross-platform support
   - Provide a way to disable colors for non-compatible environments

2. **Use Colors Consistently**:
   - Choose a consistent color scheme for different error types
   - Use bright colors sparingly for high-priority messages
   - Consider accessibility (red/green color blindness is common)

3. **Don't Overuse Colors**:
   - Too many colors can be distracting
   - Use colors to highlight important information, not everywhere
   - Focus on readability over aesthetic appeal

4. **Provide Color Disabling Options**:
   - Add a way to disable colors (environment variable, command-line flag, etc.)
   - Detect non-interactive environments where colors might not be appropriate

```python
import os
import sys

# Detect if colors should be disabled
def should_use_colors():
    # Check for explicit environment variable
    if os.environ.get("NO_COLOR") is not None:
        return False
    
    # Check if the output is redirected to a file
    if not sys.stdout.isatty():
        return False
    
    # Check for terminals known to support colors
    term = os.environ.get("TERM", "")
    return term in ("xterm", "xterm-color", "xterm-256color", "linux", "screen", "screen-256color")

# Color functions that respect the settings
def red(text):
    return f"\033[31m{text}\033[0m" if should_use_colors() else text

def yellow(text):
    return f"\033[33m{text}\033[0m" if should_use_colors() else text

def green(text):
    return f"\033[32m{text}\033[0m" if should_use_colors() else text

# Example usage
print(red("This is red if the terminal supports it"))
print(yellow("This is yellow if the terminal supports it"))
print(green("This is green if the terminal supports it"))
```

## Module 7: Object-Oriented Programming (OOP) Deep Dive

### 7.1 When to Use `__str__` vs `__repr__` for Debugging

The `__str__` and `__repr__` methods control how objects are converted to strings in Python. Understanding their different purposes and when to use each is essential for effective debugging and logging.

#### The Core Differences

- **`__str__`**: Called by `str()` and `print()`. Should return a human-readable, concise representation.
- **`__repr__`**: Called by `repr()`. Should return an unambiguous, preferably eval-able representation.

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        """Human-readable string representation."""
        return f"Point at ({self.x}, {self.y})"
    
    def __repr__(self):
        """Unambiguous string representation for debugging."""
        return f"Point(x={self.x}, y={self.y})"

# Create a point and see the different string representations
p = Point(3, 4)
print(str(p))   # Calls __str__: "Point at (3, 4)"
print(repr(p))  # Calls __repr__: "Point(x=3, y=4)"

# Using print() calls __str__ implicitly
print(p)  # "Point at (3, 4)"

# In collections, __repr__ is used
points = [Point(1, 2), Point(3, 4)]
print(points)  # [Point(x=1, y=2), Point(x=3, y=4)]
```

#### The Eval Roundtrip Principle

A good `__repr__` implementation follows the "eval roundtrip" principle: `eval(repr(obj))` should create an equivalent object.

```python
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def __str__(self):
        return f"Rectangle with width={self.width} and height={self.height}"
    
    def __repr__(self):
        return f"Rectangle({self.width}, {self.height})"

# Create a rectangle
rect = Rectangle(10, 20)

# Get the string representations
str_representation = str(rect)
repr_representation = repr(rect)

print(str_representation)  # "Rectangle with width=10 and height=20"
print(repr_representation)  # "Rectangle(10, 20)"

# Try to recreate the object using eval
try:
    rect_copy = eval(repr_representation)
    print(f"Recreated: {rect_copy}")
    print(f"Equal to original: {rect_copy.width == rect.width and rect_copy.height == rect.height}")
except Exception as e:
    print(f"Cannot eval: {e}")
```

#### Default Implementations

If you don't define `__str__`, Python uses `__repr__`. If you don't define `__repr__`, Python uses the default implementation which shows the object's memory address.

```python
class DefaultStrRepr:
    def __init__(self, value):
        self.value = value

class ReprOnly:
    def __init__(self, value):
        self.value = value
    
    def __repr__(self):
        return f"ReprOnly({self.value!r})"

class StrOnly:
    def __init__(self, value):
        self.value = value
    
    def __str__(self):
        return f"StrOnly with value: {self.value}"

# Create instances
default_obj = DefaultStrRepr(42)
repr_obj = ReprOnly("hello")
str_obj = StrOnly("world")

# Default behavior
print(f"Default __str__: {str(default_obj)}")  # Uses default __repr__
print(f"Default __repr__: {repr(default_obj)}")  # Shows memory address

# ReprOnly behavior
print(f"ReprOnly __str__: {str(repr_obj)}")  # Uses __repr__
print(f"ReprOnly __repr__: {repr(repr_obj)}")  # Uses custom __repr__

# StrOnly behavior
print(f"StrOnly __str__: {str(str_obj)}")  # Uses custom __str__
print(f"StrOnly __repr__: {repr(str_obj)}")  # Shows memory address
```

#### When to Use Each Method

**Use `__str__` when:**

1. You need a user-friendly description
2. The object will be displayed to end-users
3. You want a concise, readable output format
4. The focus is on human understanding

**Use `__repr__` when:**

1. You need a detailed, unambiguous representation
2. The object will be used for debugging
3. You want to show all essential state
4. You want to enable object recreation (when possible)

```python
class User:
    def __init__(self, username, email, user_id):
        self.username = username
        self.email = email
        self.user_id = user_id
        self.is_active = True
        self.login_count = 0
    
    def __str__(self):
        """User-friendly string representation for end-users."""
        return f"User: {self.username}"
    
    def __repr__(self):
        """Detailed representation for debugging."""
        return (f"User(username={self.username!r}, email={self.email!r}, "
                f"user_id={self.user_id!r})")

# Create a user
user = User("alice", "alice@example.com", 42)

# For end-users, use __str__
print(f"Welcome, {user}")  # "Welcome, User: alice"

# For debugging, use __repr__
print(f"Debug info: {user!r}")  # "Debug info: User(username='alice', email='alice@example.com', user_id=42)"
```

#### Best Practices and Advanced Techniques

##### 1. Implementing Both Methods Efficiently

When both methods are needed, you can implement one in terms of the other to avoid duplication.

```python
class Product:
    def __init__(self, name, price, sku):
        self.name = name
        self.price = price
        self.sku = sku
    
    def __repr__(self):
        """Detailed representation for debugging."""
        return f"Product(name={self.name!r}, price={self.price!r}, sku={self.sku!r})"
    
    def __str__(self):
        """User-friendly representation based on __repr__."""
        return f"{self.name} (${self.price:.2f})"

# For simple classes, you might implement __repr__ based on __str__
class SimpleNote:
    def __init__(self, text):
        self.text = text
    
    def __str__(self):
        """Primary representation."""
        return f"Note: {self.text}"
    
    def __repr__(self):
        """Debug representation based on __str__."""
        return f"SimpleNote({self.text!r})"
```

##### 2. Nested Objects and Collection Formatting

When your object contains other objects or collections, format them with appropriate string representation.

```python
class ShoppingCart:
    def __init__(self, customer_name):
        self.customer_name = customer_name
        self.items = []
    
    def add_item(self, product, quantity=1):
        self.items.append((product, quantity))
    
    def __str__(self):
        """Human-readable representation."""
        if not self.items:
            return f"{self.customer_name}'s Shopping Cart (empty)"
        
        item_strs = [f"{qty}x {str(product)}" for product, qty in self.items]
        return f"{self.customer_name}'s Shopping Cart:\n" + "\n".join(item_strs)
    
    def __repr__(self):
        """Detailed debugging representation."""
        items_repr = ", ".join(
            f"({repr(product)}, {quantity})" for product, quantity in self.items
        )
        return f"ShoppingCart(customer_name={self.customer_name!r}, items=[{items_repr}])"

# Create a shopping cart
cart = ShoppingCart("Alice")
cart.add_item(Product("Laptop", 999.99, "TECH-001"), 1)
cart.add_item(Product("Mouse", 24.95, "TECH-002"), 2)

print(str(cart))
print(repr(cart))
```

##### 3. Using `!r` Format Specifier

The `!r` format specifier calls `repr()` on the formatted value, which is useful when implementing `__repr__`.

```python
class Person:
    def __init__(self, name, age, email=None):
        self.name = name
        self.age = age
        self.email = email
    
    def __repr__(self):
        # Use !r to ensure proper quoting of strings
        params = [f"name={self.name!r}", f"age={self.age!r}"]
        if self.email is not None:
            params.append(f"email={self.email!r}")
        
        return f"Person({', '.join(params)})"

# Create a person
p1 = Person("Bob", 30)
p2 = Person("Alice", 25, "alice@example.com")

print(repr(p1))  # Person(name='Bob', age=30)
print(repr(p2))  # Person(name='Alice', age=25, email='alice@example.com')
```

##### 4. Debugging with Custom Representations

```python
class DebugMixin:
    """A mixin that provides detailed debugging representation."""
    
    def _get_debug_attrs(self):
        """Get attributes to include in debug representation.
        Override in subclasses to customize.
        """
        return self.__dict__.items()
    
    def __repr__(self):
        """Detailed debugging representation with all attributes."""
        attrs = ", ".join(f"{k}={v!r}" for k, v in self._get_debug_attrs())
        return f"{self.__class__.__name__}({attrs})"

class DebuggableUser(DebugMixin):
    def __init__(self, username, email):
        self.username = username
        self.email = email
        self._password_hash = "secret_hash"  # Private attribute
    
    def __str__(self):
        """User-friendly representation."""
        return f"User: {self.username}"
    
    def _get_debug_attrs(self):
        """Override to exclude sensitive info from debug output."""
        return ((k, v) for k, v in self.__dict__.items() 
                if not k.startswith('_password'))

# Create a debuggable user
debug_user = DebuggableUser("charlie", "charlie@example.com")

print(str(debug_user))  # User: charlie
print(repr(debug_user))  # DebuggableUser(username='charlie', email='charlie@example.com')
```

### 7.2 Understanding `super()` and Method Resolution Order (MRO)

The `super()` function and Method Resolution Order (MRO) are fundamental to understanding how inheritance works in Python, especially with multiple inheritance.

#### Basic Usage of `super()`

The `super()` function returns a proxy object that delegates method calls to a parent or sibling class, following the MRO.

```python
class Parent:
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello, I'm {self.name}"

class Child(Parent):
    def __init__(self, name, age):
        super().__init__(name)  # Call Parent.__init__
        self.age = age
    
    def greet(self):
        parent_greeting = super().greet()  # Call Parent.greet
        return f"{parent_greeting} and I'm {self.age} years old"

# Create and use a Child instance
child = Child("Alice", 8)
print(child.greet())  # "Hello, I'm Alice and I'm 8 years old"
```

#### How `super()` Works Internally

When you call `super()`, Python:

1. Gets the current class (`Child` in the example above)
2. Gets the instance's MRO (method resolution order)
3. Finds the current class in the MRO
4. Returns a proxy object that delegates to the next class in the MRO

```python
class A:
    def method(self):
        print("A.method called")

class B(A):
    def method(self):
        print("B.method called")
        super().method()

class C(A):
    def method(self):
        print("C.method called")
        super().method()

class D(B, C):
    def method(self):
        print("D.method called")
        super().method()

# Create an instance of D and call method
d = D()
d.method()

# Output:
# D.method called
# B.method called
# C.method called
# A.method called

# Examining the MRO directly
print(D.__mro__)  # (<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>)
```

#### Understanding Method Resolution Order (MRO)

MRO determines the order in which Python searches for methods in a class hierarchy. Python uses the C3 linearization algorithm to create this order, which ensures:

1. A child class is searched before its parents
2. The order of parent classes in the class definition matters
3. The MRO preserves monotonicity (once a class appears, all its subclasses come before its parents)

```python
# Simple linear inheritance
class A:
    def method(self):
        print("A.method")

class B(A):
    def method(self):
        print("B.method")
        super().method()

class C(B):
    def method(self):
        print("C.method")
        super().method()

# Examine the MRO
print(C.__mro__)  # (<class '__main__.C'>, <class '__main__.B'>, <class '__main__.A'>, <class 'object'>)

# Call the method
c = C()
c.method()
# Output:
# C.method
# B.method
# A.method
```

#### The Diamond Problem and How Python Solves It

The "diamond problem" occurs in multiple inheritance when a class inherits from two classes that have a common ancestor. Python uses MRO to resolve method calls unambiguously.

```python
class Base:
    def greet(self):
        print("Base greeting")

class Left(Base):
    def greet(self):
        print("Left greeting")
        super().greet()

class Right(Base):
    def greet(self):
        print("Right greeting")
        super().greet()

class Bottom(Left, Right):
    def greet(self):
        print("Bottom greeting")
        super().greet()

# Check the MRO
print(Bottom.__mro__)
# Output: (<class '__main__.Bottom'>, <class '__main__.Left'>, <class '__main__.Right'>, <class '__main__.Base'>, <class 'object'>)

# Call the method
bottom = Bottom()
bottom.method()
# Output:
# Bottom greeting
# Left greeting
# Right greeting
# Base greeting
```

#### Using `super()` with Parameters

While `super()` is often used without arguments, you can explicitly specify the class and instance to use.

```python
class A:
    def method(self):
        print("A.method called")

class B(A):
    def method(self):
        print("B.method called")
        # Equivalent ways to call super().method()
        super().method()                 # Implicit form
        super(B, self).method()          # Explicit form
        super(__class__, self).method()  # Using __class__ (Python 3.4+)

b = B()
b.method()
# Output:
# B.method called
# A.method called
# A.method called
# A.method called
```

#### `super()` in Multiple Inheritance Scenarios

Using `super()` with multiple inheritance requires careful consideration of the MRO.

```python
class Titled:
    def __init__(self, title):
        self.title = title
        print(f"Titled.__init__({title})")

class Named:
    def __init__(self, name):
        self.name = name
        print(f"Named.__init__({name})")

# Incorrect way - skips some parent initializers
class Person1(Titled, Named):
    def __init__(self, title, name, age):
        Titled.__init__(self, title)  # Direct call bypasses MRO
        Named.__init__(self, name)    # Direct call bypasses MRO
        self.age = age
        print(f"Person1.__init__({title}, {name}, {age})")

# Correct way - respects MRO
class Person2(Titled, Named):
    def __init__(self, title, name, age):
        super().__init__(title)  # Calls next class in MRO (Titled.__init__)
        self.name = name
        self.age = age
        print(f"Person2.__init__({title}, {name}, {age})")

# Fixed way - all parent initializers called
class TitledNamed(Titled, Named):
    def __init__(self, title, name):
        Titled.__init__(self, title)
        Named.__init__(self, name)
        print(f"TitledNamed.__init__({title}, {name})")

class Person3(TitledNamed):
    def __init__(self, title, name, age):
        super().__init__(title, name)  # Calls TitledNamed.__init__
        self.age = age
        print(f"Person3.__init__({title}, {name}, {age})")

# Create instances
p1 = Person1("Dr.", "Alice", 30)
# Output:
# Titled.__init__(Dr.)
# Named.__init__(Alice)
# Person1.__init__(Dr., Alice, 30)

p2 = Person2("Prof.", "Bob", 40)
# Output:
# Titled.__init__(Prof.)
# Person2.__init__(Prof., Bob, 40)
# Note: Named.__init__ was skipped!

p3 = Person3("Mr.", "Charlie", 50)
# Output:
# Titled.__init__(Mr.)
# Named.__init__(Charlie)
# TitledNamed.__init__(Mr., Charlie)
# Person3.__init__(Mr., Charlie, 50)
```

#### Practical Uses of `super()`

##### 1. Extending Functionality in Subclasses

```python
class Shape:
    def __init__(self, color):
        self.color = color
    
    def area(self):
        """Calculate the area of the shape."""
        raise NotImplementedError("Subclasses must implement area()")
    
    def describe(self):
        """Return a description of the shape."""
        return f"A {self.color} shape"

class Circle(Shape):
    def __init__(self, color, radius):
        super().__init__(color)
        self.radius = radius
    
    def area(self):
        """Calculate the area of the circle."""
        import math
        return math.pi * self.radius ** 2
    
    def describe(self):
        """Extend the parent's describe method."""
        parent_description = super().describe()
        return f"{parent_description} with radius {self.radius}"

# Create and use a Circle
circle = Circle("red", 5)
print(circle.describe())  # "A red shape with radius 5"
print(f"Area: {circle.area():.2f}")  # "Area: 78.54"
```

##### 2. Cooperative Multiple Inheritance

Mixins become more powerful with `super()` when they cooperate to build functionality.

```python
class JSONSerializableMixin:
    def to_json(self):
        import json
        return json.dumps(self.to_dict())
    
    def to_dict(self):
        """Convert object to a dictionary."""
        return self.__dict__

class LoggableMixin:
    def log(self, message):
        print(f"[LOG] {self}: {message}")
    
    def __str__(self):
        """Default string representation for logging."""
        return f"{self.__class__.__name__}(id={id(self)})"

class Persistable:
    def save(self, filename):
        """Save object to a file."""
        data = self.to_dict()  # This will use JSONSerializableMixin's method
        with open(filename, 'w') as f:
            import json
            json.dump(data, f)
        self.log(f"Saved to {filename}")  # This will use LoggableMixin's method

class User(JSONSerializableMixin, LoggableMixin, Persistable):
    def __init__(self, username, email):
        self.username = username
        self.email = email
    
    def __str__(self):
        """Override the mixin's __str__ method."""
        return f"User(username={self.username})"

# Create and use a User
user = User("alice", "alice@example.com")
print(user.to_json())  # {"username": "alice", "email": "alice@example.com"}
user.log("User created")  # [LOG] User(username=alice): User created
user.save("user.json")  # [LOG] User(username=alice): Saved to user.json
```

##### 3. Adding Hooks for Subclasses

```python
class BaseValidator:
    def validate(self, value):
        """Validate a value using all validation methods."""
        self.pre_validate(value)  # Hook for subclasses
        
        if not self.is_valid_type(value):
            raise TypeError(f"Expected {self.expected_type.__name__}, got {type(value).__name__}")
        
        self.validate_value(value)  # Hook for subclasses
        self.post_validate(value)  # Hook for subclasses
        
        return True
    
    def pre_validate(self, value):
        """Hook for pre-validation steps."""
        pass
    
    def is_valid_type(self, value):
        """Check if the value has the expected type."""
        return isinstance(value, self.expected_type)
    
    def validate_value(self, value):
        """Hook for main validation logic."""
        pass
    
    def post_validate(self, value):
        """Hook for post-validation steps."""
        pass
    
    @property
    def expected_type(self):
        """The expected type for this validator."""
        return object

class IntegerValidator(BaseValidator):
    @property
    def expected_type(self):
        return int
    
    def validate_value(self, value):
        """Validate that the integer is in range."""
        if value < self.min_value or value > self.max_value:
            raise ValueError(f"Value must be between {self.min_value} and {self.max_value}")
    
    def __init__(self, min_value=float('-inf'), max_value=float('inf')):
        self.min_value = min_value
        self.max_value = max_value

class PositiveIntegerValidator(IntegerValidator):
    def __init__(self, max_value=float('inf')):
        super().__init__(min_value=1, max_value=max_value)
    
    def pre_validate(self, value):
        super().pre_validate(value)
        if hasattr(value, 'real') and value == 0:
            raise ValueError("Value must be positive (greater than zero)")

# Use the validators
int_validator = IntegerValidator(0, 100)
print(int_validator.validate(50))  # True
# print(int_validator.validate(200))  # ValueError: Value must be between 0 and 100

pos_validator = PositiveIntegerValidator(100)
print(pos_validator.validate(50))  # True
# print(pos_validator.validate(0))  # ValueError: Value must be positive (greater than zero)
# print(pos_validator.validate(-5))  # ValueError: Value must be between 1 and 100
```

#### Best Practices for Using `super()`

1. **Always Use `super()` for Parent Class Method Calls**:
   - Instead of explicitly calling `ParentClass.method(self, ...)`, use `super().method(...)`
   - This ensures the MRO is respected, especially with multiple inheritance

2. **Pass All Necessary Arguments**:
   - When using `super().__init__(...)`, make sure to pass all arguments needed by parent initializers
   - If a parent needs specific arguments, ensure they're provided

3. **Be Consistent with Method Signatures**:
   - Methods overridden in subclasses should accept the same parameters as their parent classes
   - This ensures that `super()` calls work correctly

4. **Understand the MRO of Your Classes**:
   - Use `Class.__mro__` to inspect the method resolution order
   - Design your inheritance hierarchy with the MRO in mind

5. **Use `super()` in Cooperative Multiple Inheritance**:
   - When a method in multiple parent classes should be called, use `super()` consistently
   - Ensure each class in the inheritance chain calls `super()` for the same methods

### 7.3 How to Use `dataclasses` to Simplify Class Definitions

The `dataclasses` module, introduced in Python 3.7, provides a decorator and functions for automatically adding special methods to classes, primarily designed to store data. Using dataclasses can significantly reduce boilerplate code.

#### Basic Usage of `dataclasses`

```python
from dataclasses import dataclass

# Regular class with a lot of boilerplate
class RegularPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"RegularPoint(x={self.x}, y={self.y})"
    
    def __eq__(self, other):
        if not isinstance(other, RegularPoint):
            return NotImplemented
        return self.x == other.x and self.y == other.y

# Equivalent dataclass with minimal code
@dataclass
class Point:
    x: float
    y: float

# Create instances
regular_point = RegularPoint(3.0, 4.0)
data_point = Point(3.0, 4.0)

# Both have similar behavior
print(regular_point)  # RegularPoint(x=3.0, y=4.0)
print(data_point)     # Point(x=3.0, y=4.0)

# Equality comparison works for both
print(regular_point == RegularPoint(3.0, 4.0))  # True
print(data_point == Point(3.0, 4.0))            # True
```

#### Generated Special Methods

The `@dataclass` decorator automatically generates several special methods:

- `__init__`: Constructor accepting field values
- `__repr__`: String representation
- `__eq__`: Equality comparison
- `__hash__`: (conditionally) Hash value
- And others like `__lt__`, `__le__`, etc. (with `order=True`)

```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    email: str

# The __init__ method accepts all fields
person = Person("Alice", 30, "alice@example.com")

# The __repr__ method shows all fields
print(person)  # Person(name='Alice', age=30, email='alice@example.com')

# The __eq__ method compares all fields
print(person == Person("Alice", 30, "alice@example.com"))  # True
print(person == Person("Bob", 30, "alice@example.com"))    # False
```

#### Customizing Field Behavior

The `field()` function allows customizing individual field behavior.

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Product:
    name: str
    price: float
    # Field with a default value
    quantity: int = 0
    
    # Field with a default factory
    tags: List[str] = field(default_factory=list)
    
    # Field excluded from __init__
    internal_id: Optional[str] = field(default=None, init=False)
    
    # Field excluded from __repr__
    secret_code: str = field(default="", repr=False)
    
    # Field excluded from comparison
    last_updated: str = field(default="", compare=False)
    
    def __post_init__(self):
        """Called after __init__ to perform additional initialization."""
        if not self.internal_id:
            self.internal_id = f"PROD-{hash(self.name) % 10000:04d}"

# Create a product
product = Product("Laptop", 999.99, 5, ["electronics", "computers"])
print(product)  # Product(name='Laptop', price=999.99, quantity=5, tags=['electronics', 'computers'])
print(product.internal_id)  # Something like "PROD-1234"

# Compare products (last_updated is ignored)
product.last_updated = "2023-01-01"
other_product = Product("Laptop", 999.99, 5, ["electronics", "computers"])
other_product.last_updated = "2023-02-01"
print(product == other_product)  # True
```

#### Inheritance with Dataclasses

Dataclasses can inherit from other dataclasses, and the fields are combined.

```python
from dataclasses import dataclass

@dataclass
class Position:
    x: float = 0.0
    y: float = 0.0

@dataclass
class Size:
    width: float = 0.0
    height: float = 0.0

@dataclass
class Rectangle(Position, Size):
    color: str = "black"
    
    def area(self):
        return self.width * self.height

# Create a rectangle
rect = Rectangle(10.0, 20.0, 30.0, 40.0, "red")
print(rect)  # Rectangle(x=10.0, y=20.0, width=30.0, height=40.0, color='red')
print(f"Area: {rect.area()}")  # Area: 1200.0

# Fields from base classes are included in __init__, __repr__, etc.
default_rect = Rectangle(color="blue")
print(default_rect)  # Rectangle(x=0.0, y=0.0, width=0.0, height=0.0, color='blue')
```

#### Advanced Dataclass Features

##### 1. Frozen Dataclasses (Immutable)

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Point:
    x: float
    y: float

# Create a point
point = Point(3.0, 4.0)

# Attempting to modify a frozen dataclass raises an exception
try:
    point.x = 5.0
except dataclasses.FrozenInstanceError as e:
    print(f"Error: {e}")  # Cannot assign to field 'x'

# Frozen dataclasses are hashable by default
points_set = {Point(1.0, 2.0), Point(3.0, 4.0)}
print(points_set)  # {Point(x=1.0, y=2.0), Point(x=3.0, y=4.0)}
```

##### 2. Post-Initialization Processing

The `__post_init__` method is called after `__init__` to perform additional initialization.

```python
from dataclasses import dataclass, field
import datetime

@dataclass
class Transaction:
    amount: float
    description: str
    date: datetime.date = field(default_factory=datetime.date.today)
    category: str = ""
    transaction_id: str = field(init=False)
    
    def __post_init__(self):
        """Generate a transaction ID and validate fields."""
        # Generate a unique ID
        self.transaction_id = f"TX-{hash(f'{self.date}-{self.amount}-{self.description}') % 10000:04d}"
        
        # Validate fields
        if self.amount == 0:
            raise ValueError("Transaction amount cannot be zero")
        
        # Set default category based on description
        if not self.category:
            if "coffee" in self.description.lower():
                self.category = "Food & Drink"
            elif "gas" in self.description.lower():
                self.category = "Transportation"
            else:
                self.category = "Miscellaneous"

# Create a transaction
tx = Transaction(25.99, "Coffee and sandwich")
print(tx)  # Transaction(amount=25.99, description='Coffee and sandwich', date=datetime.date(2023, 1, 1), category='Food & Drink', transaction_id='TX-1234')
```

##### 3. Controlling the Order of Fields

```python
from dataclasses import dataclass, field

@dataclass
class OrderedFields:
    # The order in __init__ and __repr__ follows the order of field definitions
    c: int
    a: int
    b: int

print(OrderedFields(1, 2, 3))  # OrderedFields(c=1, a=2, b=3)
```

##### 4. Using InitVar for Initialization-Only Parameters

```python
from dataclasses import dataclass, field, InitVar

@dataclass
class Circle:
    radius: float
    calculate_area: InitVar[bool] = False
    area: float = field(init=False, default=0.0)
    
    def __post_init__(self, calculate_area):
        if calculate_area:
            import math
            self.area = math.pi * self.radius ** 2

# Create circles
circle1 = Circle(5.0)
print(circle1)  # Circle(radius=5.0, area=0.0)

circle2 = Circle(5.0, True)
print(circle2)  # Circle(radius=5.0, area=78.53981633974483)

# Note that calculate_area is not stored as an attribute
print(hasattr(circle2, 'calculate_area'))  # False
```

##### 5. Generating `__hash__` for Using Instances as Dictionary Keys

```python
from dataclasses import dataclass

@dataclass(frozen=True, eq=True)
class Point:
    x: float
    y: float

# Create points
p1 = Point(1.0, 2.0)
p2 = Point(3.0, 4.0)
p3 = Point(1.0, 2.0)  # Same as p1

# Use as dictionary keys
points_data = {
    p1: "First point",
    p2: "Second point"
}

print(points_data[p1])  # "First point"
print(points_data[p3])  # "First point" (p3 is equal to p1)
```

##### 6. Converting to/from Dictionaries

```python
from dataclasses import dataclass, asdict, astuple

@dataclass
class User:
    name: str
    email: str
    age: int

# Create a user
user = User("Alice", "alice@example.com", 30)

# Convert to dictionary
user_dict = asdict(user)
print(user_dict)  # {'name': 'Alice', 'email': 'alice@example.com', 'age': 30}

# Convert to tuple
user_tuple = astuple(user)
print(user_tuple)  # ('Alice', 'alice@example.com', 30)

# Create from dictionary (using unpacking)
user_data = {'name': 'Bob', 'email': 'bob@example.com', 'age': 25}
new_user = User(**user_data)
print(new_user)  # User(name='Bob', email='bob@example.com', age=25)
```

#### Best Practices for Dataclasses

1. **Use Appropriate Type Hints**:
   - Be specific with type hints to improve code readability and enable static type checking
   - Use `Optional[type]` for fields that might be None

2. **Set Sensible Defaults**:
   - Use `default_factory` for mutable default values
   - Consider whether fields should be required or optional

3. **Make Immutable When Appropriate**:
   - Use `frozen=True` for immutable dataclasses
   - Immutable dataclasses can be used as dictionary keys or set elements

4. **Follow Single Responsibility Principle**:
   - Dataclasses are best for classes that primarily store data
   - For complex behavior, consider traditional classes or mixins

5. **Leverage Post-Initialization**:
   - Use `__post_init__` for field validation and derived fields
   - Be careful with modifying fields in frozen dataclasses (use object.__setattr__)

6. **Consider Performance**:
   - Dataclasses add some overhead compared to optimized manual classes
   - For performance-critical code with thousands of instances, benchmark both approaches

## Module 8: Advanced Data Structures & Algorithms in Python

### 8.1 Difference Between Tuple vs List vs Set vs Dict in Terms of Performance

Python offers several built-in data structures, each with different performance characteristics. Understanding these differences helps you choose the right structure for your specific use case.

#### Memory Usage Comparison

First, let's compare the memory footprint of each data structure:

```python
import sys

# Create sample data structures with the same elements
data = list(range(1000))
list_data = list(data)
tuple_data = tuple(data)
set_data = set(data)
dict_data = {i: i for i in data}

# Display memory usage
print(f"List size: {sys.getsizeof(list_data)} bytes")
print(f"Tuple size: {sys.getsizeof(tuple_data)} bytes")
print(f"Set size: {sys.getsizeof(set_data)} bytes")
print(f"Dict size: {sys.getsizeof(dict_data)} bytes")

# For small collections, compare overhead per element
small_data = list(range(5))
small_list = list(small_data)
small_tuple = tuple(small_data)
small_set = set(small_data)
small_dict = {i: i for i in small_data}

print(f"\nSmall list size: {sys.getsizeof(small_list)} bytes ({sys.getsizeof(small_list) / len(small_list):.1f} per item)")
print(f"Small tuple size: {sys.getsizeof(small_tuple)} bytes ({sys.getsizeof(small_tuple) / len(small_tuple):.1f} per item)")
print(f"Small set size: {sys.getsizeof(small_set)} bytes ({sys.getsizeof(small_set) / len(small_set):.1f} per item)")
print(f"Small dict size: {sys.getsizeof(small_dict)} bytes ({sys.getsizeof(small_dict) / len(small_dict):.1f} per item)")
```

#### Time Complexity Comparison

| Operation          | List         | Tuple        | Set          | Dict         |
|-------------------|--------------|--------------|--------------|--------------|
| Access by index    | O(1)         | O(1)         | N/A          | N/A          |
| Access by key      | N/A          | N/A          | N/A          | O(1)*        |
| Insertion          | O(1)** or O(n) | N/A (immutable) | O(1)*   | O(1)*        |
| Deletion           | O(1)** or O(n) | N/A (immutable) | O(1)*   | O(1)*        |
| Membership test    | O(n)         | O(n)         | O(1)*        | O(1)* (for keys) |
| Iteration          | O(n)         | O(n)         | O(n)         | O(n)         |

\* Average case, worst case is O(n) for hash collisions  
\** When operating at the end of the list

#### Benchmarking Different Operations

```python
import timeit
import random

# Create test data
list_data = list(range(10000))
tuple_data = tuple(list_data)
set_data = set(list_data)
dict_data = {i: i for i in list_data}

# Random elements to look up
lookup_elements = [random.randint(0, 9999) for _ in range(1000)]
nonexistent_elements = [random.randint(10000, 20000) for _ in range(1000)]

# Benchmark: Access by index/key
list_access = timeit.timeit(lambda: [list_data[i] for i in range(0, 10000, 100)], number=1000)
tuple_access = timeit.timeit(lambda: [tuple_data[i] for i in range(0, 10000, 100)], number=1000)
dict_access = timeit.timeit(lambda: [dict_data[i] for i in range(0, 10000, 100)], number=1000)

print(f"Access by index/key (1000 iterations):")
print(f"List: {list_access:.6f}s")
print(f"Tuple: {tuple_access:.6f}s")
print(f"Dict: {dict_access:.6f}s")

# Benchmark: Membership testing (element exists)
list_in = timeit.timeit(lambda: [x in list_data for x in lookup_elements], number=10)
tuple_in = timeit.timeit(lambda: [x in tuple_data for x in lookup_elements], number=10)
set_in = timeit.timeit(lambda: [x in set_data for x in lookup_elements], number=10)
dict_in = timeit.timeit(lambda: [x in dict_data for x in lookup_elements], number=10)

print(f"\nMembership testing (element exists, 10 iterations):")
print(f"List: {list_in:.6f}s")
print(f"Tuple: {tuple_in:.6f}s")
print(f"Set: {set_in:.6f}s")
print(f"Dict (keys): {dict_in:.6f}s")

# Benchmark: Membership testing (element doesn't exist)
list_not_in = timeit.timeit(lambda: [x in list_data for x in nonexistent_elements], number=10)
tuple_not_in = timeit.timeit(lambda: [x in tuple_data for x in nonexistent_elements], number=10)
set_not_in = timeit.timeit(lambda: [x in set_data for x in nonexistent_elements], number=10)
dict_not_in = timeit.timeit(lambda: [x in dict_data for x in nonexistent_elements], number=10)

print(f"\nMembership testing (element doesn't exist, 10 iterations):")
print(f"List: {list_not_in:.6f}s")
print(f"Tuple: {tuple_not_in:.6f}s")
print(f"Set: {set_not_in:.6f}s")
print(f"Dict (keys): {dict_not_in:.6f}s")

# Benchmark: Insertion
list_append = timeit.timeit(lambda: list_data.copy() + [10001], number=10000)
list_insert = timeit.timeit(lambda: list_data.copy().insert(0, 10001), number=10000)
set_add = timeit.timeit(lambda: set_data.copy().add(10001), number=10000)
dict_add = timeit.timeit(lambda: dict_data.copy().update({10001: 10001}), number=10000)

print(f"\nInsertion operations (10000 iterations):")
print(f"List append: {list_append:.6f}s")
print(f"List insert at beginning: {list_insert:.6f}s")
print(f"Set add: {set_add:.6f}s")
print(f"Dict add: {dict_add:.6f}s")

# Benchmark: Iteration
list_iteration = timeit.timeit(lambda: [x for x in list_data], number=1000)
tuple_iteration = timeit.timeit(lambda: [x for x in tuple_data], number=1000)
set_iteration = timeit.timeit(lambda: [x for x in set_data], number=1000)
dict_iteration = timeit.timeit(lambda: [x for x in dict_data], number=1000)

print(f"\nIteration (1000 iterations):")
print(f"List: {list_iteration:.6f}s")
print(f"Tuple: {tuple_iteration:.6f}s")
print(f"Set: {set_iteration:.6f}s")
print(f"Dict (keys): {dict_iteration:.6f}s")
```

#### Detailed Performance Characteristics

##### Lists

- **Strengths**:
  - Fast append/pop operations at the end (O(1))
  - Good performance for small collections
  - In-place modifications
  - Maintains order and allows duplicates

- **Weaknesses**:
  - Slow membership testing (O(n))
  - Slow insertions/deletions in the middle (O(n))
  - Memory overhead for each element

```python
# Comparing append vs. insert performance
import timeit

# Setup
setup = """
data = list(range(10000))
"""

append_time = timeit.timeit("data.append(10000)", setup=setup, number=100000)
insert_begin_time = timeit.timeit("data.insert(0, 10000)", setup=setup, number=100000)
insert_middle_time = timeit.timeit("data.insert(5000, 10000)", setup=setup, number=100000)

print(f"Append: {append_time:.6f}s")
print(f"Insert at beginning: {insert_begin_time:.6f}s")
print(f"Insert in middle: {insert_middle_time:.6f}s")
```

##### Tuples

- **Strengths**:
  - Slightly smaller memory footprint than lists
  - Faster iteration than lists in some cases
  - Immutable (can be used as dictionary keys or set elements)
  - Slightly faster creation than lists

- **Weaknesses**:
  - Immutable (can't be modified after creation)
  - Slow membership testing (O(n))

```python
# Comparing creation and access performance between lists and tuples
import timeit

creation_list = timeit.timeit("list(range(1000))", number=10000)
creation_tuple = timeit.timeit("tuple(range(1000))", number=10000)

# Setup
setup = """
list_data = list(range(1000))
tuple_data = tuple(range(1000))
import random
indices = [random.randint(0, 999) for _ in range(1000)]
"""

access_list = timeit.timeit("[list_data[i] for i in indices]", setup=setup, number=1000)
access_tuple = timeit.timeit("[tuple_data[i] for i in indices]", setup=setup, number=1000)

print(f"Creation time:")
print(f"List: {creation_list:.6f}s")
print(f"Tuple: {creation_tuple:.6f}s")

print(f"\nRandom access time:")
print(f"List: {access_list:.6f}s")
print(f"Tuple: {access_tuple:.6f}s")
```

##### Sets

- **Strengths**:
  - Very fast membership testing (O(1) average case)
  - Fast adding and removing elements (O(1) average case)
  - Automatically handles duplicates (uniqueness)
  - Set operations (union, intersection, difference)

- **Weaknesses**:
  - No indexing
  - Unordered (though Python 3.7+ preserves insertion order)
  - Higher memory overhead per element
  - Mutable (can't be used as dictionary keys)

```python
# Set operations performance
import timeit

# Setup
setup = """
set1 = set(range(10000))
set2 = set(range(5000, 15000))
"""

union = timeit.timeit("set1 | set2", setup=setup, number=10000)
intersection = timeit.timeit("set1 & set2", setup=setup, number=10000)
difference = timeit.timeit("set1 - set2", setup=setup, number=10000)
symmetric_difference = timeit.timeit("set1 ^ set2", setup=setup, number=10000)

print(f"Set operations (10000 iterations):")
print(f"Union: {union:.6f}s")
print(f"Intersection: {intersection:.6f}s")
print(f"Difference: {difference:.6f}s")
print(f"Symmetric difference: {symmetric_difference:.6f}s")

# Comparing set vs. list for uniqueness
setup = """
import random
data = [random.randint(0, 1000) for _ in range(10000)]
"""

list_unique = timeit.timeit("list(set(data))", setup=setup, number=1000)
manual_unique = timeit.timeit("""
unique = []
for item in data:
    if item not in unique:
        unique.append(item)
""", setup=setup, number=1000)

print(f"\nRemoving duplicates (1000 iterations):")
print(f"Using set: {list_unique:.6f}s")
print(f"Manual deduplication: {manual_unique:.6f}s")
```

##### Dictionaries

- **Strengths**:
  - Very fast key lookup (O(1) average case)
  - Fast adding and removing items (O(1) average case)
  - Flexible (any hashable object as key)
  - Key-value mapping
  - Preserves insertion order (Python 3.7+)

- **Weaknesses**:
  - Higher memory overhead
  - Keys must be hashable (immutable)
  - No duplicate keys

```python
# Dictionary operations performance
import timeit

# Setup
setup = """
dict_data = {i: i*2 for i in range(10000)}
keys_exist = [i for i in range(0, 10000, 10)]
keys_missing = [i for i in range(10000, 11000, 10)]
"""

dict_get = timeit.timeit("[dict_data.get(k) for k in keys_exist]", setup=setup, number=1000)
dict_get_default = timeit.timeit("[dict_data.get(k, -1) for k in keys_missing]", setup=setup, number=1000)
dict_update = timeit.timeit("dict_data.update({10000: 20000})", setup=setup, number=10000)
dict_del = timeit.timeit("""
d = dict_data.copy()
del d[5000]
""", setup=setup, number=10000)

print(f"Dictionary operations:")
print(f"Get (exists): {dict_get:.6f}s")
print(f"Get (missing, with default): {dict_get_default:.6f}s")
print(f"Update: {dict_update:.6f}s")
print(f"Delete: {dict_del:.6f}s")
```

#### Choosing the Right Data Structure

1. **Use Lists When**:
   - You need an ordered collection that may contain duplicates
   - You need to modify the collection frequently (append, extend)
   - You frequently access elements by index
   - The collection is small, or membership testing is rare

2. **Use Tuples When**:
   - You need an immutable sequence
   - You need to use the collection as a dictionary key or set element
   - You want a slightly more memory-efficient alternative to lists
   - The data represents a collection of related values (like coordinates)

3. **Use Sets When**:
   - You need to ensure elements are unique
   - You frequently test for membership
   - You need to perform set operations (union, intersection, etc.)
   - Order is not important

4. **Use Dictionaries When**:
   - You need key-value mapping
   - You need to look up values by a key 
   - You want to count occurrences (Counter is even better)
   - You need a flexible, fast lookup table

#### Real-World Performance Examples

```python
import timeit
import random
import string

# Generate test data
def generate_random_string(length=10):
    return ''.join(random.choices(string.ascii_letters, k=length))

random.seed(42)  # For reproducibility
small_data = [generate_random_string() for _ in range(100)]
medium_data = [generate_random_string() for _ in range(10000)]
lookup_elements = random.sample(medium_data, 1000)

# Example 1: Finding unique elements
list_unique = timeit.timeit(lambda: len(set(medium_data)), number=100)
manual_unique = timeit.timeit(lambda: len({x: None for x in medium_data}), number=100)

print(f"Finding unique elements in 10,000 strings (100 iterations):")
print(f"Using set: {list_unique:.6f}s")
print(f"Using dict: {manual_unique:.6f}s")

# Example 2: Frequency counting
using_list = timeit.timeit("""
counts = {}
for item in data:
    if item in counts:
        counts[item] += 1
    else:
        counts[item] = 1
""", setup="data = " + repr(small_data), number=10000)

using_dict = timeit.timeit("""
counts = {}
for item in data:
    counts[item] = counts.get(item, 0) + 1
""", setup="data = " + repr(small_data), number=10000)

from_collections = timeit.timeit("""
counts = Counter(data)
""", setup="from collections import Counter; data = " + repr(small_data), number=10000)

print(f"\nCounting frequencies in 100 strings (10,000 iterations):")
print(f"Using if-in-dict: {using_list:.6f}s")
print(f"Using dict.get: {using_dict:.6f}s")
print(f"Using Counter: {from_collections:.6f}s")

# Example 3: Membership testing with different data structures
list_lookup = timeit.timeit("""
count = 0
for item in lookup:
    if item in data:
        count += 1
""", setup=f"data = {repr(medium_data)}; lookup = {repr(lookup_elements[:100])}", number=10)

set_lookup = timeit.timeit("""
count = 0
for item in lookup:
    if item in data:
        count += 1
""", setup=f"data = set({repr(medium_data)}); lookup = {repr(lookup_elements[:100])}", number=10)

dict_lookup = timeit.timeit("""
count = 0
for item in lookup:
    if item in data:
        count += 1
""", setup=f"data = dict.fromkeys({repr(medium_data)}); lookup = {repr(lookup_elements[:100])}", number=10)

print(f"\nMembership testing for 100 items in 10,000 strings (10 iterations):")
print(f"Using list: {list_lookup:.6f}s")
print(f"Using set: {set_lookup:.6f}s")
print(f"Using dict keys: {dict_lookup:.6f}s")
```

### 8.2 How `collections` Module Enhances Python Data Structures

The `collections` module provides specialized container datatypes that extend and improve upon Python's built-in containers. These specialized containers offer enhanced functionality, better performance for specific use cases, and more readable code.

#### Overview of Key Data Structures in `collections`

```python
import collections

# Print available collections
print("Available collections:")
for name in dir(collections):
    if not name.startswith('_') and name[0].isupper():
        print(f"- {name}")
```

#### 1. `namedtuple`: Factory Function for Creating Tuple Subclasses

```python
from collections import namedtuple

# Define a named tuple class
Point = namedtuple('Point', ['x', 'y'])

# Create instances
p1 = Point(3, 4)
p2 = Point(x=5, y=6)  # Can use keyword arguments

# Access by index (like tuples)
print(f"p1[0] = {p1[0]}, p1[1] = {p1[1]}")

# Access by name (unlike regular tuples)
print(f"p1.x = {p1.x}, p1.y = {p1.y}")

# Unpacking works like regular tuples
x, y = p1
print(f"Unpacked: x = {x}, y = {y}")

# Named tuples are immutable
try:
    p1.x = 10  # Raises AttributeError
except AttributeError as e:
    print(f"Error: {e}")

# Additional features of namedtuple
print(f"As dict: {p1._asdict()}")
print(f"Fields: {p1._fields}")
p3 = p1._replace(x=10)  # Returns a new instance with updated value
print(f"Replace: {p3}")

# Creating from dictionary
data = {'x': 7, 'y': 8}
p4 = Point(**data)
print(f"From dict: {p4}")

# Adding a default value (Python 3.7+)
Point3D = namedtuple('Point3D', ['x', 'y', 'z'], defaults=[0])  # z defaults to 0
p5 = Point3D(1, 2)
print(f"With default: {p5}")
```

#### 2. `Counter`: Dict Subclass for Counting Hashable Objects

```python
from collections import Counter

# Create a counter
word = "mississippi"
char_count = Counter(word)
print(f"Character counts: {char_count}")

# Count elements in a list
colors = ['red', 'blue', 'red', 'green', 'blue', 'blue']
color_count = Counter(colors)
print(f"Color counts: {color_count}")

# Most common elements
print(f"Most common: {char_count.most_common(3)}")

# Updating counts
char_count.update("missouri")
print(f"After update: {char_count}")

# Mathematical operations
counter1 = Counter(a=3, b=1, c=5)
counter2 = Counter(a=1, b=2, c=3, d=4)

print(f"Addition: {counter1 + counter2}")
print(f"Subtraction: {counter1 - counter2}")  # Keeps only positive counts
print(f"Intersection (min): {counter1 & counter2}")
print(f"Union (max): {counter1 | counter2}")

# Zero and negative counts
counter1.update({'e': 0, 'f': -2})
print(f"With zero/negative: {counter1}")
print(f"Elements: {list(counter1.elements())}")  # Only positive counts
```

#### 3. `defaultdict`: Dict Subclass with Factory Function for Missing Keys

```python
from collections import defaultdict

# Regular dict behavior with missing keys
regular_dict = {}
try:
    regular_dict['key']  # Raises KeyError
except KeyError as e:
    print(f"Regular dict error: {e}")

# defaultdict with int factory (starts at 0)
int_dict = defaultdict(int)
int_dict['key'] += 1  # No KeyError, automatically initialized to 0
print(f"int_dict['key'] = {int_dict['key']}")

# defaultdict with list factory
list_dict = defaultdict(list)
list_dict['key'].append('value')  # No KeyError, automatically initialized to []
print(f"list_dict['key'] = {list_dict['key']}")

# defaultdict with set factory
set_dict = defaultdict(set)
set_dict['key'].add('value')  # No KeyError, automatically initialized to set()
print(f"set_dict['key'] = {set_dict['key']}")

# Custom factory function
def default_factory():
    return {'count': 0, 'values': []}

custom_dict = defaultdict(default_factory)
custom_dict['key']['count'] += 1
custom_dict['key']['values'].append('value')
print(f"custom_dict['key'] = {custom_dict['key']}")

# Common use case: grouping items
words = ['apple', 'banana', 'apple', 'orange', 'banana', 'apple']
by_length = defaultdict(list)

for word in words:
    by_length[len(word)].append(word)

print("Grouped by length:")
for length, words_list in by_length.items():
    print(f"{length} letters: {words_list}")
```

#### 4. `OrderedDict`: Dict Subclass that Remembers Insertion Order

```python
from collections import OrderedDict

# Regular dict (Python 3.7+) already maintains insertion order
regular_dict = {'c': 3, 'a': 1, 'b': 2}
print(f"Regular dict: {regular_dict}")

# OrderedDict
ordered_dict = OrderedDict([('c', 3), ('a', 1), ('b', 2)])
print(f"OrderedDict: {ordered_dict}")

# OrderedDict-specific methods
ordered_dict.move_to_end('c')
print(f"After move_to_end('c'): {ordered_dict}")

ordered_dict.move_to_end('c', last=False)  # Move to beginning
print(f"After move_to_end('c', last=False): {ordered_dict}")

# popitem removes and returns the last item by default
last_item = ordered_dict.popitem()
print(f"Popped last item: {last_item}")
print(f"After popitem(): {ordered_dict}")

# popitem(last=False) removes and returns the first item
first_item = ordered_dict.popitem(last=False)
print(f"Popped first item: {first_item}")
print(f"After popitem(last=False): {ordered_dict}")

# Equality comparison behavior
od1 = OrderedDict([('a', 1), ('b', 2)])
od2 = OrderedDict([('b', 2), ('a', 1)])
regular1 = {'a': 1, 'b': 2}
regular2 = {'b': 2, 'a': 1}

print(f"Regular dicts equal? {regular1 == regular2}")  # True
print(f"OrderedDicts equal? {od1 == od2}")            # False, order matters
```

#### 5. `deque`: List-like Container with Fast Appends and Pops on Both Ends

```python
from collections import deque
import timeit

# Create a deque
d = deque([1, 2, 3, 4, 5])
print(f"Deque: {d}")

# Add elements to both ends
d.append(6)       # Add to right end
d.appendleft(0)   # Add to left end
print(f"After appends: {d}")

# Remove elements from both ends
right_element = d.pop()      # Remove from right end
left_element = d.popleft()   # Remove from left end
print(f"Popped: left={left_element}, right={right_element}")
print(f"After pops: {d}")

# Rotate the deque
d.rotate(2)  # Rotate right by 2
print(f"After rotate(2): {d}")
d.rotate(-2)  # Rotate left by 2
print(f"After rotate(-2): {d}")

# Extend on both sides
d.extend([6, 7, 8])         # Extend on right
d.extendleft([0, -1, -2])   # Extend on left (note the order)
print(f"After extends: {d}")

# Bounded deque
bounded_deque = deque(maxlen=3)
for i in range(5):
    bounded_deque.append(i)
    print(f"Bounded deque: {bounded_deque}")

# Performance comparison: deque vs. list for queue operations
setup = """
from collections import deque
list_queue = list(range(10000))
deque_queue = deque(range(10000))
"""

list_queue_pop = timeit.timeit(
    "if list_queue: list_queue.pop(0)",
    setup=setup,
    number=10000
)

deque_queue_pop = timeit.timeit(
    "if deque_queue: deque_queue.popleft()",
    setup=setup,
    number=10000
)

print(f"\nQueue operation (10,000 pops):")
print(f"List pop(0): {list_queue_pop:.6f}s")
print(f"Deque popleft(): {deque_queue_pop:.6f}s")
```

#### 6. `ChainMap`: Dict-like Class for Combining Multiple Mappings

```python
from collections import ChainMap

# Create separate dictionaries
defaults = {'theme': 'default', 'language': 'en', 'showIndex': True}
user_settings = {'theme': 'dark'}

# Combine with ChainMap
combined = ChainMap(user_settings, defaults)
print(f"Combined settings: {dict(combined)}")
print(f"Theme: {combined['theme']}")         # From user_settings
print(f"Language: {combined['language']}")   # From defaults

# Adding a new mapping
cli_settings = {'language': 'fr'}
new_combined = ChainMap(cli_settings, user_settings, defaults)
print(f"New combined: {dict(new_combined)}")

# Modifying the first mapping
new_combined['theme'] = 'light'
print(f"After modification: {dict(new_combined)}")
print(f"Underlying mappings:")
for i, mapping in enumerate(new_combined.maps):
    print(f"Mapping {i}: {mapping}")

# Creating a child with new maps
child = new_combined.new_child({'debug': True})
print(f"Child: {dict(child)}")

# Using ChainMap for scoped variables
def variable_lookup():
    locals_dict = {'x': 'local'}
    globals_dict = {'x': 'global', 'y': 'global'}
    builtin_dict = {'y': 'builtin', 'z': 'builtin'}
    
    # Similar to Python's variable lookup
    scope = ChainMap(locals_dict, globals_dict, builtin_dict)
    
    print(f"x: {scope['x']}")  # 'local'
    print(f"y: {scope['y']}")  # 'global'
    print(f"z: {scope['z']}")  # 'builtin'

variable_lookup()
```

#### 7. `UserDict`, `UserList`, and `UserString`

These classes are base classes for creating custom container classes that have the same interface as the built-in containers.

```python
from collections import UserDict, UserList, UserString

# Custom dictionary that keeps track of access count
class TrackedDict(UserDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.access_count = {}
    
    def __getitem__(self, key):
        # Track access
        self.access_count[key] = self.access_count.get(key, 0) + 1
        return super().__getitem__(key)
    
    def most_accessed(self, n=1):
        """Return the n most accessed keys."""
        return sorted(self.access_count.items(), key=lambda x: x[1], reverse=True)[:n]

# Custom list that prevents duplicates
class UniqueList(UserList):
    def append(self, item):
        if item not in self.data:
            super().append(item)
    
    def extend(self, other):
        for item in other:
            self.append(item)

# Custom string that counts character frequency
class AnalyzedString(UserString):
    def character_frequency(self):
        from collections import Counter
        return Counter(self.data)
    
    def uppercase(self):
        return AnalyzedString(self.data.upper())

# Test the custom containers
tracked_dict = TrackedDict({'a': 1, 'b': 2, 'c': 3})
print(tracked_dict['a'])
print(tracked_dict['a'])
print(tracked_dict['b'])
print(f"Most accessed: {tracked_dict.most_accessed(2)}")

unique_list = UniqueList([1, 2, 3, 2, 1])
print(f"Unique list: {unique_list}")
unique_list.append(4)
unique_list.append(2)  # Won't be added
print(f"After appends: {unique_list}")

analyzed_str = AnalyzedString("Hello, World!")
print(f"Frequency: {analyzed_str.character_frequency()}")
print(f"Uppercase: {analyzed_str.uppercase()}")
```

#### Best Practices and Use Cases

1. **Use `namedtuple` when**:
   - You need a lightweight, immutable data class
   - You want a readable alternative to plain tuples
   - You don't need to update attributes after creation

2. **Use `Counter` when**:
   - You need to count occurrences of elements
   - You want to find the most common elements
   - You need to perform set-like operations on counts

3. **Use `defaultdict` when**:
   - You're tired of checking if keys exist before using them
   - You're grouping items or building nested data structures
   - You want to simplify code by eliminating key existence checks

4. **Use `OrderedDict` when**:
   - You need to maintain insertion order (Python < 3.7)
   - You need specific ordering operations like `move_to_end()`
   - Order matters for equality comparisons

5. **Use `deque` when**:
   - You need efficient appends/pops from both ends (queues, stacks)
   - You need a circular buffer with a maximum length
   - You need to rotate a sequence efficiently

6. **Use `ChainMap` when**:
   - You need to search through multiple dictionaries
   - You have a hierarchy of configurations or settings
   - You want to model nested scopes (like variable lookup)

7. **Use `UserDict`, `UserList`, or `UserString` when**:
   - You need a custom container with specialized behavior
   - You want to override built-in container methods
   - You need to add new functionality to container types

### 8.3 Efficient Searching and Sorting Techniques (`bisect`, `heapq`)

Python provides specialized modules for efficient searching and sorting operations that go beyond the basic sorting methods like `list.sort()` and the `sorted()` function. The `bisect` and `heapq` modules offer optimized algorithms for specific use cases.

#### The `bisect` Module for Binary Search

The `bisect` module provides support for maintaining a list in sorted order without having to sort the list after each insertion. This is especially useful for searching in sorted sequences.

```python
import bisect

# Create a sorted list
sorted_list = [1, 4, 6, 8, 12, 15, 20]

# Find insertion point for a value
pos = bisect.bisect(sorted_list, 10)
print(f"Insertion point for 10: {pos}")  # 5

# Insert a value at the right position
bisect.insort(sorted_list, 10)
print(f"After insertion: {sorted_list}")  # [1, 4, 6, 8, 10, 12, 15, 20]

# bisect_left vs bisect_right (bisect)
left_pos = bisect.bisect_left(sorted_list, 8)
right_pos = bisect.bisect_right(sorted_list, 8)
print(f"bisect_left(8): {left_pos}")   # 3 (first position where 8 can go)
print(f"bisect_right(8): {right_pos}") # 4 (first position after 8)

# Practical example: Grade determination
def grade(score, breakpoints=[60, 70, 80, 90], grades='FDCBA'):
    i = bisect.bisect(breakpoints, score)
    return grades[i]

print([grade(score) for score in [33, 60, 75, 89, 90, 100]])  # ['F', 'D', 'C', 'B', 'A', 'A']
```

#### Common Use Cases for `bisect`

```python
import bisect
import random

# 1. Finding the closest match in a sorted list
def find_closest(sorted_list, target):
    """Find the value in sorted_list that is closest to target."""
    pos = bisect.bisect_left(sorted_list, target)
    
    if pos == 0:
        return sorted_list[0]
    if pos == len(sorted_list):
        return sorted_list[-1]
    
    before = sorted_list[pos - 1]
    after = sorted_list[pos]
    
    if after - target < target - before:
        return after
    else:
        return before

data = sorted([random.randrange(100) for _ in range(10)])
print(f"Data: {data}")
print(f"Closest to 50: {find_closest(data, 50)}")

# 2. Range queries - counting elements within a range
def count_in_range(sorted_list, lower, upper):
    """Count elements in sorted_list between lower and upper."""
    lower_pos = bisect.bisect_left(sorted_list, lower)
    upper_pos = bisect.bisect_right(sorted_list, upper)
    return upper_pos - lower_pos

print(f"Elements between 30 and 70: {count_in_range(data, 30, 70)}")

# 3. Maintaining a sorted list with insort
def sorted_insert(sorted_list, values):
    for value in values:
        bisect.insort(sorted_list, value)
    return sorted_list

more_data = [random.randrange(100) for _ in range(5)]
print(f"More data: {more_data}")
print(f"After insertion: {sorted_insert(data.copy(), more_data)}")

# 4. Percentile calculations
def percentile(sorted_list, p):
    """Find the p-th percentile in sorted_list."""
    index = (len(sorted_list) - 1) * p
    lower = int(index)
    weight = index - lower
    
    if lower == len(sorted_list) - 1:
        return sorted_list[-1]
    
    return sorted_list[lower] * (1 - weight) + sorted_list[lower + 1] * weight

larger_data = sorted([random.randrange(100) for _ in range(100)])
print(f"Median (50th percentile): {percentile(larger_data, 0.5)}")
print(f"25th percentile: {percentile(larger_data, 0.25)}")
print(f"75th percentile: {percentile(larger_data, 0.75)}")
```

#### The `heapq` Module for Priority Queues

The `heapq` module provides an implementation of the heap queue algorithm, also known as the priority queue algorithm.

```python
import heapq

# Create a heap from a list (in-place)
data = [3, 1, 4, 1, 5, 9, 2, 6]
heapq.heapify(data)
print(f"Heap: {data}")  # Not fully sorted, but satisfies heap property

# Extract minimum element (heap-pop)
smallest = heapq.heappop(data)
print(f"Smallest: {smallest}")
print(f"Heap after pop: {data}")

# Add an element to the heap
heapq.heappush(data, 0)
print(f"Heap after push: {data}")

# Push an item on the heap, then pop the smallest item
value = heapq.heappushpop(data, 7)
print(f"Pushpop returned: {value}")
print(f"Heap after pushpop: {data}")

# Pop the smallest item, then push a new item
value = heapq.heapreplace(data, 8)
print(f"Replace returned: {value}")
print(f"Heap after replace: {data}")

# Get the n largest elements
largest = heapq.nlargest(3, data)
print(f"3 largest elements: {largest}")

# Get the n smallest elements
smallest = heapq.nsmallest(3, data)
print(f"3 smallest elements: {smallest}")
```

#### Common Use Cases for `heapq`

```python
import heapq
import time
from dataclasses import dataclass
import random

# 1. Priority Queue
@dataclass
class Task:
    priority: int
    description: str
    
    # For comparison in the heap
    def __lt__(self, other):
        return self.priority < other.priority

# Create some tasks
tasks = [
    Task(3, "Write documentation"),
    Task(1, "Fix critical bug"),
    Task(2, "Implement new feature"),
    Task(5, "Refactor old code"),
    Task(4, "Add tests")
]

# Convert to a heap
task_heap = tasks.copy()
heapq.heapify(task_heap)

# Process tasks in priority order
print("Processing tasks by priority:")
while task_heap:
    task = heapq.heappop(task_heap)
    print(f"Processing: {task.description} (priority: {task.priority})")

# 2. Top-K elements without sorting the entire list
numbers = list(range(10000))
random.shuffle(numbers)

start = time.time()
top_5_sorted = sorted(numbers, reverse=True)[:5]
time_sorted = time.time() - start

start = time.time()
top_5_heap = heapq.nlargest(5, numbers)
time_heap = time.time() - start

print(f"\nTop 5 elements: {top_5_heap}")
print(f"Time using sort: {time_sorted:.6f}s")
print(f"Time using heapq: {time_heap:.6f}s")

# 3. Merging sorted iterables
list1 = [1, 5, 9, 13]
list2 = [2, 6, 10]
list3 = [3, 7, 11]

# Merge the three sorted lists
merged = list(heapq.merge(list1, list2, list3))
print(f"\nMerged sorted lists: {merged}")

# 4. Running median using two heaps
def running_median(stream):
    """Calculate the running median for a stream of numbers."""
    min_heap = []  # for the larger half
    max_heap = []  # for the smaller half (stored with negative values)
    result = []
    
    for x in stream:
        # Add new element
        if len(min_heap) == len(max_heap):
            # Add to max_heap (smaller half)
            # But first check if it belongs in the larger half
            if min_heap and x > min_heap[0]:
                # Add smallest element from min_heap to max_heap
                heapq.heappush(max_heap, -heapq.heappop(min_heap))
                heapq.heappush(min_heap, x)
            else:
                heapq.heappush(max_heap, -x)
        else:
            # Add to min_heap (larger half)
            # But first check if it belongs in the smaller half
            if x < -max_heap[0]:
                # Add largest element from max_heap to min_heap
                heapq.heappush(min_heap, -heapq.heappop(max_heap))
                heapq.heappush(max_heap, -x)
            else:
                heapq.heappush(min_heap, x)
        
        # Calculate median
        if len(min_heap) == len(max_heap):
            median = (min_heap[0] - max_heap[0]) / 2
        else:
            median = -max_heap[0]
            
        result.append(median)
    
    return result

data_stream = [3, 1, 4, 1, 5, 9, 2, 6]
medians = running_median(data_stream)
print(f"\nRunning median: {medians}")
```

#### Combining `bisect` and `heapq` for Advanced Algorithms

```python
import bisect
import heapq
import random
import time

# Example: k-closest points to a given point
def k_closest(points, target, k):
    """Find k closest points to target using a max heap."""
    # Use a max heap of size k to track closest points
    heap = []
    
    for point in points:
        # Calculate distance (we use negative because heapq is a min-heap)
        dist = abs(point - target)
        
        if len(heap) < k:
            # Add to heap if not full
            heapq.heappush(heap, (-dist, point))
        elif -dist > heap[0][0]:
            # Replace farthest point if this point is closer
            heapq.heapreplace(heap, (-dist, point))
    
    # Extract the points (without distances)
    return [point for _, point in sorted(heap, reverse=True)]

points = [random.randint(0, 100) for _ in range(20)]
print(f"Points: {points}")
print(f"5 closest points to 50: {k_closest(points, 50, 5)}")

# Example: k-way sorted merge using heapq.merge
def merge_k_sorted_lists(lists):
    """Merge k sorted lists efficiently."""
    return list(heapq.merge(*lists))

sorted_lists = [
    sorted([random.randint(0, 100) for _ in range(10)]),
    sorted([random.randint(0, 100) for _ in range(15)]),
    sorted([random.randint(0, 100) for _ in range(12)])
]

merged = merge_k_sorted_lists(sorted_lists)
print(f"\nMerged {len(sorted_lists)} lists (total {len(merged)} elements)")
print(f"Is sorted? {merged == sorted(merged)}")

# Example: efficient range search in a large sorted array
def range_search(sorted_arr, lower, upper):
    """Find all elements between lower and upper in a sorted array."""
    left = bisect.bisect_left(sorted_arr, lower)
    right = bisect.bisect_right(sorted_arr, upper)
    return sorted_arr[left:right]

large_arr = sorted([random.randint(0, 1000) for _ in range(10000)])
range_result = range_search(large_arr, 250, 260)
print(f"\nElements between 250 and 260: {range_result}")

# Performance comparison with linear search
def linear_range_search(arr, lower, upper):
    """Linear search for elements in range."""
    return [x for x in arr if lower <= x <= upper]

start = time.time()
binary_result = range_search(large_arr, 400, 600)
binary_time = time.time() - start

start = time.time()
linear_result = linear_range_search(large_arr, 400, 600)
linear_time = time.time() - start

print(f"\nRange search performance comparison:")
print(f"Binary search time: {binary_time:.6f}s")
print(f"Linear search time: {linear_time:.6f}s")
print(f"Same results? {binary_result == linear_result}")
```

#### Using `sorted()` with Key Functions for Custom Sorting

While `bisect` and `heapq` provide specialized algorithms, the built-in `sorted()` function with custom key functions can be very powerful.

```python
import random
from operator import itemgetter, attrgetter
from dataclasses import dataclass

# Sample data
@dataclass
class Person:
    name: str
    age: int
    height: float
    
    def __repr__(self):
        return f"Person({self.name}, {self.age}, {self.height:.1f})"

people = [
    Person("Alice", 30, 165.5),
    Person("Bob", 25, 180.0),
    Person("Charlie", 35, 175.0),
    Person("Diana", 28, 162.5),
    Person("Eve", 35, 170.0),
]

# Sorting with lambda function
sorted_by_age = sorted(people, key=lambda p: p.age)
print(f"Sorted by age:\n{sorted_by_age}")

# Sorting with itemgetter (for dictionaries/tuples)
employees = [
    {"name": "Alice", "department": "HR", "salary": 75000},
    {"name": "Bob", "department": "Dev", "salary": 85000},
    {"name": "Charlie", "department": "Dev", "salary": 90000},
    {"name": "Diana", "department": "HR", "salary": 80000},
]

sorted_by_dept = sorted(employees, key=itemgetter("department", "salary"))
print(f"\nSorted by department and salary:\n{sorted_by_dept}")

# Sorting with attrgetter (for objects)
sorted_by_multiple = sorted(people, key=attrgetter("age", "height"))
print(f"\nSorted by age then height:\n{sorted_by_multiple}")

# Custom sort order
department_order = {"HR": 1, "Finance": 2, "Dev": 3, "Marketing": 4}
sorted_by_custom = sorted(
    employees,
    key=lambda e: (department_order.get(e["department"], 999), -e["salary"])
)
print(f"\nSorted by custom department order, then by descending salary:\n{sorted_by_custom}")

# Partial sorting (similar to nlargest/nsmallest but with custom key)
n = 3
bottom_n_salaries = sorted(employees, key=itemgetter("salary"))[:n]
top_n_salaries = sorted(employees, key=itemgetter("salary"), reverse=True)[:n]

print(f"\nTop {n} salaries:\n{top_n_salaries}")
print(f"Bottom {n} salaries:\n{bottom_n_salaries}")
```

#### Performance Considerations and Best Practices

1. **Use `bisect` when**:
   - You have a sorted list or need to maintain a sorted list
   - You need to find insertion points or closest matches
   - You need to perform range queries efficiently

2. **Use `heapq` when**:
   - You need a priority queue for processing items by priority
   - You want to efficiently find the n largest or smallest items
   - You need to merge multiple sorted iterables

3. **Use `sorted()` with key functions when**:
   - You need to sort by multiple criteria
   - You need complex custom sorting logic
   - You're sorting objects or dictionaries by attributes or keys

4. **Performance tips**:
   - For very large datasets, consider using specialized libraries like NumPy or Pandas
   - Binary search (`bisect`) is O(log n), much faster than linear search for large lists
   - Heaps (`heapq`) are more efficient for priority queues than keeping a list sorted
   - Extracting the n largest/smallest items with `nlargest`/`nsmallest` is more efficient than sorting the entire list when n is small

## Module 9: Testing, Debugging & Logging

### 9.1 Using `pdb` for Debugging and Stepping Through Code

The Python Debugger (`pdb`) is a powerful tool for debugging Python programs. It allows you to pause execution, inspect variables, and step through your code line by line to understand and fix issues.

#### Basic Usage of `pdb`

```python
import pdb

def complex_function(a, b):
    result = a * b
    # Set a breakpoint
    pdb.set_trace()
    for i in range(a):
        result += i
    return result

# Call the function
result = complex_function(3, 4)
print(f"Result: {result}")
```

When execution reaches `pdb.set_trace()`, it pauses and gives you a prompt where you can enter debugging commands.

#### Key `pdb` Commands

| Command | Description |
|---------|-------------|
| `h` or `help` | Show help |
| `n` or `next` | Execute the current line and move to the next line |
| `s` or `step` | Step into a function call |
| `r` or `return` | Continue execution until the current function returns |
| `c` or `continue` | Continue execution until the next breakpoint |
| `q` or `quit` | Quit the debugger |
| `p expression` | Print the value of an expression |
| `pp expression` | Pretty-print the value of an expression |
| `l` or `list` | Show the current line and context |
| `w` or `where` | Show the call stack |
| `b` or `break` | Set a breakpoint (line number or function) |
| `cl` or `clear` | Clear breakpoints |
| `u` or `up` | Move up one level in the call stack |
| `d` or `down` | Move down one level in the call stack |
| `a` or `args` | Print the arguments of the current function |

#### Using `pdb` as a Post-Mortem Debugger

```python
import pdb

def divide(a, b):
    return a / b

def calculate(x, y):
    result = 0
    try:
        result = divide(x, y)
    except Exception as e:
        print(f"An error occurred: {e}")
        # Start post-mortem debugging
        pdb.post_mortem()
    return result

# This will cause an error and trigger the debugger
result = calculate(10, 0)
```

#### Using `pdb` as a Module

You can also run a script under the debugger from the command line:

```
python -m pdb script.py
```

#### Command Line Options

When using `pdb` as a module, it starts before executing any code, and you can use these commands:

- `b` or `break`: Set a breakpoint at a line or function
- `c` or `continue`: Start execution until a breakpoint is hit
- `q` or `quit`: Quit the debugger

#### Using Breakpoints in Python 3.7+

Python 3.7 introduced a built-in `breakpoint()` function that works as an alias for `pdb.set_trace()`.

```python
def complex_function(a, b):
    result = a * b
    # Set a breakpoint using the built-in function
    breakpoint()
    for i in range(a):
        result += i
    return result

# Call the function
result = complex_function(3, 4)
print(f"Result: {result}")
```

The environment variable `PYTHONBREAKPOINT` can be used to control which debugger is used:
- `PYTHONBREAKPOINT=0` disables breakpoints
- `PYTHONBREAKPOINT=pdb.set_trace` uses the standard debugger (default)
- `PYTHONBREAKPOINT=module.function` uses a custom debugging function

#### Interactive Debugging Session Example

```python
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

def calculate_average(numbers):
    if not numbers:
        return 0
    total = calculate_sum(numbers)
    return total / len(numbers)

def process_data(data):
    filtered_data = [x for x in data if isinstance(x, (int, float))]
    breakpoint()  # Pause execution here
    average = calculate_average(filtered_data)
    return average

# Test with mixed data
result = process_data([1, 2, "3", 4.5, "text", 6])
print(f"Result: {result}")
```

Let's walk through the debugging session:

```
-> average = calculate_average(filtered_data)
(Pdb) p filtered_data
[1, 2, 4.5, 6]
(Pdb) p len(filtered_data)
4
(Pdb) p calculate_sum(filtered_data)
13.5
(Pdb) s
--Call--
> .../example.py(9)calculate_average()
-> def calculate_average(numbers):
(Pdb) n
> .../example.py(10)calculate_average()
-> if not numbers:
(Pdb) n
> .../example.py(12)calculate_average()
-> total = calculate_sum(numbers)
(Pdb) s
--Call--
> .../example.py(1)calculate_sum()
-> def calculate_sum(numbers):
(Pdb) n
> .../example.py(2)calculate_sum()
-> total = 0
(Pdb) n
> .../example.py(3)calculate_sum()
-> for num in numbers:
(Pdb) p numbers
[1, 2, 4.5, 6]
(Pdb) c
Result: 3.375
```

#### Advanced `pdb` Techniques

##### 1. Temporary Breakpoints

```python
def complex_function():
    # Long function with many steps
    step1()
    step2()
    step3()
    result = step4()  # We want to break only once, right before this line
    step5()
    return result

def debug_example():
    # Instead of permanently adding pdb.set_trace() to the code
    # Set a temporary breakpoint before the function call
    pdb.run("complex_function()")
```

##### 2. Conditional Breakpoints

```python
def process_items(items):
    for i, item in enumerate(items):
        # Only break when a specific condition is met
        if item < 0 and pdb.set_trace():
            pass
        result = item * 2
        print(f"Item {i}: {result}")

# Test with mixed data
process_items([1, 2, -3, 4, -5])
```

Alternatively, using the command line:

```
(Pdb) b 3, item < 0
```

##### 3. Using `.pdbrc` for Customization

You can create a `.pdbrc` file in your home directory or project directory with common commands:

```
# .pdbrc file
alias ll longlist
alias pl pp locals().items()
alias pg pp globals().items()

# Custom function
import pprint
alias pp pprint.pprint
```

##### 4. Debugging Remote Processes

For debugging server processes, you can use `rpdb` (remote pdb):

```python
import rpdb

def server_function():
    # Complex server logic
    rpdb.set_trace()  # This will listen on a network port
    # Continue processing
```

Connect to it using telnet:

```
telnet localhost 4444
```

##### 5. Using with Context Managers

```python
from contextlib import contextmanager

@contextmanager
def debug_section(section_name):
    print(f"Entering section: {section_name}")
    try:
        # Set a breakpoint at the beginning of the section
        pdb.set_trace()
        yield
    finally:
        print(f"Exiting section: {section_name}")

def complex_function():
    # Normal code
    with debug_section("Critical section"):
        # This section will start with a debugger
        critical_calculation()
    # More normal code
```

#### Best Practices for `pdb`

1. **Add Strategic Breakpoints**:
   - Place breakpoints at critical points in your code
   - Use conditional breakpoints for specific edge cases
   - Remember to remove or comment out breakpoints when done

2. **Inspect Variables Effectively**:
   - Use `p` for simple variables
   - Use `pp` for complex data structures
   - Check `locals()` and `globals()` for the current namespace

3. **Control Flow Carefully**:
   - Use `n` (next) for stepping over function calls
   - Use `s` (step) for stepping into functions
   - Use `r` (return) to execute until the current function returns
   - Use `c` (continue) to run to the next breakpoint

4. **Combine with Print Debugging**:
   - Use `print` statements for permanent logging
   - Use `pdb` for interactive investigation
   - Consider `logging` for production code

5. **Consider Alternative Debuggers**:
   - `ipdb`: Enhanced debugger with IPython features
   - IDE debuggers (PyCharm, VS Code): Graphical debugging
   - Web-based debuggers (for web applications)

### 9.2 Properly Structuring Logs for Debugging