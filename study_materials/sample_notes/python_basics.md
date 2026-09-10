# Python Basics

## Variables and Types

Python is dynamically typed. Variables are created by assignment, no type
declaration needed.

```python
name = "Alice"       # str
age = 30             # int
height = 5.7         # float
is_student = True    # bool
scores = [95, 87, 92]  # list
```

## Functions

Functions are defined with `def`. They are first-class objects in Python.
you can pass them as arguments, return them from other functions, and assign
them to variables.

```python
def greet(name: str) -> str:
    return f"Hello, {name}!"

# Functions as values
say_hello = greet
print(say_hello("Bob"))  # Hello, Bob!
```

## Scope: The LEGB Rule

Python resolves variable names in this order:
1. **L**ocal, inside the current function
2. **E**nclosing, inside any wrapping function (important for closures)
3. **G**lobal, at the module level
4. **B**uilt-in, Python's built-in names (len, print, etc.)

```python
x = "global"

def outer():
    x = "enclosing"
    def inner():
        x = "local"
        print(x)   # "local"
    inner()
    print(x)       # "enclosing"

outer()
print(x)           # "global"
```
