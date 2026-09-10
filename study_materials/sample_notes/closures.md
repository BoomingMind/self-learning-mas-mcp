# Python Closures

## What is a Closure?

A closure is a nested function that remembers variables from its enclosing
scope, even after that enclosing function has finished executing.

Three things must be true for a closure to exist:
1. There is a nested (inner) function
2. The inner function refers to a variable from the enclosing scope
3. The enclosing function returns the inner function

## The Classic Example

```python
def make_counter(start=0):
    count = start          # "count" is closed over by increment()

    def increment():
        nonlocal count     # tells Python: use the enclosing "count"
        count += 1
        return count

    return increment       # return the function, not the result

counter = make_counter(10)
print(counter())   # 11
print(counter())   # 12
print(counter())   # 13, "count" persists between calls
```

Why does `count` survive after `make_counter()` returns? Because `increment`
holds a reference to it. Python keeps `count` alive in a "cell object" as
long as `increment` exists.

## Why Use Closures?

Closures are a lightweight alternative to classes when you need state but
only one behavior.

```python
# With a class:
class Multiplier:
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, x):
        return x * self.factor

# With a closure, same behavior, less code:
def make_multiplier(factor):
    def multiply(x):
        return x * factor
    return multiply

double = make_multiplier(2)
triple = make_multiplier(3)
print(double(5))   # 10
print(triple(5))   # 15
```

## The Late Binding Gotcha

This is the most common closure mistake in Python:

```python
# BUG: what does this print?
funcs = [lambda: i for i in range(5)]
print(funcs[0]())   # Prints 4, not 0!
print(funcs[3]())   # Prints 4, not 3!
```

All five lambdas close over the same variable `i`. By the time you call
any of them, the loop has finished and `i` is 4.

```python
# FIX: capture the value at definition time
funcs = [lambda i=i: i for i in range(5)]
print(funcs[0]())   # 0, correct
print(funcs[3]())   # 3, correct
```

The `i=i` trick creates a new local variable `i` with the current value
of the loop variable, breaking the late binding.
