# Python Decorators

## What is a Decorator?

A decorator is a function that takes another function as input and returns
a new function with modified behavior, without changing the original
function's source code.

Decorators are built on closures. The `@` syntax is syntactic sugar.

```python
# These two are identical:

@my_decorator
def my_function():
    pass

# is exactly the same as:

def my_function():
    pass
my_function = my_decorator(my_function)
```

## Building a Decorator from Scratch

```python
import time

def timer(func):
    """Decorator that prints how long a function takes to run."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)          # call the original function
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(0.5)
    return "done"

slow_function()   # prints: slow_function took 0.5002s
```

`wrapper` is a closure: it closes over `func` from the enclosing `timer`
scope. Every decorated function gets its own `wrapper` with its own `func`.

## Preserving Function Metadata

A problem with the basic pattern above: `slow_function.__name__` now returns
`"wrapper"` instead of `"slow_function"`. Fix this with `functools.wraps`:

```python
import functools
import time

def timer(func):
    @functools.wraps(func)        # copies __name__, __doc__, etc. to wrapper
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper
```

Always use `@functools.wraps` in production decorators. Debugging is much
harder when all your functions report `__name__ == "wrapper"`.

## Decorators with Arguments

Sometimes you want a decorator that accepts its own arguments:

```python
def retry(max_attempts=3, delay=1.0):
    """Retry a function up to max_attempts times on failure."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def unstable_api_call():
    # might fail sometimes
    pass
```

Notice there are now three layers: `retry` returns `decorator`, which
returns `wrapper`. Each layer is a closure over the layer above it.
