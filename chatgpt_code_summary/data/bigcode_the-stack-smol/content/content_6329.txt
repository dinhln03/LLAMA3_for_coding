from functools import wraps


def multiply_by(multiplier):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            print(f'Calling "{func_name}({args[0]}, {args[1]})" function')
            print(f'"{func_name}" function is multiplied by {multiplier}')
            result = func(*args, **kwargs) * multiplier
            print(f'Result equals to {result}')
            return result
        return wrapper
    return decorator


@multiply_by(multiplier=3)
def add(a, b):
    return a + b


add(2, 3)
