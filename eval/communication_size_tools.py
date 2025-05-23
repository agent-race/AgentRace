import sys
from functools import wraps
def communication_size_input(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs) 
        print(f"Communication Size: {sys.getsizeof(kwargs['ev'].model_dump_json())}") 
        return result
    return wrapper


def communication_size_async_input(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs) 
        print(f"Communication Size: {sys.getsizeof(kwargs['ev'].model_dump_json())}") 
        return result
    return wrapper


def communication_size_output(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs) 
        print(f"Communication Size: {sys.getsizeof(result)}") 
        return result
    return wrapper


def communication_size_async_output(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs) 
        print(f"Communication Size: {sys.getsizeof(result)}") 
        return result
    return wrapper
