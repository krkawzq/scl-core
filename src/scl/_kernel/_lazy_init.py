"""
Lazy Initialization Helper for Kernel Modules

Provides decorators and utilities for delayed library loading.
"""

from functools import wraps


def lazy_init_decorator(init_func_name='_init_signatures'):
    """
    Decorator factory for lazy initialization.
    
    Usage:
        _signatures_initialized = False
        
        def _init_signatures():
            # ... setup code ...
            pass
        
        @lazy_init(init_func_name='_init_signatures')
        def my_kernel_function(...):
            # Will auto-call _init_signatures() on first use
            pass
    
    Args:
        init_func_name: Name of the initialization function in the module
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the module where func is defined
            module = func.__globals__
            
            # Check if initialized
            state_var = f'_{init_func_name.lstrip("_")}_initialized'
            if not module.get(state_var, False):
                # Call initialization function
                init_func = module.get(init_func_name)
                if init_func:
                    try:
                        init_func()
                        module[state_var] = True
                    except Exception as e:
                        # Re-raise with context
                        raise RuntimeError(
                            f"Failed to initialize {func.__module__}: {e}"
                        ) from e
            
            # Call the actual function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Convenient alias
lazy_kernel = lazy_init_decorator('_init_signatures')

