"""
Utility decorators for the LangChain project

This module provides reusable decorators for:
- Exception handling
- Logging
- Performance measurement
- Etc.
"""

import logging
import functools
import traceback
import time
import inspect
import sys
from typing import Any, Callable, TypeVar, cast

# Logger configuration
logging.basicConfig(
    level=logging.ERROR,
    # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    format="%(asctime)s - %(levelname)s - %(name)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s : ",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("../logs/errors.log"),
    ],
)

logger = logging.getLogger("langchain")

# TypeVar definition for decorator typing
F = TypeVar("F", bound=Callable[..., Any])


def handle_exception(func: F) -> F:
    """
    Decorator to capture and log exceptions in a comprehensive way.

    This decorator:
    - Intercepts all unhandled exceptions
    - Logs detailed information about the exception and execution context
    - Propagates the exception or returns None based on configuration

    Args:
        func: The function to decorate

    Returns:
        The decorated function with exception handling
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Gathering contextual information
            module_name = func.__module__
            function_name = func.__name__
            exception_type = type(e).__name__
            exception_msg = str(e)
            stack_trace = traceback.format_exc()

            # Gathering call argument details
            # For class methods, exclude 'self' from logs
            call_args = args[1:] if args and hasattr(args[0], func.__name__) else args

            # Filter sensitive information in kwargs
            filtered_kwargs = {
                k: (v[:50] + "..." if isinstance(v, str) and len(v) > 50 else v)
                for k, v in kwargs.items()
            }

            # Building log message
            log_message = f"""
======== Exception Detected ========
Module:     {module_name}
Function:   {function_name}
Type:       {exception_type}
Message:    {exception_msg}
Arguments:  {call_args}
Parameters: {filtered_kwargs}
Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}
Stack trace:
{stack_trace}
===================================
"""

            # Logging
            logger.error(log_message)

            # Inform the user that an error occurred
            print(
                f"[ERROR] An error occurred while executing {function_name}: {exception_msg}"
            )
            print("Check the log file for more details.")

            # We can choose to propagate the exception or return None
            # Here we propagate to allow custom handling at a higher level
            raise

    return cast(F, wrapper)


# Examples of other useful decorators that could complement this module
def timing_decorator(func: F) -> F:
    """Measures and logs the execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(
            f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds"
        )
        return result

    return cast(F, wrapper)


def retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator that retries the function execution if it fails.

    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            last_exception = None

            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    last_exception = e

                    if attempts < max_attempts:
                        logger.warning(
                            f"Attempt {attempts} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {delay} seconds..."
                        )
                        time.sleep(delay)

            logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            raise last_exception

        return cast(F, wrapper)

    return decorator
