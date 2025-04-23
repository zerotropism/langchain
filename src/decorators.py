from functools import wraps
import logging
import inspect

# logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s : ",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def handle_exception(func):
    """Decorator to handle class methods exceptions"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get the caller's frame to log the correct module and line info
            frame = inspect.currentframe().f_back
            module_name = frame.f_globals["__name__"]
            line_no = frame.f_lineno

            # Log the error with the caller's context
            logger.error(
                f"Error in {func.__name__} (module: {module_name}, line: {line_no}): {str(e)}"
            )
            return None

    return wrapper
