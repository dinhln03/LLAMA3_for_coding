import asyncio
import functools
import logging
from types import FunctionType, ModuleType
from typing import Type

from prometheus_client import Histogram, Counter

logger = logging.getLogger(__name__)

H = Histogram(f"management_layer_call_duration_seconds", "API call duration (s)",
              ["call"])


def _prometheus_module_metric_decorator(f: FunctionType):
    """
    A Prometheus decorator adding timing metrics to a function.
    This decorator will work on both asynchronous and synchronous functions.
    Note, however, that this function will turn synchronous functions into
    asynchronous ones when used as a decorator.
    :param f: The function for which to capture metrics
    """
    module_ = f.__module__.split(".")[-1]
    call_key = "{}_{}".format(module_, f.__name__)

    @functools.wraps(f)
    async def wrapper(*args, **kwargs):
        with H.labels(call=call_key).time():
            if asyncio.iscoroutinefunction(f):
                return await f(*args, **kwargs)
            else:
                return f(*args, **kwargs)

    return wrapper


def _prometheus_class_metric_decorator(f: FunctionType):
    """
    A Prometheus decorator adding timing metrics to a function in a class.
    This decorator will work on both asynchronous and synchronous functions.
    Note, however, that this function will turn synchronous functions into
    asynchronous ones when used as a decorator.
    :param f: The function for which to capture metrics
    """

    @functools.wraps(f)
    async def wrapper(*args, **kwargs):
        with H.labels(call=f.__name__).time():
            if asyncio.iscoroutinefunction(f):
                return await f(*args, **kwargs)
            else:
                return f(*args, **kwargs)

    return wrapper


def add_prometheus_metrics_for_module(module_: ModuleType):
    """
    Convenience function applying the Prometheus metrics decorator to the
    specified module's functions.
    :param module_: The module to which the instrumentation will be applied
    """
    decorate_all_in_module(module_, _prometheus_module_metric_decorator, [])


def add_prometheus_metrics_for_class(klass: Type):
    """
    Convenience function applying the Prometheus metrics decorator to the
    specified class functions.
    :param klass: The class to which the instrumentation will be applied
    """
    decorate_all_in_class(klass, _prometheus_class_metric_decorator, [])


def decorate_all_in_module(module_: ModuleType, decorator: FunctionType, whitelist: list):
    """
    Decorate all functions in a module with the specified decorator
    :param module_: The module to interrogate
    :param decorator: The decorator to apply
    :param whitelist: Functions not to be decorated.
    """
    for name in dir(module_):
        if name not in whitelist:
            obj = getattr(module_, name)
            if isinstance(obj, FunctionType) or asyncio.iscoroutinefunction(obj):
                # We only check functions that are defined in the module we
                # specified. Some of the functions in the module may have been
                # imported from other modules. These are ignored.
                if obj.__module__ == module_.__name__:
                    logger.debug(f"Adding metrics to {module_}:{name}")
                    setattr(module_, name, decorator(obj))
                else:
                    logger.debug(f"No metrics on {module_}:{name} because it belongs to another "
                                 f"module")
            else:
                logger.debug(f"No metrics on {module_}:{name} because it is not a coroutine or "
                             f"function")


def decorate_all_in_class(klass: Type, decorator: FunctionType, whitelist: list):
    """
    Decorate all functions in a class with the specified decorator
    :param klass: The class to interrogate
    :param decorator: The decorator to apply
    :param whitelist: Functions not to be decorated.
    """
    for name in dir(klass):
        if name not in whitelist:
            obj = getattr(klass, name)
            if isinstance(obj, FunctionType) or asyncio.iscoroutinefunction(obj):
                logger.debug(f"Adding metrics to {klass}:{name}")
                setattr(klass, name, decorator(obj))
            else:
                logger.debug(f"No metrics on {klass}:{name} because it is not a coroutine or "
                             f"function")
