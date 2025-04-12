"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Identity function."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> bool:
    """Less than operator."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Equal operator."""
    return x == y


def max(x: float, y: float) -> float:
    """Maximum operator."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two numbers are close."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """ReLU function."""
    return max(0.0, x)


def log(x: float) -> float:
    """Natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Exponential function."""
    return math.exp(x)


def log_back(x: float, grad: float) -> float:
    """Derivative of the natural logarithm."""
    return grad / x


def inv(x: float) -> float:
    """Inverse function."""
    return 1.0 / x


def inv_back(x: float, grad: float) -> float:
    """Derivative of the inverse function."""
    return -grad / (x * x)


def relu_back(x: float, grad: float) -> float:
    """Derivative of the ReLU function."""
    return grad if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(f: Callable[[float], float], xs: Iterable[float]) -> Iterable[float]:
    """Map a function over a list."""
    yield from (f(x) for x in xs)


def zipWith(
    f: Callable[[float, float], float], xs: Iterable[float], ys: Iterable[float]
) -> Iterable[float]:
    """Zip two lists together and apply a function."""
    yield from (f(x, y) for x, y in zip(xs, ys))


def reduce(
    f: Callable[[float, float], float], xs: Iterable[float], init: float
) -> float:
    """Reduce a list using a binary function."""
    result = init
    for x in xs:
        result = f(result, x)
    return result


def negList(xs: Iterable[float]) -> Iterable[float]:
    """Negate a list."""
    yield from map(neg, xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
    """Add two lists together."""
    yield from zipWith(add, xs, ys)


def sum(xs: Iterable[float]) -> float:
    """Sum a list."""
    return reduce(add, xs, 0.0)


def prod(xs: Iterable[float]) -> float:
    """Take the product of a list."""
    return reduce(mul, xs, 1.0)
