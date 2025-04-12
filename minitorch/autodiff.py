from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Callable
from collections import deque, defaultdict

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(
    f: Callable[[Any], float], *vals: Any, arg: int = 0, epsilon: float = 1e-6
) -> float:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    assert arg < len(vals), "arg must be less than the number of vals"

    modified_vals = list(vals)
    modified_vals[arg] = modified_vals[arg] + epsilon

    return (f(*modified_vals) - f(*vals)) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    in_degree = defaultdict(int)
    search_interface = [variable]
    while search_interface:
        new_search_interface = []
        for var in search_interface:
            if var.is_constant() or var.is_leaf():
                continue
            for parent in var.parents:
                if parent.unique_id not in in_degree:
                    new_search_interface.append(parent)
                in_degree[parent.unique_id] += 1
        search_interface = new_search_interface

    result = []
    queue = deque([variable])
    while queue:
        current = queue.popleft()
        if current.is_constant():
            continue

        result.append(current)
        for parent in current.parents:
            in_degree[parent.unique_id] -= 1
            if in_degree[parent.unique_id] == 0:
                queue.append(parent)

    yield from result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    grad_dict = {variable.unique_id: deriv}

    for var in topological_sort(variable):
        if var.is_leaf():
            var.accumulate_derivative(grad_dict[var.unique_id])
            continue

        for parent, parent_deriv in var.chain_rule(grad_dict[var.unique_id]):
            if parent.unique_id not in grad_dict:
                grad_dict[parent.unique_id] = parent_deriv.zeros()
            grad_dict[parent.unique_id] += parent_deriv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
