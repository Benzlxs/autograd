import numpy as np
from .operations_grad import *
from .utilis import *
"""
autograd from basic element, Node (scalar)
    value: input
    grad: gradient
    vjp: vector jacobian product
"""
class Node(object):
    def __init__(self, value: float, grad_ops=None):
        self.value = value
        self.grad = 0.0
        self.grad_ops = grad_ops

    def __str__(self):
       return f"scalar:\nvalue: {self.value}\ngrad: {self.grad}\ngrad_ops: {self.grad_ops}"
    
    def __repr__(self):
        return str(self.value)

    def __add__(self, other):
        if not isinstance(other, Node):
            other = Node(other)
        return Node(
            self.value + other.value,
            grad_ops=add_grad(self, other)
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if not isinstance(other, Node):
            other = Node(other)
        return Node(
            self.value * other.value,
            grad_ops=multiplication_grad(self, other)
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if not isinstance(other, Node):
            other = Node(other)
        return Node(
            self.value - other.value,
            grad_ops=substraction_grad(self, other)
        )

    def __truediv__(self, other):
        if not isinstance(other, Node):
            other = Node(other)
        return Node(
            self.value / other.value,
            grad_ops=division_grad(self, other)
        )

    def __rtruediv__(self, other):
        if not isinstance(other, Node):
            other = Node(other)
        return Node(
            other.value / self.value,
            grad_ops=division_grad(other, self)
        )

    def backward(self):
        vertices = []
        edges = {}
        generate_graph(self, vertices, edges)

        # Sort The Graph
        sorted_vertices = topological_sort(vertices, edges)

        # Run backward on the graph
        sorted_vertices[0].grad = 1.0
        for vertex in sorted_vertices:
            if vertex.grad_ops is not None:
                vertex.grad_ops.backward(vertex.grad)

def exp(x: Node):
    return Node(
        np.exp(x.value),
        grad_ops=exp_grad(x)
    )

def log(x: Node):
    return Node(
        np.log(x.value),
        grad_ops=log_grad(x)
    )

def sin(x: Node):
    return Node(
        np.sin(x.value),
        grad_ops=sin_grad(x)
    )

def cos(x: Node):
    return Node(
        np.cos(x.value),
        grad_ops=cos_grad(x)
    )

