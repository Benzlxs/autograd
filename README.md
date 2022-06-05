# Autograd
A toy code to demonstrate how autograd works! It includes all necessary components for auto differentitation, and these components are wrappted functions, gradient functions, and computation graph sortting. They will be introduced as follows.

## Wrapping functions
The autograd is to transform computation equation into gradient function, and we should have access to each operation in the computation equation. In this demo project, I chose to trace each function when it is being executed, and build computation graph on the fly. To do this, I need to wrap each original function with my function. The node with wrapped function takes care of the necessary book-keeping, recording the name of the function, the arguments, and the return value. To wrap the function, I define the basic element in autograd as node, (`autograd/nodes.py`). In the naive python, `a=3`, a is actually a scalar class in python, in autograd we wrap the scalar and redefined a with `a=node(3)`, and functions that can be applied to node are also defined in the node class. Basic functions in the demo project include add, substruction, mulitiplication, divid, power, an exponent. Each time a wrapped function is called, it inspects its arguments and calls its underlying function with these arguments. Then, it returns a new node of calculate results, along with function and arguments. Therefore, each computation can be recorded and traced.
```bash

class node(object):
    def __init__(self, value: float, grad_ops=None):
        self.value = value
        self.grad = 0.0 
        self.grad_ops = grad_ops
    def __add__(self, other):
    def __radd__(self, other):
    def __mul__(self, other):
    def __rmul__(self, other):
    def __sub__(self, other):
    def __rpow__(self, other):
    def __truediv__(self, other):
```


## Gradient functions

## Computation graph construction

## How to use
