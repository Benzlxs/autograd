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
## Computation graph construction
We cannot calucated the gradient of a multiply-referenced value until we have traversed all of the paths that lead up to it. Therefore, we need to do a topological sorting of the computation graph, and get an ordering of nodes with child nodes always appearing before their parents. When evaluting the node value, the forward computation traversal graph has been constructed with the opposite ordering constraint: parents before children. In this demo project, I can generate the backward computation graph by simply reversing the forward computation graph, and backward computation graph satisfies the constraint: children before parents.

```bash
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
```


## Gradient functions
After having backward computation graph, we can use the chain rule in differentiation to get the derivative of each variable to the final function value. The component in the chain rule is the graident of each arithmetic operator, and it should be defined maunally according to its differential equation. All gradient equations are defined in the file `autograd/operations_grad.py`. Here are some examples.

```bash
class multiplication_grad:
    # func: z = x * y
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vertices = [x, y]
    def backward(self, z_grad):
        self.x.grad += self.y.value * z_grad
        self.y.grad += self.x.value * z_grad
        
class log_grad:
    # func: z = log(y)
    def __init__(self, y): 
        self.y = y 
        self.vertices = [y] 
    def backward(self, z_grad):
        self.y.grad += z_grad / self.y.value

class sin_grad:
    # func: z = sin(y)
    def __init__(self, y): 
        self.y = y 
        self.vertices = [y] 
    def backward(self, z_grad):
        self.y.grad += z_grad * np.cos(self.y.value)
```

Currently, I just support addition, substraction, multiplication, division, log, exponentiation, cos, and sin, but other arithmetic operators can be appended easily to this demo project. 

## How to use
