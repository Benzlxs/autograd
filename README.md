# Autograd
A toy code to demonstrate how autograd works! It includes all necessary components for auto differentitation, and these components are wrappted functions, gradient functions, and computation graph sortting. They will be introduced as follows.

## Wrapping functions
The autograd is to transform computation equation into gradient function, and we should have access to each operation in the computation equation. In this demo project, I chose to trace each function when it is being executed, and build computation graph on the fly. To do this, I need to wrap each original function with my function. The node with wrapped function takes care of the necessary book-keeping, recording the name of the function, the arguments, and the return value. To wrap the function, I define the basic element in autograd as node, (`autograd/nodes.py`).



## Gradient functions

## Computation graph construction

## How to use
