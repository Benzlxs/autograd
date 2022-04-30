import numpy as np

## gradient function of basic operations

class addition_grad:
    def __init__(self, parent_node, child_node):
        self.parent_node = parent_node
        self.child_node = child_node
        self.vertices = [parent_node, child_node]
    def backward(self, parent_grad):
        self.parent_node.grad += parent_grad
        self.child_node.grad += parent_grad

class substraction_grad:
    def __init__(self, parent_node, child_node):
        self.parent_node = parent_node
        self.child_node = child_node
        self.vertices
    def backward(self, parent_grad):
        self.parent_node.grad += parent_grad
        self.child_node.grad += -1 * parent_grad

class multiplication_grad:
    def __init__(self, parent_node, child_node):
        self.parent_node = parent_node
        self.child_node = child_node
        self.vertices = [parent_node, child_node]
    def backward(self, parent_grad):
        self.parent_node.grad += self.child_node.value * parent_grad
        self.child_node.grad += self.parent_node.value * parent_grad

class division_grad:
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator
        self.vertices = [numerator, denominator]
    def backward(self, parent_grad):
        self.numerator.grad += parent_grad / self.denominator.value
        self.denominator.grad += -1 * parent_grad * self.numerator / (self.denominator * self.denominator)

class exp_grad:
    def __init__(self, child_node):
        self.child_node = child_node
        self.vertices = [child_node]
    def backward(self, parent_grad):
        self.child_node.grad += parent_grad * np.exp(self.child_node.value)

class log_grad:
    def __init__(self, child_node):
        self.child_node = child_node
        self.vertices = [child_node]
    def backward(self, parent_grad):
        self.child_node.grad += parent_grad / self.child_node.value

class sin_grad:
    def __init__(self, child_node): 
        self.child_node = child_node
        self.vertices = [child_node]
    def backward(self, parent_grad):
        self.child_node.grad += parent_grad * np.cos(self.child_node.value)

class cos_grad:
    def __init__(self, child_node):
        self.child_node = child_node
        self.vertices = [child_node]
    def backward(self, parent_grad):
        self.child_node.grad += -1 * parent_grad * np.sin(self.child_node.value)

