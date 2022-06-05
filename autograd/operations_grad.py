import numpy as np

## gradient function of basic operations

class addition_grad:
    """
        func: z = x + y
        grad_z = dF/dz, 
        x.grad: dF/dx = dF/dz * dz/dx
        y.grad: dF/dy = dF/dz * dz/dy
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vertices = [x, y]
    def backward(self, z_grad):
        self.x.grad += z_grad
        self.y.grad += z_grad

class substraction_grad:
    # func: z = x - y
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vertices = [x, y]
    def backward(self, z_grad):
        self.x.grad += z_grad
        self.y.grad += -1 * z_grad

class multiplication_grad:
    # func: z = x * y
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vertices = [x, y]
    def backward(self, z_grad):
        self.x.grad += self.y.value * z_grad
        self.y.grad += self.x.value * z_grad

class division_grad:
    # func: z = x/y
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vertices = [x, y]
    def backward(self, z_grad):
        self.x.grad += z_grad / self.y.value
        self.y.grad += -1 * z_grad * self.x.value / (self.y.value * self.y.value)

class power_grad:
    # func: z = x**y
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vertices = [x, y]
    def backward(self, z_grad):
        self.x.grad += z_grad * self.y.value * self.x.value **\
                                        np.where(self.y.value, self.y.value - 1, 1.)
        self.y.grad += z_grad * np.log(self.y.value) * self.x.value **\
                                        self.y.value

class exp_grad:
    # func: z = e**y
    def __init__(self, y):
        self.y = y
        self.vertices = [y]
    def backward(self, z_grad):
        self.y.grad += z_grad * np.exp(self.y.value)

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

class cos_grad:
    # func: z = cos(y)
    def __init__(self, y):
        self.y = y
        self.vertices = [y]
    def backward(self, z_grad):
        self.y.grad += -1 * z_grad * np.sin(self.y.value)

