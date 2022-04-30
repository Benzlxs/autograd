import autograd as gd

def equation(x):
    return gd.exp(3*x + 3 - x**5)

def equation_2(x, y):
    return x**4 + x*y + 4*y - gd.cos(y)**2 - gd.sin(x)*gd.log(y)

if __name__=="__main__":
    # define node
    x = gd.node(2.)
    # define function
    f = equation(x)
    # calculate graident
    f.backward()
    print("Gradient of x:", x.grad)

    x = gd.node(2.)
    y = gd.node(3.)
    f = equation_2(x, y)
    f.backward()
    print("Graident of x:", x.grad)
    print("Graident of y:", y.grad)

