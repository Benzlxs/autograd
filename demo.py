import autograd as gd
import pudb
pudb.set_trace()
def exp_mul(x):
    return gd.exp(3*x)

if __name__=="__main__":
    ## test exp and multiplication
    x = gd.node(2.)
    func = exp_mul(x)
    func.backward()
    print(x.grad)

    ## test power
    x = gd.node(2.)
    y = gd.node(3.)**x
    y.backward()
    print("power: ", x.grad)

