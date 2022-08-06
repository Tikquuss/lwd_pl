import torch

def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    """    
    else :
        a = grad_outputs.shape == y[0].shape
        assert a or grad_outputs.shape == y.shape
        if a :
             grad_outputs = grad_outputs.repeat(y.shape[0], 1, 1)
    """
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def jacobian(y, x):
    """Compute dy/dx = dy/dx @ grad_outputs; 
    for grad_outputs in [1, 0, ..., 0], [0, 1, 0, ..., 0], ...., [0, ..., 0, 1]"""
    m, n = y.shape[0], x.shape[0]
    jac = torch.zeros(m, n) 
    for i in range(m):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[i] = 1
        jac[i] = gradient(y, x, grad_outputs = grad_outputs)
    return jac