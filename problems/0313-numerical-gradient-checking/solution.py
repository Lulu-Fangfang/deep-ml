import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # Your code here
    # grad_minus = 1e-5
    n = len(x)
    grads = []
    for i in range(n):
        x_minus = np.asarray(x).copy()
        x_minus[i] = x[i] - epsilon
        x_maxus = np.asarray(x).copy()
        x_maxus[i] = x[i] + epsilon
        g = (f(x_maxus) - f(x_minus)) / (2 * epsilon)
        grads.append(float(g))

    error = (
        np.linalg.norm(grads - analytical_grad) / (np.linalg.norm(grads) + np.linalg.norm(analytical_grad))
    )
    if error == 0:
        flag = True
    else:
        flag = error
    return (grads, flag)
         