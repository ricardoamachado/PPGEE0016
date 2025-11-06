import numpy as np
def steepest_descent(function, gradient, x_init, alpha = 0.2, tol = 1e-6, num_iter = 1000):
    x_k = x_init
    for k in range(num_iter):
        grad_k = gradient(x_k)
        x_new = x_k - alpha * grad_k
        if np.linalg.norm(alpha * grad_k) < tol:
            break
        x_k = x_new
    return x_new, function(x_new)


def function(x):
    for i in range(len(x)):
        return np.sum(x**2)

def grad_function(x):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        grad[i] = 2 * x[i]
    return grad

if __name__ == "__main__":
    init_point = np.array([1.0, 1.0, 1.0])
    optimal_point, optimal_value = steepest_descent(function, grad_function, init_point)
    print("Optimal Point:", optimal_point)
    print("Optimal Value:", optimal_value)