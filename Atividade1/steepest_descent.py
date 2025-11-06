import numpy as np
def steepest_descent(function, gradient, x_init, alpha = 0.01, tol = 1e-6, num_iter = 1000):
    x_k = x_init
    for k in range(num_iter):
        grad_k = gradient(x_k)
        x_new = x_k - alpha * grad_k
        if np.linalg.norm(alpha * grad_k) < tol:
            break
        x_k = x_new
    return x_new, function(x_new)


def function(x):
    d = x[0]
    fs = x[1]

    loss = 1/(1+d**2) + fs/(10000) * np.sin(10*d) + 0.1*(d-0.5)**2
    return loss

def grad_function(x):
    grad = np.zeros_like(x)
    d = x[0]
    fs = x[1]
    grad[0] = -2*d/(1+d**2)**2 - fs*np.cos(10*d)/(1000) + 0.2*(d-0.5)
    grad[1] = np.sin(10*d)/(10000)
    return grad

if __name__ == "__main__":
    init_point = np.array([0.3, 5000])
    optimal_point, optimal_value = steepest_descent(function, grad_function, init_point)
    print("Ponto encontrado:", optimal_point)
    print("Valor ótimo:", optimal_value)