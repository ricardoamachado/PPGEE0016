import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

def steepest_descent(function:Callable, gradient:Callable, x_init:np.ndarray, alpha = 0.01, tol = 1e-8, num_iter = 1000):
    """Algoritmo do Steepest Descent (Gradiente Descendente)
    Inputs:
        function: função objetivo a ser minimizada
        gradient: função que calcula o gradiente da função objetivo
        x_init: estimativa inicial
        alpha: passo inicial para a atualização do ponto
        tol: tolerância para o critério de parada
        num_iter: número máximo de iterações
    Outputs:
        x_new: ponto ótimo encontrado
        function(x_new): valor da função objetivo no ponto ótimo
    """
    x_k = x_init
    for k in range(num_iter):
        grad_k = gradient(x_k)
        alpha = backtracking(function, alpha, x_k, grad_k)
        x_new = x_k - alpha * grad_k
        if np.linalg.norm(alpha * grad_k) < tol:
            break
        x_k = x_new
    return x_new, function(x_new)



def backtracking(function:Callable, step:float, x_k:np.ndarray, grad_k:np.ndarray, c = 1e-4, tol = 1e-8):
    """Algoritmo de backtracking para ajuste do passo
    Inputs:
        function: função objetivo a ser minimizada
        step: passo inicial
        x_k: ponto atual
        grad_k: gradiente no ponto atual
        c: parâmetro do critério de Armijo
        tol: tolerância para o tamanho do passo
    Outputs:
        step: passo ajustado
    """
    f_x = function(x_k)
    while function(x_k - step * grad_k) >= (f_x - c * step * np.linalg.norm(grad_k)**2):
        # Fator de contração do passo = 0.5
        step *= 0.5
        if step < tol:
            break

    return step

def function(x):
    d = x[0]
    fs = x[1]

    loss = 1/(1+d**2) + fs/(10000) * np.sin(10*d) + 0.1*(d-0.5)**2
    return loss

def grad_function(x):
    grad = np.zeros_like(x)
    d = x[0]
    fs = x[1]
    grad[0] = -2*d/(1+d**2)**2 + fs*np.cos(10*d)/(1000) + 0.2*(d-0.5)
    grad[1] = np.sin(10*d)/(10000)
    return grad

if __name__ == "__main__":
    init_point = np.array([0.2, 3000])
    optimal_point, optimal_value = steepest_descent(function, grad_function, init_point)
    print("Ponto encontrado:", optimal_point)
    print("Valor ótimo:", optimal_value)
    d_range = np.linspace(0,1,100)
    fs_range = np.linspace(0,20000,200)  # reduced resolution to keep mesh size reasonable
    D, FS = np.meshgrid(d_range, fs_range)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plot the surface
    Z = function(np.array([D, FS]))
    ax.plot_surface(D, FS, Z, cmap='viridis')

    plt.show()