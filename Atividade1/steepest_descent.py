import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from scipy.optimize import minimize
import time


# Lista para armazenar o histórico do gradiente
history_grad_bfgs = []
# Lista para armazenar o histórico das variáveis D e fs
history_bfgs_D = []
history_bfgs_fs = []

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

    num_iter_sd = 0
    history_sd_D = []
    history_sd_fs = []
    history_grad_sd = []

    for k in range(num_iter):
        
        grad_k = gradient(x_k)
        history_grad_sd.append(np.linalg.norm((grad_k))) #armazena o valor do gradiente em cada interação
        alpha = backtracking(function, alpha, x_k, grad_k)
        x_new = x_k - alpha * grad_k
        history_sd_D.append(x_new[0]) #armazena o valor de D em cada interação
        history_sd_fs.append(x_new[1]) #armazena o valor de fs em cada interação
        num_iter_sd += 1
        if np.linalg.norm(alpha * grad_k) < tol:
            break
        x_k = x_new
    return x_new, function(x_new), num_iter_sd, history_grad_sd, history_sd_D, history_sd_fs



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


def callback_bfgs(xk, *args):
    gradiente_xk = grad_function(xk) 
    history_grad_bfgs.append(np.linalg.norm(gradiente_xk))#armazena o valor do gradiente em cada interação
    history_bfgs_D.append(xk[0])#armazena o valor de D em cada interação
    history_bfgs_fs.append(xk[1])#armazena o valor de fs em cada interação

def main():
    fun = function
    x0 = np.array([0.3, 15000.0])  # Ponto inicial (D, fs)


    print("Iniciando a otimização com BFGS...")
    print("Função: loss = 1/(1+d**2) + fs/10000 * sin(10*d) + 0.1*(d-0.5)**2")
    print(f"Ponto inicial (D, fs): {x0}")
    num_iter = 1000


    start_time_bfgs = time.time()

    # reset history lists before running BFGS to avoid carrying data from previous runs
    history_grad_bfgs.clear()
    history_bfgs_D.clear()
    history_bfgs_fs.clear()

    result_bfgs = minimize(
        fun,
        x0,
        method='BFGS',
        jac=grad_function,
        callback=callback_bfgs,
        options={'gtol': 1e-4, "maxiter":num_iter}
    )

    end_time_bfgs = time.time()
    time_bfgs = end_time_bfgs - start_time_bfgs

    start_time_sd = time.time()
    x_sd, f_sd, num_iter_sd, history_grad_sd, history_sd_D, history_sd_fs = steepest_descent(function=fun, gradient=grad_function, x_init=x0, alpha=0.01, tol=1e-6, num_iter=num_iter)
    end_time_sd = time.time()
    time_sd = end_time_sd - start_time_sd

    print("Steepest-Descent Method (SD)")
    print("--------------------|-------------------------")
    print(f"Iterações (nit)         | {num_iter_sd:<23} ")
    print(f"Avaliações Função       | {num_iter_sd:<23}")
    print(f"Avaliações Gradiente    | {num_iter_sd:<23}")
    print(f"Valor Final f(x)        | {f_sd:<23.6e}")
    print(f"Ponto encontrado (D,fs))| {x_sd}")
    print(f"Tempo Total             | {time_sd:<23.8f}")
    print(f"Tempo por iteração      | {time_sd/num_iter_sd:<23.8f}")


    print("BFGS (Quase-Newton)")
    print("--------------------|-------------------------")
    print(f"Iterações (nit)         | {result_bfgs.nit:<23} ")
    print(f"Avaliações Função       | {result_bfgs.nfev:<23}")
    print(f"Avaliações Gradiente    | {result_bfgs.njev:<23}")
    print(f"Valor Final f(x)        | {result_bfgs.fun:<23.6e}")
    print(f"Ponto encontrado (D,fs))| {result_bfgs.x}")
    print(f"Tempo Total             | {time_bfgs:<23.8f}")
    print(f"Tempo por iteração      | {time_bfgs/result_bfgs.nit:<23.8f}")

    #Evolução o gradiente
    plt.figure(1,figsize=(15, 6))
    plt.semilogy(history_grad_sd, label='Stepeest-Descent Method (SD)', marker='x', markersize=4)
    plt.semilogy(history_grad_bfgs, label='Quase-Newton BFGS', marker='x', markersize=8)

    plt.title('Curvas de convergência (log||g|| vs Iterações)', fontsize=14)
    plt.xlabel('Número de Iterações (k)', fontsize=12)
    plt.ylabel('log||g|| (Escala Logarítmica)', fontsize=12)
    plt.legend()
    plt.grid(True, which="both", ls="--")



    plt.figure(2, figsize=(15, 7))

    #Evolução da variável D
    plt.subplot(1, 2, 1)
    plt.plot(history_sd_D, label='Stepeest-Descent Method (SD)', marker='o', markersize=4)
    plt.plot(history_bfgs_D, label='Quase-Newton BFGS', marker='x', markersize=8)

    plt.title('Evolução da variável D vs Iterações', fontsize=12)
    plt.xlabel('Número de Iterações (k)', fontsize=10)
    plt.ylabel('D (Ciclo de Trabalho)', fontsize=10)
    plt.legend()
    plt.grid(True, which="both", ls="--")

    #Evolução da variável fs
    plt.subplot(1, 2, 2)
    plt.plot(history_sd_fs, label='Stepeest-Descent Method (SD)', marker='o', markersize=4)
    plt.plot(history_bfgs_fs, label='Quase-Newton BFGS', marker='x', markersize=8)

    plt.title('Evolução da variável fs vs Iterações', fontsize=12)
    plt.xlabel('Número de Iterações (k)', fontsize=10)
    plt.ylabel('fs (Freq de chaveamento)', fontsize=10)
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.show()

if __name__ == "__main__":
    main()


