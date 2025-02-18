import numpy as np
import matplotlib.pyplot as plt

# Función original
def f(x):
    return np.sin(x)-x/2

# Interpolación de Lagrange
def lagrange_interpolation(x, x_points, y_points):
    n = len(x_points)
    result = 0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

# Método de Bisección
def bisect(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) > 0:
        raise ValueError("El intervalo no contiene una raíz")

    iteraciones = []
    errores_abs = []
    errores_rel = []
    errores_cua = []

    print("\nIteraciones del Método de Bisección:")
    print("{:<6} {:<12} {:<12} {:<12} {:<12} {:<18} {:<18} {:<18}".format(
        "Iter |", " a  |", " b |", " c |", " f(c) |", " Error Absoluto |", " Error Relativo |", "Error Cuadrático"))

    c_old = a
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        iteraciones.append(c)
        error_abs = abs(c - c_old)
        error_rel = error_abs / c if c != 0 else 0
        error_cua = error_abs ** 2
        errores_abs.append(error_abs)
        errores_rel.append(error_rel)
        errores_cua.append(error_cua)

        print(f"{i:<6} |{a:<12.6f} |{b:<12.6f} |{c:<12.6f} |{func(c):<12.6f}| {error_abs:<18.6e} |{error_rel:<18.6e} |{error_cua:<18.6e}")

        if abs(func(c)) < tol or (b - a) / 2 < tol:
            return c, iteraciones, errores_abs, errores_rel, errores_cua

        if func(a) * func(c) < 0:
            b = c
        else:
            a = c

        c_old = c

    # Si el método alcanza el máximo de iteraciones, devolver la mejor estimación con listas vacías
    return (a + b) / 2, iteraciones, errores_abs, errores_rel, errores_cua # Retorna la mejor estimación de la raíz y los errores 

# Selección de tres puntos de interpolación
x0 = 0.0
x1 = 1.0
x2 = 2.0
x_points = np.array([x0, x1, x2])
y_points = f(x_points)

# Construcción del polinomio interpolante
# mediante interpolacion de Lagrange
x_vals = np.linspace(x0, x2, 100)
y_interp = [lagrange_interpolation(x, x_points, y_points) for x in x_vals]


# Encontrar raíz del polinomio interpolante usando bisección
# en el intervalo inducido por los puntos donde se hace la interpolacion

# Llamada corregida a la función bisección
root, iteraciones, errores_abs, errores_rel, errores_cua = bisect(
    lambda x: lagrange_interpolation(x, x_points, y_points), x0, x2 )
# Gráfica
plt.figure(figsize=(8, 6))
plt.plot(x_vals, f(x_vals), label="f(x) = sen(x)-x/2", linestyle='dashed', color='blue')
plt.plot(x_vals, y_interp, label="Interpolación de Lagrange", color='red')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(root, color='green', linestyle='dotted', label=f"Raíz aproximada: {root:.4f}")
plt.scatter(x_points, y_points, color='black', label="Puntos de interpolación")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Interpolación y búsqueda de raíces")
plt.legend()
plt.grid(True)

plt.show()

# ----- GRAFICAR LOS ERRORES -----

casos = np.arange(1, len(errores_abs) + 1)

plt.figure(figsize=(8, 5))
plt.plot(casos, errores_abs, marker='o', linestyle='-', label="Error Absoluto", color='blue')
plt.plot(casos, errores_rel, marker='s', linestyle='--', label="Error Relativo", color='green')
plt.plot(casos, errores_cua, marker='^', linestyle='-.', label="Error Cuadrático", color='red')

plt.xlabel("Iteración")
plt.ylabel("Valor del Error")
plt.title("Comparación de Errores en el Método de Bisección")
plt.yscale("log")  # Escala logarítmica para mejor visualización si hay valores pequeños
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

# Imprimir la raíz encontrada
print(f"La raíz aproximada usando interpolación es: {root:.4f}")