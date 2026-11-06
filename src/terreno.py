"""
Modelagem do terreno z = f(x, y)  (Cálculo II)

O relevo é uma soma de "morros" gaussianos:

    f(x, y) = Σ A_i · exp( -((x - cx_i)² + (y - cy_i)²) / (2·s_i²) )

As derivadas parciais são calculadas ANALITICAMENTE (regra da cadeia),
sem bibliotecas de derivação automática:

    ∂f/∂x = Σ -A_i · (x - cx_i)/s_i² · exp(...)
    ∂f/∂y = Σ -A_i · (y - cy_i)/s_i² · exp(...)
"""

import numpy as np


class Terreno:
    def __init__(self, morros):
        """morros: lista de tuplas (cx, cy, altura, largura)"""
        self.morros = morros

    def altura(self, x, y):
        """f(x, y): altitude do relevo (aceita escalares ou matrizes do NumPy)."""
        z = np.zeros_like(np.asarray(x, dtype=float))
        for cx, cy, A, s in self.morros:
            z = z + A * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s ** 2))
        return z

    def gradiente(self, x, y):
        """∇f(x, y) = (∂f/∂x, ∂f/∂y), derivado à mão pela regra da cadeia."""
        fx = 0.0
        fy = 0.0
        for cx, cy, A, s in self.morros:
            e = A * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s ** 2))
            fx += -(x - cx) / s ** 2 * e
            fy += -(y - cy) / s ** 2 * e
        return np.array([fx, fy])

    def plano_tangente(self, a, b):
        """
        Plano tangente em (a, b):
            z = f(a,b) + f_x(a,b)·(x - a) + f_y(a,b)·(y - b)
        Retorna uma função z(x, y) e a inclinação máxima |∇f| (declive do terreno).
        """
        z0 = self.altura(a, b)
        fx, fy = self.gradiente(a, b)

        def plano(x, y):
            return z0 + fx * (x - a) + fy * (y - b)

        return plano, float(np.hypot(fx, fy))
