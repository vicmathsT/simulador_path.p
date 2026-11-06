"""
Núcleo numérico de Cálculo I, II e III implementado pela equipe.

Nenhuma função aqui usa rotinas prontas de derivação/integração
(np.gradient, np.trapz, scipy.integrate...). O NumPy é usado apenas
para aritmética básica de vetores.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Cálculo III: derivadas parciais numéricas (diferenças centrais)
# ---------------------------------------------------------------------------
def derivada_parcial(f, x, y, var, h=1e-5):
    """
    ∂f/∂x ≈ [f(x+h, y) - f(x-h, y)] / 2h      (erro O(h²))
    ∂f/∂y ≈ [f(x, y+h) - f(x, y-h)] / 2h
    """
    if var == "x":
        return (f(x + h, y) - f(x - h, y)) / (2 * h)
    return (f(x, y + h) - f(x, y - h)) / (2 * h)


def gradiente_numerico(f, x, y, h=1e-5):
    """∇f ≈ (∂f/∂x, ∂f/∂y) por diferenças centrais."""
    return np.array([derivada_parcial(f, x, y, "x", h),
                     derivada_parcial(f, x, y, "y", h)])


def derivada_direcional(grad, direcao):
    """D_u f = ∇f · u, com u unitário: taxa de subida do terreno na direção do voo."""
    u = np.asarray(direcao, dtype=float)
    norma = np.hypot(u[0], u[1])
    if norma == 0:
        return 0.0
    return float(grad[0] * u[0] / norma + grad[1] * u[1] / norma)


# ---------------------------------------------------------------------------
# Cálculo I: comprimento de arco  L = ∫ sqrt((dx/dt)² + (dy/dt)²) dt
# ---------------------------------------------------------------------------
def comprimento_de_arco(caminho):
    """
    Trajetória discreta r(t_k) = (x_k, y_k) com t_k = k e Δt = 1.
    Em cada intervalo, r'(t) ≈ (x_{k+1} - x_k, y_{k+1} - y_k) / Δt (diferença progressiva),
    e a integral vira a soma de Riemann  L ≈ Σ |r'(t_k)| Δt.
    """
    p = np.asarray(caminho, dtype=float)
    dt = 1.0
    total = 0.0
    for k in range(len(p) - 1):
        vx = (p[k + 1, 0] - p[k, 0]) / dt
        vy = (p[k + 1, 1] - p[k, 1]) / dt
        total += np.sqrt(vx ** 2 + vy ** 2) * dt
    return float(total)


def comprimento_de_arco_3d(caminho, f):
    """
    Comprimento do voo acompanhando o relevo (drone a altura constante do solo):
        L = ∫ sqrt(x'² + y'² + z'²) dt,   com z(t) = f(x(t), y(t))
    """
    p = np.asarray(caminho, dtype=float)
    total = 0.0
    for k in range(len(p) - 1):
        dx = p[k + 1, 0] - p[k, 0]
        dy = p[k + 1, 1] - p[k, 1]
        dz = f(p[k + 1, 0], p[k + 1, 1]) - f(p[k, 0], p[k, 1])
        total += np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return float(total)


# ---------------------------------------------------------------------------
# Cálculo III: integral de linha  W = ∫_C F · dr  (trabalho do vento)
# ---------------------------------------------------------------------------
def integral_de_linha(campo, caminho):
    """
    W = ∫ P dx + Q dy, aproximada pelo ponto médio de cada segmento.
    W > 0: o vento ajudou o drone; W < 0: o drone gastou energia contra o vento.
    """
    p = np.asarray(caminho, dtype=float)
    W = 0.0
    for k in range(len(p) - 1):
        xm = (p[k, 0] + p[k + 1, 0]) / 2
        ym = (p[k, 1] + p[k + 1, 1]) / 2
        P, Q = campo(xm, ym)
        W += P * (p[k + 1, 0] - p[k, 0]) + Q * (p[k + 1, 1] - p[k, 1])
    return float(W)


# ---------------------------------------------------------------------------
# Cálculo III: integral dupla  V = ∬_R f(x, y) dA  (soma de Riemann, ponto médio)
# ---------------------------------------------------------------------------
def integral_dupla(f, x0, x1, y0, y1, n=200):
    """
    ∬ f dA ≈ Σ Σ f(x_i*, y_j*) Δx Δy
    Ex.: volume de relevo (terra/obstáculo) dentro da região de voo.
    """
    dx = (x1 - x0) / n
    dy = (y1 - y0) / n
    total = 0.0
    for i in range(n):
        xi = x0 + (i + 0.5) * dx
        for j in range(n):
            yj = y0 + (j + 0.5) * dy
            total += f(xi, yj)
    return float(total * dx * dy)
