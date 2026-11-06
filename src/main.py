"""
Planejador de Trajetórias para Drones: ponto de entrada.

Uso:
    python src/main.py              # abre a janela interativa
    python src/main.py --salvar     # salva docs/resultado.png sem abrir janela
"""

import argparse
import os
import time

import numpy as np

from astar import AStarPlanner2D
from calculo import (comprimento_de_arco, comprimento_de_arco_3d, derivada_direcional,
                     gradiente_numerico, integral_de_linha, integral_dupla)
from planejador import PlanejadorGradiente
from terreno import Terreno

# ===========================================================================
# CONFIGURAÇÃO DO CENÁRIO (mapa 10 x 10 m)
# ===========================================================================
MORROS = [            # (cx, cy, altura, largura)
    (5.0, 5.0, 2.0, 1.0),
    (3.0, 7.0, 1.5, 0.7),
    (7.0, 3.0, 1.2, 0.6),
]
INICIO = (1.0, 1.0)   # Ponto A
DESTINO = (9.0, 9.0)  # Ponto B
ALTITUDE_MAX = 0.8    # Restrição de segurança: f(x, y) <= C


def vento(x, y):
    """Campo vetorial F(x, y) = (P, Q): brisa constante para leste + redemoinho no centro."""
    P = 0.4 - 0.15 * (y - 5)
    Q = 0.15 * (x - 5)
    return P, Q


def relatorio(nome, caminho, terreno, tempo):
    L = comprimento_de_arco(caminho)
    L3 = comprimento_de_arco_3d(caminho, terreno.altura)
    W = integral_de_linha(vento, caminho)
    alturas = terreno.altura(caminho[:, 0], caminho[:, 1])
    print(f"\n--- {nome} ---")
    print(f"  Pontos na trajetória:            {len(caminho)}")
    print(f"  Comprimento de arco no plano L:  {L:.3f} m")
    print(f"  Comprimento acompanhando relevo: {L3:.3f} m")
    print(f"  Trabalho do vento ∫F·dr:          {W:+.3f} J  ({'a favor' if W > 0 else 'contra'})")
    print(f"  Altitude máxima sobrevoada:      {alturas.max():.3f}  (limite C = {ALTITUDE_MAX})")
    print(f"  Tempo de cálculo:                {tempo * 1000:.1f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--salvar", action="store_true", help="salva a figura em docs/ em vez de abrir janela")
    args = parser.parse_args()

    import matplotlib
    if args.salvar:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    terreno = Terreno(MORROS)

    # -----------------------------------------------------------------------
    # Validação: gradiente analítico x gradiente numérico (diferenças centrais)
    # -----------------------------------------------------------------------
    ponto = (4.2, 5.7)
    g_an = terreno.gradiente(*ponto)
    g_num = gradiente_numerico(terreno.altura, *ponto)
    print("=== Validação do gradiente em", ponto, "===")
    print(f"  analítico: ({g_an[0]:+.6f}, {g_an[1]:+.6f})")
    print(f"  numérico:  ({g_num[0]:+.6f}, {g_num[1]:+.6f})")
    print(f"  erro:      {np.hypot(*(g_an - g_num)):.2e}")

    _, inclinacao = terreno.plano_tangente(*ponto)
    rumo = np.array(DESTINO) - np.array(ponto)
    print(f"  inclinação máxima |∇f|: {inclinacao:.3f}")
    print(f"  taxa de subida rumo a B (derivada direcional): {derivada_direcional(g_an, rumo):+.3f}")

    volume = integral_dupla(terreno.altura, 0, 10, 0, 10)
    print(f"\nVolume de relevo na área de voo ∬f dA: {volume:.3f} m³")

    # -----------------------------------------------------------------------
    # Planejamento
    # -----------------------------------------------------------------------
    t0 = time.perf_counter()
    gd = PlanejadorGradiente(terreno, altitude_max=ALTITUDE_MAX)
    caminho_gd, normas, chegou = gd.planejar(INICIO, DESTINO)
    t_gd = time.perf_counter() - t0
    if not chegou:
        print("\n[aviso] gradiente descendente não chegou ao destino (mínimo local?)")

    t0 = time.perf_counter()
    astar = AStarPlanner2D(10.0, 10.0, 0.2)
    astar.obstaculos_do_terreno(terreno, ALTITUDE_MAX)
    caminho_astar = astar.planning(INICIO, DESTINO)
    t_astar = time.perf_counter() - t0

    relatorio("Gradiente Descendente (campo potencial)", caminho_gd, terreno, t_gd)
    if len(caminho_astar):
        relatorio("A* (linha de base da AV1)", caminho_astar, terreno, t_astar)

    # -----------------------------------------------------------------------
    # Visualização (Matplotlib apenas desenha; o cálculo foi feito acima)
    # -----------------------------------------------------------------------
    xs = np.linspace(0, 10, 120)
    X, Y = np.meshgrid(xs, xs)
    Z = terreno.altura(X, Y)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Planejador de Trajetórias para Drones: Gradiente Descendente x A*")

    # 1) Superfície z = f(x, y) com a trajetória sobre o relevo
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot_surface(X, Y, Z, cmap="terrain", alpha=0.7, linewidth=0)
    zc = terreno.altura(caminho_gd[:, 0], caminho_gd[:, 1]) + 0.05
    ax1.plot(caminho_gd[:, 0], caminho_gd[:, 1], zc, "r-", linewidth=2.5)
    ax1.set_title("Superfície do terreno z = f(x, y)")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")

    # 2) Curvas de nível, campo -∇U e as duas trajetórias
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.contourf(X, Y, Z, levels=20, cmap="terrain", alpha=0.6)
    ax2.contour(X, Y, Z, levels=[ALTITUDE_MAX], colors="k", linewidths=2, linestyles="--")
    xq = np.linspace(0.25, 9.75, 20)
    Xq, Yq = np.meshgrid(xq, xq)
    U_x = np.zeros_like(Xq)
    U_y = np.zeros_like(Yq)
    for i in range(Xq.shape[0]):
        for j in range(Xq.shape[1]):
            g = gd.gradiente_potencial(np.array([Xq[i, j], Yq[i, j]]), np.array(DESTINO))
            n = np.hypot(g[0], g[1]) + 1e-9
            U_x[i, j], U_y[i, j] = -g[0] / n, -g[1] / n
    ax2.quiver(Xq, Yq, U_x, U_y, color="gray", alpha=0.6, width=0.003)
    ax2.plot(caminho_gd[:, 0], caminho_gd[:, 1], "r-", linewidth=2.5, label="Gradiente descendente")
    if len(caminho_astar):
        ax2.plot(caminho_astar[:, 0], caminho_astar[:, 1], "b--", linewidth=1.8, label="A*")
    ax2.plot(*INICIO, "go", markersize=12, label="Início (A)")
    ax2.plot(*DESTINO, "bo", markersize=12, label="Destino (B)")
    ax2.set_title(f"Curvas de nível, −∇U e restrição f(x,y) = {ALTITUDE_MAX} (tracejado)")
    ax2.set_aspect("equal")
    ax2.legend(loc="upper left", fontsize=8)

    # 3) Convergência: |∇U| ao longo das iterações
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(normas, color="purple")
    ax3.set_title("Norma do gradiente |∇U| por iteração")
    ax3.set_xlabel("iteração")
    ax3.set_ylabel("|∇U|")
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4) Perfil de altitude sobrevoada x restrição
    ax4 = fig.add_subplot(2, 2, 4)
    s_gd = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(caminho_gd, axis=0).T))])
    ax4.plot(s_gd, terreno.altura(caminho_gd[:, 0], caminho_gd[:, 1]), "r-", label="Gradiente descendente")
    if len(caminho_astar):
        s_as = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(caminho_astar, axis=0).T))])
        ax4.plot(s_as, terreno.altura(caminho_astar[:, 0], caminho_astar[:, 1]), "b--", label="A*")
    ax4.axhline(ALTITUDE_MAX, color="k", linestyle=":", label="limite C")
    ax4.set_title("Altitude do relevo sob o drone")
    ax4.set_xlabel("distância percorrida s [m]")
    ax4.set_ylabel("f(x(s), y(s))")
    ax4.legend(fontsize=8)
    ax4.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    # Sem backend gráfico (ex.: falta o tkinter no Linux), salva a figura em vez de mostrar
    sem_janela = plt.get_backend().lower() in ("agg", "pdf", "svg", "ps", "cairo")
    if sem_janela and not args.salvar:
        print("\n[aviso] nenhuma janela gráfica disponível (no Linux: sudo apt install python3-tk); "
              "salvando a figura em docs/")
    if args.salvar or sem_janela:
        destino_png = os.path.join(os.path.dirname(__file__), "..", "docs", "resultado.png")
        os.makedirs(os.path.dirname(destino_png), exist_ok=True)
        fig.savefig(destino_png, dpi=110)
        print(f"\nFigura salva em {os.path.normpath(destino_png)}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
