"""
Planejador por Campo Potencial + Gradiente Descendente  (Cálculo III)

Função potencial (o drone "rola ladeira abaixo" em U):

    U(x, y) = ½·k_atr·|p - B|²                    atração ao destino B
            + k_rep·f(x, y)                       repulsão do relevo
            + ½·μ·max(0, f(x, y) - C̃)²           penalidade da restrição f(x,y) ≤ C

com C̃ = C - margem. A penalidade é "macia": o mínimo penalizado fica um pouco
acima de C̃, por isso a margem de segurança garante que o voo real fique abaixo de C.

Gradiente (derivado à mão, regra da cadeia):

    ∇U = k_atr·(p - B) + k_rep·∇f + μ·max(0, f - C̃)·∇f

Atualização do gradiente descendente (passo de comprimento fixo = velocidade constante):

    p_{k+1} = p_k - α · ∇U / |∇U|

A penalidade quadrática é a versão numérica da otimização condicionada:
quanto maior μ, mais o caminho respeita a altitude máxima de segurança C
(equivalente ao método de penalidade, irmão dos Multiplicadores de Lagrange).
"""

import numpy as np


class PlanejadorGradiente:
    def __init__(self, terreno, k_atr=1.0, k_rep=4.0, mu=60.0, altitude_max=1.0, margem=0.15,
                 passo=0.05, momento=0.7, max_iter=3000):
        self.terreno = terreno
        self.momento = momento
        self.k_atr = k_atr
        self.k_rep = k_rep
        self.mu = mu
        self.C = altitude_max - margem   # C̃
        self.passo = passo
        self.max_iter = max_iter

    def potencial(self, x, y, destino):
        """U(x, y): usado para plotar o campo e as curvas de nível do potencial."""
        f = self.terreno.altura(x, y)
        atr = 0.5 * self.k_atr * ((x - destino[0]) ** 2 + (y - destino[1]) ** 2)
        violacao = np.maximum(0.0, f - self.C)
        return atr + self.k_rep * f + 0.5 * self.mu * violacao ** 2

    def gradiente_potencial(self, p, destino):
        """∇U(p) analítico."""
        f = self.terreno.altura(p[0], p[1])
        grad_f = self.terreno.gradiente(p[0], p[1])
        violacao = max(0.0, float(f) - self.C)
        return (self.k_atr * (p - destino)
                + self.k_rep * grad_f
                + self.mu * violacao * grad_f)

    def planejar(self, inicio, destino):
        """
        Executa o gradiente descendente de A até B.
        Retorna (caminho, normas_do_gradiente, chegou).
        """
        p = np.array(inicio, dtype=float)
        destino = np.array(destino, dtype=float)
        caminho = [p.copy()]
        normas = []
        melhor_dist = np.inf
        sem_progresso = 0
        lado = 0.0                  # lado escolhido para contornar um mínimo local
        velocidade = np.zeros(2)    # termo de momento

        for _ in range(self.max_iter):
            dist = float(np.hypot(*(p - destino)))
            if dist < self.passo:
                caminho.append(destino.copy())
                return np.array(caminho), normas, True

            g = self.gradiente_potencial(p, destino)
            norma = float(np.hypot(g[0], g[1]))
            normas.append(norma)

            direcao = -g / norma if norma > 1e-9 else np.zeros(2)

            # Mínimo local: ∇U ≈ 0 ou o drone parou de se aproximar do destino.
            # Saída: soma uma componente perpendicular à reta até o destino,
            # pelo lado onde o potencial U é menor (contorna o morro pela lateral).
            if dist < melhor_dist - 1e-3:
                melhor_dist = dist
                sem_progresso = 0
                lado = 0.0
            else:
                sem_progresso += 1
            if norma < 1e-3 or sem_progresso > 20:
                alvo = (destino - p) / dist
                perpendicular = np.array([-alvo[1], alvo[0]])
                if lado == 0.0:
                    pe = p + 5 * self.passo * perpendicular
                    pd = p - 5 * self.passo * perpendicular
                    u_e = self.potencial(pe[0], pe[1], destino)
                    u_d = self.potencial(pd[0], pd[1], destino)
                    lado = 1.0 if u_e <= u_d else -1.0
                direcao = direcao + lado * perpendicular

            # Momento (média móvel das direções): amortece o zigue-zague
            # na borda da restrição, como no otimizador "SGD com momentum".
            velocidade = self.momento * velocidade + (1 - self.momento) * direcao
            nv = float(np.hypot(velocidade[0], velocidade[1]))
            if nv > 1e-9:
                p = p + self.passo * velocidade / nv
            caminho.append(p.copy())

        return np.array(caminho), normas, False
