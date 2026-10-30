# simulador_path.p
import numpy as np
import matplotlib.pyplot as plt
import heapq

class AStarPlanner2D:
    def __init__(self, width, height, resolution):
        self.width = width
        self.height = height
        self.res = resolution
        self.obstacles = set()

    def add_obstacle_circle(self, cx, cy, radius):
        """Adiciona um obstáculo circular expandido no grid 2D"""
        for x in np.arange(cx - radius, cx + radius, self.res):
            for y in np.arange(cy - radius, cy + radius, self.res):
                if np.hypot(x - cx, y - cy) <= radius:
                    ix, iy = int(x / self.res), int(y / self.res)
                    self.obstacles.add((ix, iy))

    def heuristic(self, n1, n2):
        """Heurística baseada na Distância Euclidiana (Cálculo de Linha Reta)"""
        return np.hypot(n1[0] - n2[0], n1[1] - n2[1])

    def planning(self, start_pos, goal_pos):
        """Executa o algoritmo A* para encontrar o menor caminho 2D"""
        start = (int(start_pos[0] / self.res), int(start_pos[1] / self.res))
        goal = (int(goal_pos[0] / self.res), int(goal_pos[1] / self.res))

        # Movimentos permitidos no Grid (8-conectado: ortogonais e diagonais)
        movements = [
            (1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
            (1, 1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (-1, -1, np.sqrt(2))
        ]

        open_set = []
        heapq.heappush(open_set, (0.0, start))
        
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                # Reconstrução do caminho ótimo
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            for dx, dy, cost in movements:
                neighbor = (current[0] + dx, current[1] + dy)

                # Verifica limites do mapa e colisões com obstáculos
                if neighbor in self.obstacles:
                    continue
                if neighbor[0] < 0 or neighbor[0] >= int(self.width / self.res) or \
                   neighbor[1] < 0 or neighbor[1] >= int(self.height / self.res):
                    continue

                tentative_g_score = g_score[current] + cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return [] # Retorna vazio se não achar caminho

# ==============================================================================
# CONFIGURAÇÃO E EXECUÇÃO DA SIMULAÇÃO 2D
# ==============================================================================
# Dimensões do mapa (10x10 metros) com resolução de 0.2m por célula
planner = AStarPlanner2D(width=10.0, height=10.0, resolution=0.2)

# Adicionando obstáculos circulares (simulando prédios/morros no plano 2D)
planner.add_obstacle_circle(cx=5.0, cy=5.0, radius=1.5)
planner.add_obstacle_circle(cx=3.0, cy=7.0, radius=1.0)

start = (1.0, 1.0) # Ponto Inicial (A)
goal = (8.0, 8.0)  # Ponto de Destino (B)

path = planner.planning(start, goal)

# ==============================================================================
# PLOTAGEM GRÁFICA DA TRAYETÓRIA 2D
# ==============================================================================
plt.figure(figsize=(8, 8))

# Desenha os obstáculos
if planner.obstacles:
    obs_x = [o[0] * planner.res for o in planner.obstacles]
    obs_y = [o[1] * planner.res for o in planner.obstacles]
    plt.scatter(obs_x, obs_y, c='gray', s=15, label='Obstáculos Expandidos')

# Desenha a trajetória planejada
if path:
    path_x = [p[0] * planner.res for p in path]
    path_y = [p[1] * planner.res for p in path]
    plt.plot(path_x, path_y, '-r', linewidth=3, label='Trajetória Ótima (A*)')
else:
    print("Nenhum caminho encontrado!")

plt.plot(start[0], start[1], 'go', markersize=12, label='Início (A)')
plt.plot(goal[0], goal[1], 'bo', markersize=12, label='Destino (B)')

plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xlabel('Eixo X (Esquerda <---> Direita) [m]')
plt.ylabel('Eixo Y (Trás <---> Frente) [m]')
plt.title('Planejador de Trajetória 2D - Algoritmo A* com Heurística Euclidiana')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
