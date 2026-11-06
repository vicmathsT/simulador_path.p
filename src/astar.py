"""
A* em grade 2D (protótipo da AV1), mantido como linha de base para comparação.

Diferença em relação à AV1: os obstáculos não são mais círculos fixos;
são as células onde o relevo viola a restrição de segurança f(x, y) > C.
"""

import heapq
import numpy as np


class AStarPlanner2D:
    def __init__(self, width, height, resolution):
        self.width = width
        self.height = height
        self.res = resolution
        self.obstacles = set()

    def obstaculos_do_terreno(self, terreno, altitude_max):
        """Marca como obstáculo toda célula com f(x, y) > C."""
        for ix in range(int(self.width / self.res)):
            for iy in range(int(self.height / self.res)):
                if terreno.altura(ix * self.res, iy * self.res) > altitude_max:
                    self.obstacles.add((ix, iy))

    def heuristic(self, n1, n2):
        """Heurística: distância euclidiana."""
        return np.hypot(n1[0] - n2[0], n1[1] - n2[1])

    def planning(self, start_pos, goal_pos):
        """Retorna o caminho em coordenadas do mundo (metros)."""
        start = (round(start_pos[0] / self.res), round(start_pos[1] / self.res))
        goal = (round(goal_pos[0] / self.res), round(goal_pos[1] / self.res))

        movements = [
            (1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
            (1, 1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (-1, -1, np.sqrt(2))
        ]

        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return np.array(path[::-1], dtype=float) * self.res

            for dx, dy, cost in movements:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in self.obstacles:
                    continue
                if neighbor[0] < 0 or neighbor[0] >= int(self.width / self.res) or \
                   neighbor[1] < 0 or neighbor[1] >= int(self.height / self.res):
                    continue

                tentative_g_score = g_score[current] + cost
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        return np.empty((0, 2))
