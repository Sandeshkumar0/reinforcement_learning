import heapq
import math

class PathFinder:
    def __init__(self, game_map, cell_size=20):
        self.game_map = game_map
        self.cell_size = cell_size
        self.cols = game_map.width // cell_size
        self.rows = game_map.height // cell_size
        
        # Precompute grid
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.update_grid()

    def update_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):
                cx = c * self.cell_size + self.cell_size / 2
                cy = r * self.cell_size + self.cell_size / 2
                # Check collision with radius roughly half cell size
                if self.game_map.is_collision(cx, cy, self.cell_size / 2):
                    self.grid[r][c] = 1 # Obstacle
                else:
                    self.grid[r][c] = 0

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_path(self, start_pos, goal_pos):
        # Convert to grid coordinates
        start = (int(start_pos[1] // self.cell_size), int(start_pos[0] // self.cell_size))
        goal = (int(goal_pos[1] // self.cell_size), int(goal_pos[0] // self.cell_size))
        
        # clamp to grid
        start = (max(0, min(self.rows - 1, start[0])), max(0, min(self.cols - 1, start[1])))
        goal = (max(0, min(self.rows - 1, goal[0])), max(0, min(self.cols - 1, goal[1])))

        if self.grid[goal[0]][goal[1]] == 1:
            # Goal is in obstacle, just move directly (will hit obstacle later)
            return []

        neighbors = [(0,1),(0,-1),(1,0),(-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        close_set = set()
        came_from = {}
        gscore = {start:0}
        fscore = {start:self.heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[start], start))

        while oheap:
            current = heapq.heappop(oheap)[1]

            if current == goal:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                return data[::-1] # Return reversed path

            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                if 0 <= neighbor[0] < self.rows:
                    if 0 <= neighbor[1] < self.cols:
                        if self.grid[neighbor[0]][neighbor[1]] == 1:
                            continue
                    else:
                        continue
                else:
                    continue

                if neighbor in close_set:
                    continue

                # Diagonal weight vs orthogonal
                weight = 1.414 if i != 0 and j != 0 else 1.0
                tentative_g_score = gscore[current] + weight

                if neighbor not in [i[1] for i in oheap] or tentative_g_score < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))

        return []
        
    def get_next_waypoint(self, start_pos, goal_pos):
        path = self.get_path(start_pos, goal_pos)
        if path and len(path) > 0:
            next_node = path[0]
            # Convert back to world coordinates
            wx = next_node[1] * self.cell_size + self.cell_size / 2
            wy = next_node[0] * self.cell_size + self.cell_size / 2
            return wx, wy
        return None
