import pygame
import random
from collections import deque
import heapq
import time

# Constants
WIDTH = 800
HEIGHT = 600
CELL_SIZE = 20
COLS = WIDTH // CELL_SIZE
ROWS = HEIGHT // CELL_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)

class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.walls = {'top': True, 'right': True, 'bottom': True, 'left': True}
        self.visited = False
        
    def draw(self, screen, color=WHITE):
        x = self.x * CELL_SIZE
        y = self.y * CELL_SIZE
        
        pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))
        
        if self.walls['top']:
            pygame.draw.line(screen, BLACK, (x, y), (x + CELL_SIZE, y), 2)
        if self.walls['right']:
            pygame.draw.line(screen, BLACK, (x + CELL_SIZE, y), (x + CELL_SIZE, y + CELL_SIZE), 2)
        if self.walls['bottom']:
            pygame.draw.line(screen, BLACK, (x, y + CELL_SIZE), (x + CELL_SIZE, y + CELL_SIZE), 2)
        if self.walls['left']:
            pygame.draw.line(screen, BLACK, (x, y), (x, y + CELL_SIZE), 2)

class Maze:
    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.grid = [[Cell(x, y) for y in range(rows)] for x in range(cols)]
        self.start = (0, 0)
        self.end = (cols - 1, rows - 1)
        
    def get_cell(self, x, y):
        if 0 <= x < self.cols and 0 <= y < self.rows:
            return self.grid[x][y]
        return None
    
    def get_neighbors(self, cell):
        neighbors = []
        directions = [
            (0, -1, 'top', 'bottom'),
            (1, 0, 'right', 'left'),
            (0, 1, 'bottom', 'top'),
            (-1, 0, 'left', 'right')
        ]
        
        for dx, dy, wall1, wall2 in directions:
            neighbor = self.get_cell(cell.x + dx, cell.y + dy)
            if neighbor:
                neighbors.append((neighbor, wall1, wall2))
        return neighbors
    
    def remove_walls(self, cell1, cell2):
        dx = cell1.x - cell2.x
        dy = cell1.y - cell2.y
        
        if dx == 1:
            cell1.walls['left'] = False
            cell2.walls['right'] = False
        elif dx == -1:
            cell1.walls['right'] = False
            cell2.walls['left'] = False
        if dy == 1:
            cell1.walls['top'] = False
            cell2.walls['bottom'] = False
        elif dy == -1:
            cell1.walls['bottom'] = False
            cell2.walls['top'] = False
    
    def generate_recursive_backtracking(self):
        """DFS-based maze generation"""
        stack = [self.grid[0][0]]
        self.grid[0][0].visited = True
        
        while stack:
            current = stack[-1]
            unvisited = [n for n, _, _ in self.get_neighbors(current) if not n.visited]
            
            if unvisited:
                next_cell = random.choice(unvisited)
                self.remove_walls(current, next_cell)
                next_cell.visited = True
                stack.append(next_cell)
            else:
                stack.pop()
        
        # Reset visited for solving
        for row in self.grid:
            for cell in row:
                cell.visited = False
    
    def generate_prims(self):
        """Prim's algorithm for maze generation"""
        start_cell = self.grid[0][0]
        start_cell.visited = True
        walls = []
        
        for neighbor, wall1, wall2 in self.get_neighbors(start_cell):
            walls.append((start_cell, neighbor, wall1, wall2))
        
        while walls:
            current, neighbor, wall1, wall2 = random.choice(walls)
            walls.remove((current, neighbor, wall1, wall2))
            
            if not neighbor.visited:
                neighbor.visited = True
                current.walls[wall1] = False
                neighbor.walls[wall2] = False
                
                for next_neighbor, w1, w2 in self.get_neighbors(neighbor):
                    if not next_neighbor.visited:
                        walls.append((neighbor, next_neighbor, w1, w2))
        
        # Reset visited
        for row in self.grid:
            for cell in row:
                cell.visited = False
    
    def can_move(self, from_cell, to_cell):
        """Check if movement between cells is possible"""
        dx = to_cell.x - from_cell.x
        dy = to_cell.y - from_cell.y
        
        if dx == 1 and not from_cell.walls['right']:
            return True
        elif dx == -1 and not from_cell.walls['left']:
            return True
        elif dy == 1 and not from_cell.walls['bottom']:
            return True
        elif dy == -1 and not from_cell.walls['top']:
            return True
        return False
    
    def solve_dfs(self, screen, visualize=True):
        """Depth-First Search solver"""
        stack = [(self.start, [self.start])]
        visited = set([self.start])
        
        while stack:
            (x, y), path = stack.pop()
            current = self.grid[x][y]
            
            if visualize:
                current.draw(screen, BLUE)
                pygame.display.flip()
                pygame.time.delay(10)
            
            if (x, y) == self.end:
                return path
            
            for neighbor, _, _ in self.get_neighbors(current):
                if (neighbor.x, neighbor.y) not in visited and self.can_move(current, neighbor):
                    visited.add((neighbor.x, neighbor.y))
                    stack.append(((neighbor.x, neighbor.y), path + [(neighbor.x, neighbor.y)]))
        
        return []
    
    def solve_bfs(self, screen, visualize=True):
        """Breadth-First Search solver"""
        queue = deque([(self.start, [self.start])])
        visited = set([self.start])
        
        while queue:
            (x, y), path = queue.popleft()
            current = self.grid[x][y]
            
            if visualize:
                current.draw(screen, ORANGE)
                pygame.display.flip()
                pygame.time.delay(10)
            
            if (x, y) == self.end:
                return path
            
            for neighbor, _, _ in self.get_neighbors(current):
                if (neighbor.x, neighbor.y) not in visited and self.can_move(current, neighbor):
                    visited.add((neighbor.x, neighbor.y))
                    queue.append(((neighbor.x, neighbor.y), path + [(neighbor.x, neighbor.y)]))
        
        return []
    
    def solve_astar(self, screen, visualize=True):
        """A* pathfinding algorithm"""
        def heuristic(pos):
            return abs(pos[0] - self.end[0]) + abs(pos[1] - self.end[1])
        
        open_set = [(0, self.start, [self.start])]
        visited = set()
        g_score = {self.start: 0}
        
        while open_set:
            f, (x, y), path = heapq.heappop(open_set)
            
            if (x, y) in visited:
                continue
            
            visited.add((x, y))
            current = self.grid[x][y]
            
            if visualize:
                current.draw(screen, PURPLE)
                pygame.display.flip()
                pygame.time.delay(10)
            
            if (x, y) == self.end:
                return path
            
            for neighbor, _, _ in self.get_neighbors(current):
                neighbor_pos = (neighbor.x, neighbor.y)
                if neighbor_pos not in visited and self.can_move(current, neighbor):
                    tentative_g = g_score[(x, y)] + 1
                    
                    if neighbor_pos not in g_score or tentative_g < g_score[neighbor_pos]:
                        g_score[neighbor_pos] = tentative_g
                        f_score = tentative_g + heuristic(neighbor_pos)
                        heapq.heappush(open_set, (f_score, neighbor_pos, path + [neighbor_pos]))
        
        return []
    
    def draw(self, screen):
        for row in self.grid:
            for cell in row:
                cell.draw(screen)
        
        # Draw start and end
        start_cell = self.grid[self.start[0]][self.start[1]]
        end_cell = self.grid[self.end[0]][self.end[1]]
        start_cell.draw(screen, GREEN)
        end_cell.draw(screen, RED)
    
    def draw_path(self, screen, path):
        for x, y in path:
            cell = self.grid[x][y]
            cell.draw(screen, YELLOW)

def draw_ui(screen, font, algorithm_name, generation_name):
    """Draw UI elements"""
    ui_text = [
        f"Generation: {generation_name}",
        f"Solver: {algorithm_name}",
        "",
        "Controls:",
        "G - Generate (Recursive)",
        "P - Generate (Prim's)",
        "1 - Solve with DFS",
        "2 - Solve with BFS",
        "3 - Solve with A*",
        "R - Reset",
        "Q - Quit"
    ]
    
    y_offset = 10
    for text in ui_text:
        surface = font.render(text, True, BLACK, WHITE)
        screen.blit(surface, (10, y_offset))
        y_offset += 25

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Maze Generator & Solver")
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()
    
    maze = Maze(COLS, ROWS)
    maze.generate_recursive_backtracking()
    
    current_algorithm = "None"
    generation_algorithm = "Recursive Backtracking"
    path = []
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                
                elif event.key == pygame.K_g:
                    maze = Maze(COLS, ROWS)
                    maze.generate_recursive_backtracking()
                    generation_algorithm = "Recursive Backtracking"
                    current_algorithm = "None"
                    path = []
                
                elif event.key == pygame.K_p:
                    maze = Maze(COLS, ROWS)
                    maze.generate_prims()
                    generation_algorithm = "Prim's Algorithm"
                    current_algorithm = "None"
                    path = []
                
                elif event.key == pygame.K_1:
                    current_algorithm = "DFS"
                    screen.fill(WHITE)
                    maze.draw(screen)
                    pygame.display.flip()
                    start_time = time.time()
                    path = maze.solve_dfs(screen, visualize=True)
                    end_time = time.time()
                    print(f"DFS: {len(path)} steps, {end_time - start_time:.4f}s")
                
                elif event.key == pygame.K_2:
                    current_algorithm = "BFS"
                    screen.fill(WHITE)
                    maze.draw(screen)
                    pygame.display.flip()
                    start_time = time.time()
                    path = maze.solve_bfs(screen, visualize=True)
                    end_time = time.time()
                    print(f"BFS: {len(path)} steps, {end_time - start_time:.4f}s")
                
                elif event.key == pygame.K_3:
                    current_algorithm = "A*"
                    screen.fill(WHITE)
                    maze.draw(screen)
                    pygame.display.flip()
                    start_time = time.time()
                    path = maze.solve_astar(screen, visualize=True)
                    end_time = time.time()
                    print(f"A*: {len(path)} steps, {end_time - start_time:.4f}s")
                
                elif event.key == pygame.K_r:
                    maze = Maze(COLS, ROWS)
                    maze.generate_recursive_backtracking()
                    generation_algorithm = "Recursive Backtracking"
                    current_algorithm = "None"
                    path = []
        
        screen.fill(WHITE)
        maze.draw(screen)
        
        if path:
            maze.draw_path(screen, path)
        
        draw_ui(screen, font, current_algorithm, generation_algorithm)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()