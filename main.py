import pygame
import sys
from maze_gen import Maze

# Maze dimensions (ncols, nrows)
nx, ny = 35, 25
# Maze entry position
ix, iy = 0, 0

maze = Maze(nx, ny, ix, iy)
maze.make_maze()

# Pygame initialization
pygame.init()

# Set up the display
cell_size = 30
width, height = nx * cell_size, ny * cell_size
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Maze Visualization")

# Pygame colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# Draw the maze on the Pygame window
def draw_maze():
    for x in range(nx):
        for y in range(ny):
            cell = maze.cell_at(x, y)
            x_pos, y_pos = x * cell_size, y * cell_size

            # Draw walls
            if cell.walls['N']:
                pygame.draw.line(screen, BLACK, (x_pos, y_pos), (x_pos + cell_size, y_pos), 2)
            if cell.walls['S']:
                pygame.draw.line(screen, BLACK, (x_pos, y_pos + cell_size), (x_pos + cell_size, y_pos + cell_size), 2)
            if cell.walls['E']:
                pygame.draw.line(screen, BLACK, (x_pos + cell_size, y_pos), (x_pos + cell_size, y_pos + cell_size), 2)
            if cell.walls['W']:
                pygame.draw.line(screen, BLACK, (x_pos, y_pos), (x_pos, y_pos + cell_size), 2)


# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Draw the maze
    screen.fill(WHITE)
    draw_maze()

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
