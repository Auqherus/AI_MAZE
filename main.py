# main.py

import pygame
import sys
import time
from maze_gen import Maze
from agent_qlearning import QLearningTable

# Maze dimensions (ncols, nrows)
nx, ny = 5, 5
# Maze entry position
ix, iy = 0, 0

maze = Maze(nx, ny, ix, iy)
maze.make_maze()

# Q-learning agent initialization
actions = ['N', 'S', 'E', 'W']  # Possible actions: move North, South, East, or West
q_agent = QLearningTable(actions)

# Pygame initialization
pygame.init()

# Set up the display
cell_size = 40
width, height = nx * cell_size, ny * cell_size
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Losowa Reprezentacja Labiryntu")

# Pygame colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

def show_progress(current, total):
    """Show the progress of the current episode out of the total episodes."""
    progress = (current + 1) / total * 100
    print(f"Episode {current + 1}/{total} - Progress: {progress:.2f}%")

# Main loop
best_path = None
total_episodes = 50  # Set the total number of episodes

for episode in range(total_episodes):
    show_progress(episode, total_episodes)

    running = True
    agent_x, agent_y = ix, iy
    start_x, start_y = ix, iy
    target_x, target_y = nx - 1, ny - 1  # Assuming the target is at the bottom-right corner
    path_taken = []  # Store the path taken during this episode
    steps = 0  # Counter for the number of steps
    total_reward = 0  # Accumulator for total reward in the episode

    # Track the start time of the episode
    start_time = time.time()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Agent's action
        action = q_agent.choose_action((agent_x, agent_y))
        prev_x, prev_y = agent_x, agent_y  # Store previous agent position

        # Determine the next state based on the action (considering walls)
        if action == 'N' and not maze.cell_at(agent_x, agent_y).walls['N']:
            agent_y = max(0, agent_y - 1)
        elif action == 'S' and not maze.cell_at(agent_x, agent_y).walls['S']:
            agent_y = min(ny - 1, agent_y + 1)
        elif action == 'E' and not maze.cell_at(agent_x, agent_y).walls['E']:
            agent_x = min(nx - 1, agent_x + 1)
        elif action == 'W' and not maze.cell_at(agent_x, agent_y).walls['W']:
            agent_x = max(0, agent_x - 1)

        steps += 1  # Increment the step counter

        # Learn from the new state
        next_state = 'terminal' if (agent_x, agent_y) == (target_x, target_y) else (agent_x, agent_y)

        # Assign rewards based on the state transition
        if next_state == 'terminal':
            reward = 1  # Reward for reaching the target
        elif (prev_x, prev_y) == (agent_x, agent_y):
            reward = -1  # Penalty for hitting a wall
        else:
            reward = -0.1  # Small penalty for each step

        q_agent.learn((prev_x, prev_y), action, reward, next_state)

        # Store the path taken during this episode
        path_taken.append((prev_x, prev_y))

        # Accumulate the total reward
        total_reward += reward

        # Draw the maze
        screen.fill(WHITE)
        # Draw the maze on the Pygame window
        maze.draw_maze(screen, cell_size)

        # Draw the starting point
        pygame.draw.rect(screen, GREEN, (start_x * cell_size, start_y * cell_size, cell_size, cell_size))

        # Draw the target point
        pygame.draw.rect(screen, (255, 255, 0), (target_x * cell_size, target_y * cell_size, cell_size, cell_size))

        # Draw the agent
        pygame.draw.rect(screen, RED, (agent_x * cell_size, agent_y * cell_size, cell_size, cell_size))

        # Update the display
        pygame.display.flip()

        # Check if the episode should terminate
        if (agent_x, agent_y) == (target_x, target_y):
            end_time = time.time()  # Track the end time of the episode
            time_taken = end_time - start_time
            print(f"Episode {episode + 1} - Steps: {steps}, Total Reward: {total_reward}, Time: {time_taken:.2f} seconds")
            print(f"Agent reached the yellow square in {steps} steps. Episode ended.")
            running = False

        # Introduce a delay of 250 milliseconds between each step
        pygame.time.delay(50)

    # Check if this episode has the best path
    if best_path is None or len(path_taken) < len(best_path):
        best_path = path_taken

# Draw the best path at step 50
screen.fill(WHITE)
# Draw the maze on the Pygame window
maze.draw_maze(screen, cell_size)
# Draw the starting point
pygame.draw.rect(screen, GREEN, (start_x * cell_size, start_y * cell_size, cell_size, cell_size))

# Draw the target point
pygame.draw.rect(screen, (255, 255, 0), (target_x * cell_size, target_y * cell_size, cell_size, cell_size))

# Draw the best path in green dots
for i in range(len(best_path) - 1):
    x, y = best_path[i]
    next_x, next_y = best_path[i + 1]

    # Draw a line connecting consecutive points in the best path
    pygame.draw.line(screen, GREEN, (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2),
                     (next_x * cell_size + cell_size // 2, next_y * cell_size + cell_size // 2), 5)

# Draw the last point in the best path
x, y = best_path[-1]
pygame.draw.circle(screen, GREEN, (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2), 5)

# Update the display
pygame.display.flip()

# Keep the window open until the user closes it
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
