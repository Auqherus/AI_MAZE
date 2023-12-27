# main.py

import matplotlib.pyplot as plt
import pygame
import time
import sys
from maze_gen import Maze
from agent_qlearning import QLearningTable

# Maze dimensions (ncols, nrows)
nx, ny = 8, 8
# Maze entry position
ix, iy = 0, 0

maze = Maze(nx, ny, ix, iy)
maze.make_maze()

# Q-learning agent initialization
actions = ['N', 'S', 'E', 'W']  # Possible actions: move North, South, East, or West
initial_epsilon = 0.7  # Adjust this value to control initial exploration
q_agent = QLearningTable(actions, e_greedy=initial_epsilon)


# Pygame initialization
pygame.init()

# Set up the display
cell_size = 35
width, height = nx * cell_size, ny * cell_size
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Losowa Reprezentacja Labiryntu")

# Pygame colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

def show_progress(current, total):
    # Show the progress of the current episode out of the total episodes
    progress = (current + 1) / total * 100
    print(f"Episode {current + 1}/{total} - Progress: {progress:.2f}%")

def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    points.append((x, y))
    return points

# Create a list to store the number of steps in each episode
steps_history = []

# Main loop
best_path = None
total_episodes = 500  # Set the total number of episodes



for episode in range(total_episodes):
    show_progress(episode, total_episodes)

    total_reward = 0  # Accumulator for total reward in the episode
    steps = 0  # Counter for the number of steps

    running = True
    agent_x, agent_y = ix, iy
    start_x, start_y = ix, iy
    target_x, target_y = nx - 1, ny - 1  # Assuming the target is at the bottom-right corner
    path_taken = []  # Store the path taken during this episode


    # Track the start time of the episode
    start_time = time.time()
    reached_target = False
    prev_x, prev_y = agent_x, agent_y  # Initialize prev_x and prev_y before the loop

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not reached_target:
            # Agent's action
            action = q_agent.choose_action((agent_x, agent_y))
        else:
            # Move towards the target
            if agent_x < target_x:
                action = 'E'
            elif agent_x > target_x:
                action = 'W'
            elif agent_y < target_y:
                action = 'S'
            elif agent_y > target_y:
                action = 'N'
            else:
                action = None  # No action needed, as the agent has reached the target

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
            reward = 10  # Reward for reaching the target
            reached_target = True  # Set the flag to True when the agent reaches the target
        elif (prev_x, prev_y) == (agent_x, agent_y):
            reward = -0.7  # Penalty for hitting a wall
        else:
            reward = -0.2  # Small penalty for each step

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

        # Calculate the center of the circle
        circle_center = (int(agent_x * cell_size + cell_size / 2), int(agent_y * cell_size + cell_size / 2))

        # Calculate the radius of the circle (half of the cell size)
        circle_radius = int(cell_size / 2)

        # Draw the circle
        pygame.draw.circle(screen, RED, circle_center, circle_radius)

        # Update the display
        pygame.display.flip()

        # Check if the episode should terminate
        if reached_target:
            end_time = time.time()  # Track the end time of the episode
            time_taken = end_time - start_time
            print(
                f"Episode {episode + 1} - Steps: {steps}, Total Reward: {total_reward}, Time: {time_taken:.2f} seconds")
            print(f"Agent reached the yellow square in {steps} steps. Episode ended.")
            running = False

        # Introduce a delay of 250 milliseconds between each step
        pygame.time.delay(1)

        # Update prev_x and prev_y for the next iteration
        prev_x, prev_y = agent_x, agent_y

    # Store the number of steps in the episode to the history
    steps_history.append(steps)

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

    # Use Bresenham's line algorithm to get points along the line
    points = bresenham_line(x * cell_size + cell_size // 2, y * cell_size + cell_size // 2,
                            next_x * cell_size + cell_size // 2, next_y * cell_size + cell_size // 2)

    # Draw the line only if there are 2 or more points
    if len(points) >= 2:
        pygame.draw.lines(screen, GREEN, False, points, 5)

        # Draw the last point in the best path
        x, y = best_path[-1]
        pygame.draw.circle(screen, GREEN, (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2), 5)


# Update the display
pygame.display.flip()

# Plotting the steps history with a line
plt.plot(steps_history, '-')  # Adjust markersize as needed
plt.title('Number of Steps in Each Episode')
plt.xlabel('Episode')
plt.ylabel('Number of Steps')

# Find the maximum step point in each episode
max_steps_in_episode = [max(steps_history[i:i + total_episodes]) for i in range(0, len(steps_history), total_episodes)]

# Draw lines connecting the maximum step points in each episode
for i in range(1, len(max_steps_in_episode)):
    plt.plot([i * total_episodes, (i + 1) * total_episodes], [max_steps_in_episode[i - 1]] * 2, 'k-')

plt.show()

# Keep the window open until the user closes it
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()