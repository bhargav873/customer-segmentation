import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# MODULE 1: ENVIRONMENT SETUP
# ==================================================

# 0 = Free space, 1 = Obstacle, 2 = Goal

grid = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 0, 0, 0, 2]
])

start = (0, 0)
goal = (4, 4)

ROWS, COLS = grid.shape

# Actions: Up, Down, Left, Right
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# ==================================================
# MODULE 2: Q-LEARNING SETTINGS
# ==================================================

q_table = np.zeros((ROWS * COLS, len(actions)))

learning_rate = 0.1
discount_factor = 0.9

episodes = 2000
max_steps = 100

exploration_rate = 1.0
min_exploration_rate = 0.05
exploration_decay = 0.995

success_history = []
reward_history = []

# ==================================================
# MODULE 3: HELPER FUNCTIONS
# ==================================================

def state_number(position):
    """Convert a grid position into a state number."""
    return position[0] * COLS + position[1]


def take_action(position, action):
    """Move the robot and return the reward."""

    row = position[0] + actions[action][0]
    col = position[1] + actions[action][1]

    # Prevent movement outside the grid
    if row < 0 or row >= ROWS or col < 0 or col >= COLS:
        return position, -5, False

    # Prevent movement into obstacles
    if grid[row, col] == 1:
        return position, -5, False

    next_position = (row, col)

    # Reward reaching the destination
    if next_position == goal:
        return next_position, 100, True

    # Small penalty for each ordinary move
    return next_position, -1, False


# ==================================================
# MODULE 4: TRAIN THE ROBOT
# ==================================================

print("Training the robot...\n")

for episode in range(episodes):

    position = start
    total_reward = 0
    reached_goal = False

    for step in range(max_steps):

        state = state_number(position)

        # Explore or choose the best-known action
        if np.random.random() < exploration_rate:
            action = np.random.randint(len(actions))
        else:
            action = np.argmax(q_table[state])

        next_position, reward, done = take_action(
            position, action
        )

        next_state = state_number(next_position)

        # Q-Learning update
        old_value = q_table[state, action]

        future_value = np.max(q_table[next_state])

        q_table[state, action] = old_value + learning_rate * (
            reward
            + discount_factor * future_value
            - old_value
        )

        position = next_position
        total_reward += reward

        if done:
            reached_goal = True
            break

    success_history.append(int(reached_goal))
    reward_history.append(total_reward)

    # Gradually reduce exploration
    exploration_rate = max(
        min_exploration_rate,
        exploration_rate * exploration_decay
    )

print("Training completed!")
print("Total training episodes:", episodes)


# ==================================================
# MODULE 5: EVALUATE THE LEARNED ROBOT
# ==================================================

def evaluate_robot():
    """Test the learned policy without random exploration."""

    test_episodes = 100
    successful_attempts = 0
    successful_steps = []

    for episode in range(test_episodes):

        position = start

        for step in range(max_steps):

            state = state_number(position)

            # Choose the action with the highest Q-value
            action = np.argmax(q_table[state])

            next_position, reward, done = take_action(
                position, action
            )

            position = next_position

            if done:
                successful_attempts += 1
                successful_steps.append(step + 1)
                break

    success_rate = (
        successful_attempts / test_episodes
    ) * 100

    print("\n--- Robot Performance ---")
    print("Test episodes:", test_episodes)
    print("Successful attempts:", successful_attempts)
    print(f"Success rate: {success_rate:.2f}%")

    if successful_steps:
        print(
            "Average steps to goal:",
            f"{np.mean(successful_steps):.2f}"
        )

    return success_rate


success_rate = evaluate_robot()


# ==================================================
# MODULE 6: DISPLAY THE LEARNED ROUTE
# ==================================================

def find_route():

    position = start
    path = [position]

    for step in range(max_steps):

        state = state_number(position)
        action = np.argmax(q_table[state])

        next_position, reward, done = take_action(
            position, action
        )

        # Stop if the robot cannot make progress
        if next_position == position:
            break

        position = next_position
        path.append(position)

        if done:
            break

    return path


path = find_route()

print("\n--- Learned Route ---")
print("Route:", path)

if path[-1] == goal:
    print("Result: Robot reached the destination!")
else:
    print("Result: Robot did not reach the destination.")


# ==================================================
# MODULE 7: VISUALIZE THE RESULTS
# ==================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# First graph: Robot navigation
axes[0].imshow(grid, cmap="viridis")

path_rows = [position[0] for position in path]
path_cols = [position[1] for position in path]

axes[0].plot(
    path_cols,
    path_rows,
    marker="o",
    color="red",
    label="Learned Route"
)

axes[0].scatter(
    start[1], start[0],
    color="blue",
    s=100,
    label="Start"
)

axes[0].scatter(
    goal[1], goal[0],
    color="yellow",
    edgecolors="black",
    s=150,
    label="Goal"
)

axes[0].set_title("Robot Navigation")
axes[0].set_xticks(range(COLS))
axes[0].set_yticks(range(ROWS))
axes[0].legend()

# Second graph: Learning progress
window = 100

moving_success_rate = np.convolve(
    success_history,
    np.ones(window) / window,
    mode="valid"
) * 100

axes[1].plot(
    range(window, episodes + 1),
    moving_success_rate
)

axes[1].set_title("Training Progress")
axes[1].set_xlabel("Training Episode")
axes[1].set_ylabel("Success Rate (%)")
axes[1].set_ylim(0, 105)
axes[1].grid(True)

plt.tight_layout()
plt.show()
