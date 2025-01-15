# test.py
import matplotlib.pyplot as plt
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from py4j.java_gateway import JavaGateway
from stable_baselines3.common.evaluation import evaluate_policy

from StreetFighter_tk import StreetFighterEnv  # Import your environment class

# Setup the game and environment
gateway = JavaGateway()
game_instance = gateway.entry_point.getGame()
game_instance.setRenderingEnabled(True)
env = StreetFighterEnv(game=game_instance)
# check_env(env)
print('Environment created for testing.')

# Load the trained model
model = PPO.load("ppo_street_fighter")
env.start_game_auto()
# Testing
num_test_episodes = 100  # Number of episodes for testing
testing_rewards = []
results = {"P1 Won": 0, "P1 Lost": 0, "Draw": 0}  # Dictionary to track win/loss/draw

print('Testing started...')
for episode in range(num_test_episodes):
    obs, _ = env.reset()  # Reset environment
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))  # Fetch result from `info`
        total_reward += reward
        if done:
            # Check the result and update the count
            result = info.get('result')
            if result in results:
                results[result] += 1

    testing_rewards.append(total_reward)
    print(f"Test Episode {episode + 1}: Total Reward = {total_reward}")

# Display the results
print(f"Win/Loss/Draw: {results}")

# Plot win/loss/draw results
labels = list(results.keys())
values = list(results.values())

plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=['green', 'red', 'blue'])
plt.title('Win/Loss/Draw Results (Trained Model)')
plt.xlabel('Result')
plt.ylabel('Number of Episodes')
plt.grid(True)
plt.savefig('win_loss_draw_plot.png')
plt.show()

# Plot testing rewards as before
plt.figure(figsize=(8, 5))
plt.plot(testing_rewards, label='Total Reward (Testing)', marker='o', color='green')
plt.title('Testing Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid()
plt.legend()
plt.savefig('testing_rewards_plot.png')
plt.show()

# # random
# Random actions testing with win/loss/draw tracking
random_rewards = []  # Store rewards for random actions
random_results = {"P1 Won": 0, "P1 Lost": 0, "Draw": 0}  # Track win/loss/draw for random actions
num_random_episodes = 100

print('Testing random actions...')
for episode in range(num_random_episodes):
    print(f'Random action episode: {episode + 1}')
    obs, _ = env.reset()
    total_reward = 0
    done = False
    i = 0
    while not done:
        i += 1
        action = env.action_space.sample()  # Take a random action
        obs, reward, done, truncated, info = env.step(action)  # Get the result in info
        total_reward += reward
        if done:
            # Check the result and update the count
            result = info.get('result')
            if result in random_results:
                random_results[result] += 1

    random_rewards.append(total_reward)
    print(f"Random Episode {episode + 1}: Total Reward = {total_reward}, total actions: {i}")

# Display the random actions results
print(f"Random Action Win/Loss/Draw: {random_results}")

# Plot win/loss/draw results for random actions
labels = list(random_results.keys())
values = list(random_results.values())

plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=['green', 'red', 'blue'])
plt.title('Win/Loss/Draw Results (Random Actions)')
plt.xlabel('Result')
plt.ylabel('Number of Episodes')
plt.grid(True)
plt.savefig('random_win_loss_draw_plot.png')
plt.show()

# Plot comparison of rewards (trained vs random)
plt.figure(figsize=(12, 6))
plt.plot(testing_rewards, label='Trained Model Rewards', marker='o', color='green')
plt.plot(random_rewards, label='Random Action Rewards', marker='o', color='red')
plt.title('Rewards Comparison: Trained Model vs Random Actions')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid()
plt.legend()
plt.savefig('comparison_rewards_plot.png')
plt.show()
