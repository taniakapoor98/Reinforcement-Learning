# test.py
import time

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

# Load the models
model = PPO.load("ppo_street_fighter_revised")
model_old = PPO.load("ppo_base_actions")
env.start_game_auto()
# Initialize data tracking
num_test_episodes = 100
testing_rewards = []
old_model_rewards = []
random_rewards = []
results = {"P1 Won": 0, "P1 Lost": 0, "Draw": 0}
old_model_results = {"P1 Won": 0, "P1 Lost": 0, "Draw": 0}
random_results = {"P1 Won": 0, "P1 Lost": 0, "Draw": 0}

# Test the current model
print('Testing current model...')
for episode in range(num_test_episodes):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        total_reward += reward
        if done:
            result = info.get('result')
            if result in results:
                results[result] += 1
    testing_rewards.append(total_reward)

# Test the old model
print('Testing old model...')
for episode in range(num_test_episodes):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model_old.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        total_reward += reward
        if done:
            result = info.get('result')
            if result in old_model_results:
                old_model_results[result] += 1
    old_model_rewards.append(total_reward)

# Test random actions
print('Testing random actions...')
for episode in range(num_test_episodes):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = env.action_space.sample()  # Take a random action
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done:
            result = info.get('result')
            print(result,episode)
            if result in random_results:
                random_results[result] += 1
    random_rewards.append(total_reward)

# Display and plot win/loss/draw results for all tests
print(f"Win/Loss/Draw for Current Model: {results}")
print(f"Win/Loss/Draw for Old Model: {old_model_results}")
print(f"Win/Loss/Draw for Random Actions: {random_results}")

# Plot Win/Loss/Draw results for all comparisons
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Current Model Win/Loss/Draw
axes[0].bar(results.keys(), results.values(), color=['green', 'red', 'blue'])
axes[0].set_title('Win/Loss/Draw (Current Model)')
axes[0].set_xlabel('Result')
axes[0].set_ylabel('Number of Episodes')
axes[0].grid(True)

# Old Model Win/Loss/Draw
axes[1].bar(old_model_results.keys(), old_model_results.values(), color=['green', 'red', 'blue'])
axes[1].set_title('Win/Loss/Draw (Old Model)')
axes[1].set_xlabel('Result')
axes[1].set_ylabel('Number of Episodes')
axes[1].grid(True)

# Random Actions Win/Loss/Draw
axes[2].bar(random_results.keys(), random_results.values(), color=['green', 'red', 'blue'])
axes[2].set_title('Win/Loss/Draw (Random Actions)')
axes[2].set_xlabel('Result')
axes[2].set_ylabel('Number of Episodes')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('comparison_win_loss_draw.png')
plt.show()

# Box Plot Comparison of Rewards for all three methods
data = [testing_rewards, old_model_rewards, random_rewards]
labels = ['Current Model', 'Old Model', 'Random Actions']

plt.figure(figsize=(12, 6))
plt.boxplot(data, patch_artist=True, labels=labels, boxprops=dict(facecolor="lightgreen", color="black"))
plt.title('Reward Distribution: Current Model vs Old Model vs Random Actions')
plt.xlabel('Model')
plt.ylabel('Total Reward per Episode')
plt.grid(True)
plt.savefig('comparison_rewards_boxplot.png')
plt.show()

#
# # Load the trained model
# model = PPO.load("ppo_street_fighter")
# env.start_game_auto()
# # Testing
# num_test_episodes = 100  # Number of episodes for testing
# testing_rewards = []
# results = {"P1 Won": 0, "P1 Lost": 0, "Draw": 0}  # Dictionary to track win/loss/draw
#
# print('Testing started...')
# for episode in range(num_test_episodes):
#     obs, _ = env.reset()  # Reset environment
#     total_reward = 0
#     done = False
#     while not done:
#         action, _ = model.predict(obs, deterministic=True)
#
#         obs, reward, done, truncated, info = env.step(int(action))  # Fetch result from `info`
#         total_reward += reward
#         if done:
#             # Check the result and update the count
#             result = info.get('result')
#             if result in results:
#                 results[result] += 1
#
#     testing_rewards.append(total_reward)
#     print(f"Test Episode {episode + 1}: Total Reward = {total_reward}")
#
# # Display the results
# print(f"Win/Loss/Draw: {results}")
#
# # Plot win/loss/draw results
# labels = list(results.keys())
# values = list(results.values())
#
# plt.figure(figsize=(8, 5))
# plt.bar(labels, values, color=['green', 'red', 'blue'])
# plt.title('Win/Loss/Draw Results (Trained Model)')
# plt.xlabel('Result')
# plt.ylabel('Number of Trials')
# plt.grid(True)
# plt.savefig('win_loss_draw_plot.png')
# plt.show()
#
# # Plot testing rewards as before
# # Box plot of testing rewards
# plt.figure(figsize=(8, 5))
# plt.boxplot(testing_rewards, vert=True, patch_artist=True, boxprops=dict(facecolor="green"))
#
# plt.title('Distribution of Testing Rewards')
# plt.xlabel('Testing Trials')
# plt.ylabel('Total Reward')
# plt.grid()
# plt.savefig('testing_rewards_boxplot.png')
# plt.show()
#
# # # random
# # Random actions testing with win/loss/draw tracking
# random_rewards = []  # Store rewards for random actions
# random_results = {"P1 Won": 0, "P1 Lost": 0, "Draw": 0}  # Track win/loss/draw for random actions
# num_random_episodes = 100
#
# print('Testing random actions...')
# for episode in range(num_random_episodes):
#     print(f'Random action episode: {episode + 1}')
#     obs, _ = env.reset()
#     total_reward = 0
#     done = False
#     i = 0
#     while not done:
#         i += 1
#         action = env.action_space.sample()  # Take a random action
#         obs, reward, done, truncated, info = env.step(action)  # Get the result in info
#         total_reward += reward
#         if done:
#             # Check the result and update the count
#             result = info.get('result')
#             if result in random_results:
#                 random_results[result] += 1
#
#     random_rewards.append(total_reward)
#     print(f"Random Episode {episode + 1}: Total Reward = {total_reward}, total actions: {i}")
#
# # Display the random actions results
# print(f"Random Action Win/Loss/Draw: {random_results}")
#
# # Plot win/loss/draw results for random actions
# labels = list(random_results.keys())
# values = list(random_results.values())
#
# plt.figure(figsize=(8, 5))
# plt.bar(labels, values, color=['green', 'red', 'blue'])
# plt.title('Win/Loss/Draw Results (Random Actions)')
# plt.xlabel('Result')
# plt.ylabel('Number of Trials')
# plt.grid(True)
# plt.savefig('random_win_loss_draw_plot.png')
# plt.show()
#
# # Plot comparison of rewards (trained vs random)
# data = [testing_rewards, random_rewards]
# labels = ['Trained Model Rewards', 'Random Action Rewards']
#
# # Box plot of rewards for the trained model and random actions
# plt.figure(figsize=(12, 6))
# plt.boxplot(data, patch_artist=True, labels=labels,
#             boxprops=dict(facecolor="lightgreen", color="black"))
#
# plt.title('Reward Distribution: Trained Model vs Random Actions')
# plt.xlabel('Reward Type')
# plt.ylabel('Total Reward per Trial')
# plt.grid(True)
# plt.savefig('comparison_rewards_boxplot.png')
# plt.show()
