import time
import matplotlib.pyplot as plt
import numpy as np
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
print('Environment created for testing.')

# Load the models
model = PPO.load("ppo_street_fighter_revised.zip")
env.start_game_auto()

# Initialize data tracking
id = 'az2'
num_test_episodes = 10
testing_rewards = []
old_model_rewards = []
random_rewards = []
results = {"P1 Won": 0, "P1 Lost": 0, "Draw": 0}

# Test the current model
print('Testing current model...')
for episode in range(num_test_episodes):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        if np.random.rand() < 0.2:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        total_reward += reward
        if done:
            result = info.get('result')
            if result in results:
                results[result] += 1
    testing_rewards.append(total_reward)

# Plotting Results
# Bar graph for win/loss/draw
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color=['green', 'red', 'blue'], alpha=0.7)
plt.title("Win/Loss/Draw Results for Current Model")
plt.ylabel("Number of Episodes")
plt.xlabel("Result")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f'winLoss_subject_{id}.png')
plt.show()

# Box plot for rewards
# Box plot for rewards (vertical orientation)
plt.figure(figsize=(10, 6))
plt.boxplot(testing_rewards, vert=True, patch_artist=True,
            boxprops=dict(facecolor='orange', color='black'),
            medianprops=dict(color='black'))
plt.title("Box Plot of Rewards for Current Model")
plt.ylabel("Rewards")

plt.savefig(f'rew_subject_{id}.png')
plt.show()

# Print Results and Summary
print("Win/Loss/Draw Results:", results)
print("Reward Summary:")
print(f"  Mean Reward: {np.mean(testing_rewards):.2f}")
print(f"  Max Reward: {np.max(testing_rewards):.2f}")
print(f"  Min Reward: {np.min(testing_rewards):.2f}")
