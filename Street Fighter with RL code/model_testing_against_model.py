# test.py
import time
import matplotlib.pyplot as plt
import pandas as pd
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO, DQN
from py4j.java_gateway import JavaGateway
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import RecurrentPPO
import seaborn as sns
from StreetFighter_with_ppo import StreetFighterEnv  # Import your environment class

# Setup the game and environment
gateway = JavaGateway()
game_instance = gateway.entry_point.getGame()
game_instance.setRenderingEnabled(True)
env = StreetFighterEnv(game=game_instance)
check_env(env)
print('Environment created for testing.')


# Load the models
model = PPO.load("best_model_ppo_lstm.zip",)

# model = DQN.load("dqn_sf_lstm_v2.zip",)
model_old = DQN.load("dqn_sf_lstm_v2.zip")  # Load your DQN model
env.start_game_auto()

# Initialize data tracking
num_test_episodes = 100
# Elo Rating Initialization
elo_ratings = {
    "PPO+LSTM": 1000,
    "DQN+LSTM": 1000,
    "Random": 1000
}
K_FACTOR = 32  # K-factor for Elo ratings
elo_progression = {"PPO+LSTM": [], "DQN+LSTM": [], "Random": []}

# Data structures to store metrics
testing_rewards = []
old_model_rewards = []
random_rewards = []

results = {"P1 Won": 0, "P1 Lost": 0, "Draw": 0}
old_model_results = {"P1 Won": 0, "P1 Lost": 0, "Draw": 0}
random_results = {"P1 Won": 0, "P1 Lost": 0, "Draw": 0}

# Additional metrics
time_taken_current_model = []  # Steps taken to win
action_efficiency_current_model = []  # Ratio of attack actions to total actions

time_taken_old_model = []
action_efficiency_old_model = []

time_taken_random = []
action_efficiency_random = []

# Define attack actions
attack_actions = [2, 3, 4]  # Light, Medium, Heavy Attack


# Elo rating functions
def calculate_expected_outcome(rating_A, rating_B):
    return 1 / (1 + 10 ** ((rating_B - rating_A) / 400))


def update_elo(rating_A, rating_B, score_A, score_B):
    expected_A = calculate_expected_outcome(rating_A, rating_B)
    expected_B = calculate_expected_outcome(rating_B, rating_A)

    new_rating_A = rating_A + K_FACTOR * (score_A - expected_A)
    new_rating_B = rating_B + K_FACTOR * (score_B - expected_B)

    return new_rating_A, new_rating_B


# Function to test a model
attack_frequency_current_model = []
attack_frequency_old_model = []
attack_frequency_random = []

# Update the test_model function to append attack counts
def test_model(model, env, num_episodes, results_dict, rewards_list, time_taken_list, action_efficiency_list,
               attack_frequency_list, elo_name, model_name="Model"):
    print(f'Testing {model_name}...')
    for episode in range(num_test_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        attack_count = 0

        while not done:
            if model_name == "Random Actions":
                action = env.action_space.sample()  # Take a random action
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)

            # Count attack actions
            if action in attack_actions:
                attack_count += 1

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if done:
                result = info.get('result')
                if result in results_dict:
                    results_dict[result] += 1

                # Append default values for consistency
                time_taken_list.append(steps)  # Append even if no win/loss condition
                efficiency = attack_count / steps if steps > 0 else 0
                action_efficiency_list.append(efficiency)

                # Update Elo Ratings
                if result == "P1 Won":
                    score_A, score_B = 1, 0
                elif result == "P1 Lost":
                    score_A, score_B = 0, 1
                else:  # Draw or undefined
                    score_A, score_B = 0.5, 0.5

                # Update Elo ratings only for the tested player and opponent
                if model_name != "Random Actions":
                    elo_ratings[elo_name], elo_ratings["Random"] = update_elo(
                        elo_ratings[elo_name], elo_ratings["Random"], score_A, score_B
                    )
                    elo_progression[elo_name].append(elo_ratings[elo_name])
                else:
                    elo_ratings["Random"], elo_ratings[elo_name] = update_elo(
                        elo_ratings["Random"], elo_ratings[elo_name], score_A, score_B
                    )
                    elo_progression["Random"].append(elo_ratings["Random"])

                break  # Exit loop after episode is done

        rewards_list.append(total_reward)
        attack_frequency_list.append(attack_count)
        if (episode + 1) % 10 == 0:
            print(f"Completed Episode {episode + 1}/{num_test_episodes}")

# Test the current PPO+LSTM model
test_model(
    model=model,
    env=env,
    num_episodes=num_test_episodes,
    results_dict=results,
    rewards_list=testing_rewards,
    time_taken_list=time_taken_current_model,
    action_efficiency_list=action_efficiency_current_model,
    attack_frequency_list=attack_frequency_current_model,
    elo_name="PPO+LSTM",
    model_name="Current PPO+LSTM Model"
)

# Test the old DQN+LSTM model
test_model(
    model=model_old,
    env=env,
    num_episodes=num_test_episodes,
    results_dict=old_model_results,
    rewards_list=old_model_rewards,
    time_taken_list=time_taken_old_model,
    action_efficiency_list=action_efficiency_old_model,
    attack_frequency_list=attack_frequency_old_model,
    elo_name="DQN+LSTM",
    model_name="Old DQN+LSTM Model"
)

# Test random actions
test_model(
    model=None,  # Random actions do not require a model
    env=env,
    num_episodes=num_test_episodes,
    results_dict=random_results,
    rewards_list=random_rewards,
    time_taken_list=time_taken_random,
    action_efficiency_list=action_efficiency_random,
    attack_frequency_list=attack_frequency_random,
    elo_name="Random",
    model_name="Random Actions"
)

# Display Elo ratings
print("\nFinal Elo Ratings:")
for model, rating in elo_ratings.items():
    print(f"{model}: {rating}")

# Plot Elo rating progression
plt.figure(figsize=(10, 6))
for model, ratings in elo_progression.items():
    plt.plot(range(1, len(ratings) + 1), ratings, label=model)

plt.title("Elo Ratings Progression")
plt.xlabel("Match Number")
plt.ylabel("Elo Rating")
plt.legend()
plt.grid(True)
plt.savefig("elo_ratings_progression_fixed.png")
plt.show()




# Display win/loss/draw results for all tests
print(f"\nWin/Loss/Draw for Current PPO+LSTM Model: {results}")
print(f"Win/Loss/Draw for Old DQN+LSTM Model: {old_model_results}")
print(f"Win/Loss/Draw for Random Actions: {random_results}")

# Plot Win/Loss/Draw results for all comparisons
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Current Model Win/Loss/Draw
axes[0].bar(results.keys(), results.values(), color=['green', 'red', 'blue'])
axes[0].set_title('Win/Loss/Draw (Current PPO+LSTM Model)')
axes[0].set_xlabel('Result')
axes[0].set_ylabel('Number of Episodes')
axes[0].grid(True)

# Old Model Win/Loss/Draw
axes[1].bar(old_model_results.keys(), old_model_results.values(), color=['green', 'red', 'blue'])
axes[1].set_title('Win/Loss/Draw (Old DQN+LSTM Model)')
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
plt.savefig('comparison_win_loss_draw_modelVsmodel.png')
plt.show()

# Box Plot Comparison of Rewards for all three methods
data_rewards = [testing_rewards, old_model_rewards, random_rewards]
labels_rewards = ['PPO+LSTM Model', 'DQN+LSTM Model', 'Random Actions']

plt.figure(figsize=(12, 6))
plt.boxplot(data_rewards, patch_artist=True, labels=labels_rewards,
            boxprops=dict(facecolor="lightgreen", color="black"),
            medianprops=dict(color="orange"))
plt.title('Reward Distribution: PPO+LSTM vs DQN+LSTM vs Random Actions')
plt.xlabel('Model')
plt.ylabel('Total Reward per Episode')
plt.grid(True)
plt.savefig('comparison_rewards_boxplot_modelVsmodel.png')
plt.show()

# Box Plot Comparison of Time Taken to Win for PPO+LSTM and DQN+LSTM
# Exclude Random Actions as they may not consistently result in wins
data_time = [time_taken_current_model, time_taken_old_model]
labels_time = ['PPO+LSTM Model', 'DQN+LSTM Model']

plt.figure(figsize=(12, 6))
plt.boxplot(data_time, patch_artist=True, labels=labels_time,
            boxprops=dict(facecolor="lightblue", color="black"),
            medianprops=dict(color="orange"))
plt.title('Time Taken to Win: PPO+LSTM vs DQN+LSTM Models')
plt.xlabel('Model')
plt.ylabel('Number of Steps to Win')
plt.grid(True)
plt.savefig('comparison_time_taken_modelVsmodel.png')
plt.show()

# Box Plot Comparison of Action Efficiency for all three methods
data_efficiency = [action_efficiency_current_model, action_efficiency_old_model, action_efficiency_random]
labels_efficiency = ['PPO+LSTM Model', 'DQN+LSTM Model', 'Random Actions']

plt.figure(figsize=(12, 6))
plt.boxplot(data_efficiency, patch_artist=True, labels=labels_efficiency,
            boxprops=dict(facecolor="lightcoral", color="black"),
            medianprops=dict(color="orange"))
plt.title('Action Efficiency: PPO+LSTM vs DQN+LSTM vs Random Actions')
plt.xlabel('Model')
plt.ylabel('Action Efficiency (Attack Actions / Total Actions)')
plt.grid(True)
plt.savefig('comparison_action_efficiency_modelVsmodel.png')
plt.show()

# Corrected action_reward_data
# Corrected action_reward_data
action_reward_data = {
    'Action Frequency': attack_frequency_current_model + attack_frequency_old_model + attack_frequency_random,
    'Reward': testing_rewards + old_model_rewards + random_rewards,
    'Model': (['PPO+LSTM'] * len(attack_frequency_current_model) +
              ['DQN+LSTM'] * len(attack_frequency_old_model) +
              ['Random'] * len(attack_frequency_random))
}

# Ensure all lists are of the same length
assert len(action_reward_data['Action Frequency']) == len(action_reward_data['Reward'])
assert len(action_reward_data['Reward']) == len(action_reward_data['Model'])

# Create a DataFrame from the action-reward data
action_reward_df = pd.DataFrame(action_reward_data)

# Create a pivot table for the heatmap
pivot_table = action_reward_df.pivot_table(
    values='Reward',
    index='Model',
    columns='Action Frequency',
    aggfunc='mean',
)

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, cmap='coolwarm', annot=True, fmt='.2f', cbar_kws={'label': 'Average Reward'})
plt.title('Heatmap of Action Frequency vs Reward')
plt.xlabel('Action Frequency')
plt.ylabel('Model')
plt.savefig('heatmap_action_frequency_vs_reward_fixed.png')
plt.show()

# Prepare data for Violin plot
time_data = {
    'Model': ['PPO+LSTM'] * len(time_taken_current_model) + ['DQN+LSTM'] * len(time_taken_old_model) + ['Random'] * len(time_taken_random),
    'Steps': time_taken_current_model + time_taken_old_model + time_taken_random
}

time_df = pd.DataFrame(time_data)

# Plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=time_df, x='Model', y='Steps', inner="quartile", palette='muted')
plt.title('Episode Length Distribution: PPO+LSTM vs DQN+LSTM vs Random Actions')
plt.xlabel('Model')
plt.ylabel('Number of Steps')
plt.grid(True)
plt.savefig('episode_length_violin_modelVsmodel.png')
plt.show()
