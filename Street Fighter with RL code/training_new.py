import matplotlib.pyplot as plt
from gymnasium.envs.registration import register
import warnings
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from py4j.java_gateway import JavaGateway
from StreetFighter_with_ppo import StreetFighterEnv  # Your environment import
warnings.filterwarnings("ignore")

# Custom Callback to log rewards
class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []  # List to store rewards per episode
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        # Accumulate rewards
        self.current_episode_reward += self.locals['rewards'][0]

        # If episode ends, log the reward and reset counter
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0  # Reset for next episode

        return True

    def get_rewards(self):
        return self.episode_rewards

# Register the environment
register(
    id='StreetFighter_tk1998-v0',
    entry_point='street_fighter_env:StreetFighterEnv',
)

# Setup the game and environment
gateway = JavaGateway()
game_instance = gateway.entry_point.getGame()
env = StreetFighterEnv(game=game_instance)
print('Environment created.')

# Ensure the environment is valid
check_env(env)

# Initialize the DQN model
# model = DQN(
#     "MlpPolicy",
#     env,
#     learning_rate=0.0005,  # DQN often needs a slightly higher learning rate
#     buffer_size=50000,     # Replay buffer size
#     learning_starts=1000,  # Start learning after 1000 steps
#     batch_size=32,         # Batch size for each learning step
#     gamma=0.99,            # Discount factor for reward
#     tau=0.05,              # Soft update coefficient for target network
#     train_freq=4,          # Update frequency
#     target_update_interval=500,  # Update target network every 500 steps
#     verbose=1
# )

# model = DQN(
#     "MlpPolicy",
#     env,
#     verbose=1,
#     gamma=0.98,
#     learning_rate=5e-4,
#     batch_size=32,
#     buffer_size=15000,
#     exploration_fraction=0.1,         # Adjusted exploration fraction for dynamic decay
#     exploration_final_eps=0.099,
#     target_update_interval=250,
#     learning_starts=1000,
# )

model = DQN.load("dqn_street_fighter", env=env)  # Load existing model and set the current environment

# Start the game
env.start_game_auto()

# Initialize the custom callback for logging rewards
reward_callback = RewardLoggingCallback()


num_timesteps = 100000  # Adjust as needed for further training
print('Additional Training started...')

# Continue training the loaded DQN model and log rewards
model.learn(total_timesteps=num_timesteps, callback=reward_callback)

print("Additional training finished.")

# Save the updated model
model.save("dqn_street_fighter_updated")


# Retrieve updated rewards logged during additional training
training_rewards = reward_callback.get_rewards()

# Plot training rewards per episode
plt.figure(figsize=(12, 6))
plt.plot(training_rewards, label='Total Reward (Additional Training)', marker='o', color='blue')
plt.title('Training Rewards per Episode with DQN (20% Random Actions)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid()
plt.legend()
plt.savefig('dqn_training_rewards_plot_against_20_random.png')
plt.show()

#
# # Set number of training timesteps
# num_timesteps = 100000  # Adjust as needed
# print('Training started...')
#
# # Train the DQN model and log rewards using the callback
# model.learn(total_timesteps=num_timesteps, callback=reward_callback)
#
# print("Training finished.")
#
# # Save the model
# model.save("dqn_street_fighter")
#
# # Retrieve rewards logged during training
# training_rewards = reward_callback.get_rewards()
#
# # Plot training rewards per episode
# plt.figure(figsize=(12, 6))
# plt.plot(training_rewards, label='Total Reward (Training)', marker='o', color='blue')
# plt.title('Training Rewards per Episode with DQN')
# plt.xlabel('Episode')
# plt.ylabel('Total Reward')
# plt.grid()
# plt.legend()
# plt.savefig('dqn_training_rewards_plot_against_ppo.png')
# plt.show()
