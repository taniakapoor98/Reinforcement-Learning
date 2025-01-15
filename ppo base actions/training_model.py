import matplotlib.pyplot as plt
from gymnasium.envs.registration import register
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from py4j.java_gateway import JavaGateway
from StreetFighter_tk import StreetFighterEnv
warnings.filterwarnings("ignore")

# Custom Callback to log rewards
class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []  # List to store rewards per episode
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        # Increment the reward for the current episode
        self.current_episode_reward += self.locals['rewards'][0]

        # Check if the episode is done
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
# game_instance.setRenderingEnabled(True)
env = StreetFighterEnv(game=game_instance)
print('Environment created.')

# Training
check_env(env)  # Ensure the environment is valid
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    gamma=0.99,
    ent_coef=0.01,
    verbose=2
)
env.start_game_auto()

# Initialize the custom callback for logging rewards
reward_callback = RewardLoggingCallback()

# Total number of timesteps for training
num_timesteps = 100000  # You can increase this as needed
print('Training started...')

# Train the model and log rewards using the callback
model.learn(total_timesteps=num_timesteps, callback=reward_callback)

print("Training finished.")

# Save the model
model.save("ppo_street_fighter")

# Get the rewards logged during training
training_rewards = reward_callback.get_rewards()

# Plotting training results (rewards per episode)
plt.figure(figsize=(12, 6))
plt.plot(training_rewards, label='Total Reward (Training)', marker='o', color='blue')
plt.title('Training Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid()
plt.legend()
plt.savefig('training_rewards_plot.png')
plt.show()
