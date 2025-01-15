import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from py4j.java_gateway import JavaGateway
from torch import nn
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from StreetFighter_with_ppo import StreetFighterEnv  # Ensure correct import path
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList

# Define a custom LSTM Feature Extractor (Optional)
class CustomLSTMFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super(CustomLSTMFeaturesExtractor, self).__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]

        # Define the LSTM feature extractor
        self.lstm = nn.LSTM(input_dim, 256, batch_first=True)
        self.fc = nn.Linear(256, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        lstm_out, _ = self.lstm(observations.unsqueeze(1))
        lstm_out = lstm_out[:, -1, :]  # Take the last output
        return self.fc(lstm_out)

# Define Curriculum Learning Callback (Enhanced to Track Phase Changes)

class CurriculumLearningCallback(BaseCallback):
    def __init__(self, env: StreetFighterEnv, patience: int = 500, threshold: float = 10.0, verbose=1):
        """
        :param env: The environment instance to adjust difficulty.
        :param patience: Number of episodes to wait for improvement before increasing difficulty.
        :param threshold: Average reward threshold to consider as performance improvement.
        :param verbose: Verbosity level.
        """
        super(CurriculumLearningCallback, self).__init__(verbose)
        self.env = env
        self.patience = patience
        self.threshold = threshold
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.episode_rewards = []
        self.phase_change_episodes = []  # List to store episodes where phase changes occur

    def _on_step(self) -> bool:
        # Check if rewards are available in the `infos` object
        if "episode" in self.locals["infos"][0]:
            reward = self.locals["infos"][0]["episode"]["r"]
            self.episode_rewards.append(reward)

            # Ensure minimum number of episodes before reward evaluation
            if len(self.episode_rewards) >= self.patience:
                mean_reward = np.mean(self.episode_rewards[-self.patience:])

                # Evaluate phase progression only if minimum episodes are met
                # if len(self.episode_rewards) >= self.patience:
                if mean_reward > self.best_mean_reward:# + self.threshold:
                    self.best_mean_reward = mean_reward
                    self.no_improvement_count = 0  # Reset counter on improvement

                    # Ensure minimum phase duration
                    if self.env.current_phase < len(self.env.curriculum) - 1:
                        self.env.current_phase += 1
                        self.env.set_current_mixture_ratio(self.env.current_phase)
                        current_episode = len(self.episode_rewards)
                        self.phase_change_episodes.append(current_episode)
                        print(f"Curriculum Phase Increased to {self.env.current_phase + 1} "
                              f"at Episode {current_episode}, ratio: {self.env.get_current_mixture_ratio()}")

                        # Enforce a minimum phase duration
                        self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                    print(f"No improvement for {self.no_improvement_count} episodes.")

                # Stop training if no improvement for a certain number of episodes
                if self.no_improvement_count >= self.patience:
                    if self.verbose:
                        print("Stopping training due to no improvement.")
                    return True  # Stop training

        return True  # Continue training


# Initialize the environment
gateway = JavaGateway()
game_instance = gateway.entry_point.getGame()
env = StreetFighterEnv(game=game_instance)

# Check if the environment follows Gym's API
check_env(env)

# Start the game if necessary
env.start_game_auto()

# Wrap the environment with Monitor and DummyVecEnv
env = Monitor(env)
env = DummyVecEnv([lambda: env])

# Create a separate evaluation environment
eval_env_instance = StreetFighterEnv(game=game_instance)  # Ensure this is correctly managed
check_env(eval_env_instance)  # Optional: Validate the evaluation environment
eval_env_instance.start_game_auto()  # Start the game for evaluation

# Wrap the evaluation environment with Monitor and DummyVecEnv
eval_env = Monitor(eval_env_instance)
eval_env = DummyVecEnv([lambda: eval_env])

# Define the Curriculum Learning Callback
curriculum_callback = CurriculumLearningCallback(env=env.envs[0], patience=500, threshold=12.0, verbose=1)

eval_callback = EvalCallback(
    eval_env=eval_env,
    best_model_save_path='./logs/best_model',
    log_path='./logs/evaluation',
    eval_freq=10000,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
    verbose=1
)

# Combine both callbacks into a CallbackList
callback = CallbackList([curriculum_callback, eval_callback])

# Define policy kwargs with the custom LSTM feature extractor
policy_kwargs = dict(
    features_extractor_class=CustomLSTMFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=256),
)

model = PPO(
    "MlpPolicy",
    env=env,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.05,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./ppo_sf_lstm_tensorboard/",
)

# Train the model with both callbacks
timesteps = 300000
model.learn(total_timesteps=timesteps, callback=callback, log_interval=10)

# Save the model
model.save("ppo_sf_lstm_v2")
print("Model saved as 'ppo_sf_lstm_v2.zip'.")

# Evaluate the model


mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Plot training rewards with curriculum phase markers
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(curriculum_callback.episode_rewards) + 1),
         curriculum_callback.episode_rewards,
         marker='o', linestyle='-', color='b', label='Total Reward')

# Add vertical dashed red lines for each curriculum phase change
for idx, phase_episode in enumerate(curriculum_callback.phase_change_episodes, start=1):
    plt.axvline(x=phase_episode, color='r', linestyle='--', linewidth=1)
    plt.text(phase_episode, max(curriculum_callback.episode_rewards)*0.95,
             f'Phase {idx}',  # Enumerate phases starting from 1
             rotation=90, verticalalignment='top', color='r')


plt.title('Training Rewards per Episode with Curriculum Phases')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.ylim(-30, 30)
plt.legend(['Rewards', 'Phase Changes'])
plt.grid(True)

plt.savefig('training_rewards_plot_curr.png')
plt.show()
