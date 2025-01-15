# training_with_custom_lstm.py

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from py4j.java_gateway import JavaGateway
import torch as th
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# Import your custom environment
from StreetFighter_with_ppo import StreetFighterEnv  # Ensure correct import path


# Define the Observation Buffer Wrapper
class ObservationBufferWrapper(gym.Wrapper):
    def __init__(self, env, buffer_size=10):
        super(ObservationBufferWrapper, self).__init__(env)
        self.buffer_size = buffer_size
        self.observation_buffer = []

        # Update the observation space to accommodate stacking
        original_obs_shape = env.observation_space.shape
        new_low = np.repeat(env.observation_space.low, self.buffer_size, axis=0)
        new_high = np.repeat(env.observation_space.high, self.buffer_size, axis=0)
        self.observation_space = gym.spaces.Box(low=new_low, high=new_high, dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.observation_buffer = [observation] * self.buffer_size
        stacked_observation = self._get_stacked_observation()
        return stacked_observation, info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        self.observation_buffer.pop(0)
        self.observation_buffer.append(observation)
        stacked_observation = self._get_stacked_observation()
        return stacked_observation, reward, done, truncated, info

    def _get_stacked_observation(self):
        return np.concatenate(self.observation_buffer, axis=0)


# Define the Custom LSTM Feature Extractor
class CustomLSTMFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256, buffer_size: int = 10):
        """
        Initializes the Custom LSTM Features Extractor.

        Args:
            observation_space (gym.Space): The observation space of the environment.
            features_dim (int): The dimension of the output features.
            buffer_size (int): The number of past observations to include in the sequence.
        """
        # Calculate the input dimension per time step
        input_dim = observation_space.shape[0] // buffer_size  # Assuming stacking along features

        super(CustomLSTMFeaturesExtractor, self).__init__(observation_space, features_dim)

        # Store buffer_size as an instance attribute
        self.buffer_size = buffer_size

        # Define the LSTM feature extractor
        self.lstm = nn.LSTM(input_dim, 256, batch_first=True)
        self.fc = nn.Linear(256, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Processes the input observations through the LSTM and fully connected layers.

        Args:
            observations (th.Tensor): The input observations with shape [batch_size, features_dim * buffer_size].

        Returns:
            th.Tensor: The extracted features with shape [batch_size, features_dim].
        """
        batch_size = observations.size(0)
        sequence_length = self.buffer_size
        input_dim = observations.size(1) // sequence_length

        # Reshape observations to [batch_size, sequence_length, input_dim]
        observations = observations.view(batch_size, sequence_length, input_dim)

        # Pass through LSTM
        lstm_out, _ = self.lstm(observations)  # lstm_out: [batch_size, sequence_length, hidden_size]

        # Take the last output of the LSTM
        lstm_out = lstm_out[:, -1, :]  # [batch_size, hidden_size]

        # Pass through the fully connected layer
        features = self.fc(lstm_out)  # [batch_size, features_dim]

        return features


# Define Curriculum Learning Callback
class CurriculumLearningCallback(BaseCallback):
    def __init__(self, env: StreetFighterEnv, patience: int = 500, threshold: float = 10.0, verbose=1):
        super(CurriculumLearningCallback, self).__init__(verbose)
        self.env = env
        self.patience = patience
        self.threshold = threshold
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.episode_rewards = []
        self.phase_change_episodes = []

    def _on_step(self) -> bool:
        # Check if any episodes have been completed
        if "episode" in self.locals["infos"][0]:
            reward = self.locals["infos"][0]["episode"]["r"]
            self.episode_rewards.append(reward)

            # Ensure minimum number of episodes before evaluation
            if len(self.episode_rewards) >= self.patience:
                mean_reward = np.mean(self.episode_rewards[-self.patience:])

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.no_improvement_count = 0

                    if self.env.current_phase < len(self.env.curriculum) - 1:
                        self.env.current_phase += 1
                        self.env.set_current_mixture_ratio(self.env.current_phase)
                        current_episode = len(self.episode_rewards)
                        self.phase_change_episodes.append(current_episode)
                        if self.verbose:
                            print(f"Curriculum Phase Increased to {self.env.current_phase + 1} "
                                  f"at Episode {current_episode}, ratio: {self.env.get_current_mixture_ratio()}")

                else:
                    self.no_improvement_count += 1
                    if self.verbose:
                        print(f"No improvement for {self.no_improvement_count} episodes.")

                # Stop training if no improvement
                if self.no_improvement_count >= self.patience:
                    if self.verbose:
                        print("Stopping training due to no improvement.")
                    return True  # Signal to stop training
        return True  # Continue training


# Main Training Function
def main():
    # Parameters
    buffer_size = 10  # Number of past observations to include

    # Initialize the Java Gateway and game instance
    gateway = JavaGateway()
    game_instance = gateway.entry_point.getGame()

    # Initialize the custom StreetFighter environment
    env = StreetFighterEnv(game=game_instance)

    # Wrap the environment with ObservationBufferWrapper
    env = ObservationBufferWrapper(env, buffer_size=buffer_size)

    # Check if the environment follows Gym's API
    check_env(env)

    # Start the game if necessary
    env.start_game_auto()

    # Wrap the environment with Monitor and DummyVecEnv
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # Create evaluation environment
    eval_env_instance = StreetFighterEnv(game=game_instance)
    eval_env_instance = ObservationBufferWrapper(eval_env_instance, buffer_size=buffer_size)
    check_env(eval_env_instance)
    eval_env_instance.start_game_auto()
    eval_env = Monitor(eval_env_instance)
    eval_env = DummyVecEnv([lambda: eval_env])

    # Define policy_kwargs with the custom LSTM feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomLSTMFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256, buffer_size=buffer_size),
    )

    # Initialize the PPO model with the custom feature extractor
    model = PPO(
        "MlpPolicy",
        env,
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

    # Define evaluation callback
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

    # Define curriculum learning callback
    curriculum_callback = CurriculumLearningCallback(env=env.envs[0], patience=500, threshold=12.0, verbose=1)

    # Combine callbacks
    callback = CallbackList([curriculum_callback, eval_callback])

    # Train the model with both callbacks
    timesteps = 150000
    model.learn(total_timesteps=timesteps, callback=callback, log_interval=10)

    # Save the model
    model.save("ppo_sf_lstm_v3")
    print("Model saved as 'ppo_sf_lstm_v3.zip'.")

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
        plt.text(phase_episode, max(curriculum_callback.episode_rewards) * 0.95,
                 f'Phase {idx}',
                 rotation=90, verticalalignment='top', color='r')

    plt.title('Training Rewards per Episode with Curriculum Phases')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.ylim(-30, 30)
    plt.legend(['Rewards', 'Phase Changes'])
    plt.grid(True)

    plt.savefig('training_rewards_plot_curr.png')
    plt.show()


main()
