import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from py4j.java_gateway import JavaGateway
import torch as th
import numpy as np
import matplotlib.pyplot as plt
#
from StreetFighter_with_ppo import StreetFighterEnv  # Ensure correct import path
#
# # Define Curriculum Learning Callback (Enhanced to Track Phase Changes)
# from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
#
# class CurriculumLearningCallback(BaseCallback):
#     def __init__(self, env: StreetFighterEnv, patience: int = 100, threshold: float = 50.0, verbose=1):
#         """
#         :param env: The environment instance to adjust difficulty.
#         :param patience: Number of episodes to wait for improvement before increasing difficulty.
#         :param threshold: Average reward threshold to consider as performance improvement.
#         :param verbose: Verbosity level.
#         """
#         super(CurriculumLearningCallback, self).__init__(verbose)
#         self.env = env
#         self.patience = patience
#         self.threshold = threshold
#         self.best_mean_reward = -np.inf
#         self.no_improvement_count = 0
#         self.episode_rewards = []
#         self.phase_change_episodes = []  # List to store episodes where phase changes occur
#
#     def _on_step(self) -> bool:
#         # Check if rewards are available in the `infos` object
#         if "episode" in self.locals["infos"][0]:
#             reward = self.locals["infos"][0]["episode"]["r"]
#             self.episode_rewards.append(reward)
#
#             # Calculate the moving average of the last 'patience' episodes
#             if len(self.episode_rewards) >= self.patience:
#                 mean_reward = np.mean(self.episode_rewards[-self.patience:])
#                 if mean_reward > self.best_mean_reward + self.threshold:
#                     self.best_mean_reward = mean_reward
#                     self.no_improvement_count = 0  # Reset counter on improvement
#                     # Increase difficulty
#                     if self.env.current_phase < len(self.env.curriculum) - 1:
#                         self.env.current_phase += 1
#                         current_episode = len(self.episode_rewards)
#                         self.phase_change_episodes.append(current_episode)  # Record the episode number
#                         print(f"Curriculum Phase Increased to {self.env.current_phase + 1} at Episode {current_episode}")
#                 else:
#                     self.no_improvement_count += 1
#                     print(f"No improvement for {self.no_improvement_count} episodes.")
#
#                 # Stop training if no improvement for a certain number of episodes
#                 if self.no_improvement_count >= self.patience:
#                     if self.verbose:
#                         print("Stopping training due to no improvement.")
#                     return False  # Stop training
#
#         return True  # Continue training
#
# # Initialize the environment
# gateway = JavaGateway()
# game_instance = gateway.entry_point.getGame()
# env = StreetFighterEnv(game=game_instance)
#
# # Check if the environment follows Gym's API
# check_env(env)
#
# # Start the game if necessary
# env.start_game_auto()
#
# # Wrap the environment with Monitor and DummyVecEnv
# env = Monitor(env)
# env = DummyVecEnv([lambda: env])
#
# # Create a separate evaluation environment
# eval_env_instance = StreetFighterEnv(game=game_instance)  # Ensure this is correctly managed
# check_env(eval_env_instance)  # Optional: Validate the evaluation environment
# eval_env_instance.start_game_auto()  # Start the game for evaluation
#
# # Wrap the evaluation environment with Monitor and DummyVecEnv
# eval_env = Monitor(eval_env_instance)
# eval_env = DummyVecEnv([lambda: eval_env])
#
# # Define the Curriculum Learning Callback
# curriculum_callback = CurriculumLearningCallback(env=env.envs[0], patience=100, threshold=50.0, verbose=1)
#
# # Define the EvalCallback without 'save_best_only'
# eval_callback = EvalCallback(
#     eval_env=eval_env,
#     best_model_save_path='./logs/best_model',
#     log_path='./logs/evaluation',
#     eval_freq=50000,          # Evaluate every 50,000 timesteps
#     n_eval_episodes=10,       # Number of episodes per evaluation
#     deterministic=True,       # Use deterministic actions during evaluation
#     render=False,             # Disable rendering during evaluation
#     verbose=1                  # Set verbosity level to 1 for informational logs
# )
#
# # Combine both callbacks into a CallbackList
# callback = CallbackList([curriculum_callback, eval_callback])
#
# # Instantiate the Recurrent PPO model for Player 2
# model = RecurrentPPO(
#     "MlpLstmPolicy",  # Using a recurrent policy
#     env=env,
#     learning_rate=1e-4,
#     n_steps=2048,
#     batch_size=64,
#     n_epochs=10,
#     gamma=0.99,
#     gae_lambda=0.95,
#     clip_range=0.2,
#     ent_coef=0.05,        # Increased entropy for better exploration
#     vf_coef=0.5,
#     max_grad_norm=0.5,
#     verbose=1,            # Set to 1 for informational logs
#     tensorboard_log="./ppo_recurrent_tensorboard/",
# )
#
# # Train the model with both callbacks
# timesteps = 500000
# model.learn(total_timesteps=timesteps, callback=callback, log_interval=10)
#
# # Save the model
# model.save("recurrent_ppo_sf_model")
# print("Model saved as 'recurrent_ppo_sf_model.zip'.")
#
# # Evaluate the model
# from stable_baselines3.common.evaluation import evaluate_policy
#
# mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
# print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
#
# # Plot training rewards with curriculum phase markers
# plt.figure(figsize=(12, 6))
# plt.plot(range(1, len(curriculum_callback.episode_rewards) + 1), curriculum_callback.episode_rewards, marker='o', linestyle='-',
#          color='b', label='Total Reward')
#
# # Add vertical dashed red lines for each curriculum phase change
# for phase_episode in curriculum_callback.phase_change_episodes:
#     plt.axvline(x=phase_episode, color='r', linestyle='--', linewidth=1)
#     plt.text(phase_episode, max(curriculum_callback.episode_rewards)*0.95, f'Phase {env.current_phase + 1}',
#              rotation=90, verticalalignment='top', color='r')
#
# plt.title('Training Rewards per Episode with Curriculum Phases')
# plt.xlabel('Episode')
# plt.ylabel('Total Reward')
# plt.legend(['Rewards', 'Phase Changes'])
# plt.grid(True)
# plt.show()

# Initialize the environment
gateway = JavaGateway()

game_instance = gateway.entry_point.getGame()
env = StreetFighterEnv(game=game_instance)
env.set_current_mixture_ratio(1)
print(env.get_current_mixture_ratio())
env.current_phase = 1
print(env.current_phase)

