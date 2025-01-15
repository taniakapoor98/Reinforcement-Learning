import time
import matplotlib.pyplot as plt
from gymnasium import spaces
import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
import warnings
warnings.filterwarnings("ignore")
from py4j.java_gateway import JavaGateway
from stable_baselines3 import PPO  # Import PPO algorithm
from stable_baselines3.common.env_checker import check_env
register(
    id='StreetFighter_tk1998-v0',  # Unique ID for the environment
    entry_point='StreetFighter_tk:StreetFighterEnv',  # Adjust the entry point as necessary
)
class KeyEvent:
    VK_ENTER = 10
    VK_UP = 38
    VK_DOWN = 40
    VK_ESCAPE = 27
    VK_1 = 49
    VK_2 = 50
class StreetFighterEnv(gym.Env):
    def __init__(self,game=None):
        # super(StreetFighterEnv, self).__init__()
        self.action_space = spaces.Discrete(6)  # left right down attacks  # 5  * 4 change to multibinary later
        self.isFirstRun = True
        if game is None:
            self.gateway = JavaGateway()
            self.game = self.gateway.entry_point.getGame()
            print(f'Game instance obtained: {self.game}')  # Debugging line
        else:
            self.game = game

        if self.game is None:
            raise Exception("Failed to connect to the game.")  # Raise an exception if game is None

        self.cumulative_reward = 0
        # Observation space-- pixel data of state frame
        # self.observation_space = spaces.Box(low=0, high=255, shape=(100, 100, 1), dtype=np.uint8)
        self.observation_space = spaces.Box(
            low=np.array([
                0,  # Player 1 position x (min value)
                -600,  # Player 1 velocity x (min value)
                0,  # Player 1 health (min)
                0,  # Player 2 position x (min)
                -600,  # Player 2 velocity x (min)
                0,  # Player 2 health (min)
                0,  # Distance between players
                0, 0,  # Player 1 and Player 2 states (e.g., idle, attacking)
                0,  # Player 1 attack states: light, medium, heavy
                0 # Player 2 attack states: light, medium, heavy
            ]),
            high=np.array([
                1400,  # Player 1 position x (max)
                600,  # Player 1 velocity x (max)
                100,  # Player 1 health (max)
                1400,  # Player 2 position x (max)
                600,  # Player 2 velocity x (max)
                100,  # Player 2 health (max)
                1800,  # Distance between players
                1, 1,  # State encoding ( 0 = idle, 1 = attacking)
                4,  # Player 1 attack states: light, medium, heavy (binary), crouched
                4 # Player 2 attack states: light, medium, heavy (binary), crouched
            ]),
            dtype=np.float32
        )

        self.prev_p1_health = 100
        self.prev_p2_health = 100

    def simulate_key_press(self, key_code):
        # Call the wrapper function in Java to handle the key press
        self.game.simulateKeyPress(key_code)

    def start_game_auto(self):
        self.simulate_key_press(KeyEvent.VK_ENTER)
        # time.sleep(1)
        self.simulate_key_press(KeyEvent.VK_ENTER)
        # time.sleep(1)

        # # Navigate the menu
        # self.simulate_key_press(KeyEvent.VK_DOWN)
        # time.sleep(0.5)
        # self.simulate_key_press(KeyEvent.VK_ENTER)
        # time.sleep(1)

        # Character selection
        self.simulate_key_press(KeyEvent.VK_1)
        time.sleep(0.5)
        self.simulate_key_press(KeyEvent.VK_2)
        time.sleep(0.5)
    # def step(self, action):
    #     self._handle_action(action)
    #     # time.sleep(0.5)
    #     obs = self._get_next_frame()
    #     reward = self._compute_reward()
    #     done = self._check_done()
    #     self.cumulative_reward += reward
    #
    #     return obs, reward, done, {},{}

    def step(self, action):
        self._handle_action(action)
        # Request Java to advance the game state
        self.game.requestAdvance()

        # After requesting, you can proceed to get the new state
        obs = self._get_next_frame()
        reward = self._compute_reward()
        done, result = self._check_done()
        truncated = False

        self.cumulative_reward += reward
        info = {"result": result} if result is not None else {}

        # Return observation, reward, done, truncated, and info (empty dict)
        return obs, reward, done, truncated, info


    def _handle_action(self, action):
        player1_controls = self.game.getPlayer1Controls()  # Retrieve the control array once
        action = int(action)
        actions = {
            0: player1_controls[2],  # Left
            1: player1_controls[3],  # Right
            2: player1_controls[4],  # Light attack
            3: player1_controls[5],  # Medium attack
            4: player1_controls[6],  # Heavy attack
            5: player1_controls[1]   # crouch (down)
        }

        if actions[action] is not None:
            self.game.player1Move(actions[action])



    def _get_next_frame(self):
        combined_observation = self.game.getPlayer1().getObs()
        obs = np.array(combined_observation, dtype=np.float32)
        return obs

    def _compute_reward(self, max_health=100):
        # Get current health values for both players
        p1_health = self.game.getPlayer1().getHealth()
        p2_health = self.game.getPlayer2().getHealth()

        # Calculate the health difference since the last frame
        p2_health_diff = self.prev_p2_health - p2_health  # Positive if player 2 lost health
        p1_health_diff = self.prev_p1_health - p1_health  # Positive if player 1 lost health

        # Normalize the health differences by the maximum health
        p2_health_diff_normalized = p2_health_diff / max_health
        p1_health_diff_normalized = p1_health_diff / max_health

        # Reward calculation with normalized values
        reward = 0
        reward += p2_health_diff_normalized  # Reward for reducing player 2's health
        reward -= p1_health_diff_normalized  # Penalty for player 1's health reduction

        # Update previous health values for next calculation
        self.prev_p1_health = p1_health
        self.prev_p2_health = p2_health

        return reward

    # def _check_done(self):
    #     # Done if player 1 or player 2 health reaches zero
    #     if self.game.getPlayer1().getHealth() <= 0 or self.game.getPlayer2().getHealth() <= 0:
    #         self.game.getPlayer1().setHealth(100)
    #         self.game.getPlayer2().setHealth(100)
    #
    #         # print('health')
    #         self.game.setP2Wins(0)
    #         self.game.setP1Wins(0)
    #         return True
    #
    #     # if time runs out
    #     elif self.game.getTimeLeft() <= 5:
    #         # print('time')
    #         return True
    #
    #     elif self.game.getP1Wins() > 0:
    #         # self.game.setP2Wins(0)
    #         self.game.setP1Wins(0)
    #         # print('win')
    #         print(self.game.getP1Wins())
    #
    #         return True
    #
    #     return False

    def _check_done(self):
        p1_health = self.game.getPlayer1().getHealth()
        p2_health = self.game.getPlayer2().getHealth()
        p1_wins = self.game.getP1Wins()
        p2_wins = self.game.getP2Wins()

        result = None  # Track win/loss/draw

        # Check if either player has zero health (win/loss condition)
        if p1_health <= 0 and p2_health > 0:
            result = "P1 Lost"  # Player 1 lost
        elif p2_health <= 0 and p1_health > 0:
            result = "P1 Won"  # Player 1 won
        elif p2_health == 0 and p1_health == 0:
            result = "Draw"

        # Check if time runs out
        elif self.game.getTimeLeft() <= 2:
            if p1_health > p2_health:
                result = "P1 Won"  # Player 1 won on health points when time ran out
            elif p2_health > p1_health:
                result = "P1 Lost"  # Player 1 lost on health points when time ran out
            else:
                result = "Draw"  # Draw because health is equal or time ran out

        # Check if P1 won via game win counter
        elif p1_wins > 0:
            result = "P1 Won"  # Player 1 won via game win counter

        # Check if P2 won via game win counter
        elif p2_wins > 0:
            result = "P1 Lost"  # Player 1 lost via game win counter




        # If a result has been determined, reset health and wins, then return `done` as True
        if result is not None:
            # Reset health and game state for the next episode
            self.game.getPlayer1().setHealth(100)
            self.game.getPlayer2().setHealth(100)
            self.game.setP2Wins(0)
            self.game.setP1Wins(0)

            return True, result  # Game is done, return the result (win/loss/draw)

        return False, None  # Game is not done, continue

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        # Reset the environment to its initial state
        # game.resetGame()
        # reward = 0
        # get frame

        self.game.resetGame()
        self.game.setP2Wins(0)
        self.game.setP1Wins(0)
        self.cumulative_reward = 0
        self.prev_p1_health = self.game.getPlayer1().getHealth()
        self.prev_p2_health = self.game.getPlayer2().getHealth()
        obs = self._get_next_frame()

        return obs, {}

    def render(self):
        pass
# gateway = JavaGateway()
# game_instance = gateway.entry_point.getGame()
# game_instance.setRenderingEnabled(True)
# env = StreetFighterEnv(game=game_instance)
# print('Environment created.')
# # Number of episodes to play
# num_episodes = 2
# env.start_game_auto()
# #
# # game_instance.requestAdvance()
#
# for episode in range(num_episodes):
#     print(f'for episode: {episode+1}')
#     obs = env.reset()
#     total_reward = 0
#     done = False
#     i = 0
#     while not done:
#         i += 1
#         action = env.action_space.sample()  # Sample a random action
#         # print(f'for action: {action}')
#
#         obs, reward, done, info,_= env.step(action)
#         total_reward += reward
#
#     # Print the cumulative reward for the episode
#     print(f"Episode {episode + 1}: Total Reward = {total_reward}, total actions: {i}")
#
# # Training...................................................
# check_env(env)
#
# # Instantiate the PPO model
# model = PPO(
#     "MlpPolicy",
#     env,
#     learning_rate=0.0003,
#     gamma=0.99,
#     # n_steps=2048,
#     # batch_size=64,
#     # n_epochs=10,
#     ent_coef=0.01,
#     verbose=2
# )
# env.start_game_auto()
# training_rewards = []
# #
# # print('Training started...')
# # num_timesteps = 2000  # Total training timesteps
# # for _ in range(num_timesteps // 100):
# #     obs, _ = env.reset()
# #     total_reward = 0
# #     done = False
# #     while not done:
# #         action, _ = model.predict(obs)
# #         action = int(action)
# #         obs, reward, done,info, _ = env.step(action)
# #         total_reward += reward
# #
# #     training_rewards.append(total_reward)
# #
# # print("Training finished.")
# #
# # model.save("ppo_street_fighter")
# #
# # # Plotting training results
# # plt.figure(figsize=(12, 6))
# # plt.plot(training_rewards, label='Total Reward (Training)', marker='o', color='blue')
# # plt.title('Training Rewards per Episode')
# # plt.xlabel('Episode')
# # plt.ylabel('Total Reward')
# # plt.grid()
# # plt.legend()
# # # plt.yticks(np.arange(1, 50, 1))
# # plt.savefig('training_rewards_plot.png')
# # plt.show()
#
# # # Testing..................
# # model = PPO.load("ppo_street_fighter")  # Load the trained model
# #
# # # Now let's play using the trained model
# # num_test_episodes = 10  # Number of episodes for testing
# # testing_rewards = []
# #
# # print('Testing started...')
# # for episode in range(num_test_episodes):
# #     obs,_ = env.reset()  # Ensure this returns observation and info
# #     total_reward = 0
# #     done = False
# #     while not done:
# #         action, _ = model.predict(obs, deterministic=True)
# #         obs, reward, done, info, _ = env.step(int(action))  # Ensure action is an integer
# #         total_reward += reward
# #     testing_rewards.append(total_reward)
# #     print(f"Test Episode {episode + 1}: Total Reward = {total_reward}")
# #
# # plt.figure(figsize=(8, 5))
# # plt.plot(testing_rewards, label='Total Reward (Testing)', marker='o', color='green')
# # plt.title('Testing Rewards per Episode')
# # plt.xlabel('Episode')
# # plt.ylabel('Total Reward')
# # plt.grid()
# # plt.legend()
# # plt.yticks(np.arange(1, 21, 1))
# # plt.savefig('testing_rewards_plot.png')
# # plt.show()

# gateway = JavaGateway()
# game_instance = gateway.entry_point.getGame()
# # game_instance.setRenderingEnabled(True)
# env = StreetFighterEnv(game=game_instance)
# print(env.observation_space)