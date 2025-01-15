import time
import numpy as np
from gymnasium import spaces
import gymnasium as gym
from gymnasium.envs.registration import register
import warnings
from py4j.java_gateway import JavaGateway
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import torch as th
from torch import nn

warnings.filterwarnings("ignore")

# Register the environment
register(
    id='StreetFighter_tk1998-v0',
    entry_point='StreetFighter_tk:StreetFighterEnv',
)

class KeyEvent:
    VK_ENTER = 10
    VK_UP = 38
    VK_DOWN = 40
    VK_ESCAPE = 27
    VK_1 = 49
    VK_2 = 50
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

class StreetFighterEnv(gym.Env):
    def __init__(self, game=None):
        super(StreetFighterEnv, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(
            6)  # Actions: 0=Left, 1=Right, 2=Light Attack, 3=Medium Attack, 4=Heavy Attack, 5=Crouch

        # Initialize PPO model for opponent (Player 1)
        self.ppo_old = PPO.load("ppo_street_fighter_revised")
        self.p_random = 0.8
        # Initialize game connection via Py4j
        if game is None:
            self.gateway = JavaGateway()
            self.game = self.gateway.entry_point.getGame()
        else:
            self.game = game

        if self.game is None:
            raise Exception("Failed to connect to the game.")

        # Initialize reward tracking variables
        self.cumulative_reward = 0
        self.hits_taken = 0  # Counter for the number of hits Player 2 has taken

        # Define observation space with normalization and relative positions
        self.observation_space = spaces.Box(
            low=np.array([
                0,  # Relative Player 1 position x (min)
                -1,  # Relative Player 1 velocity x (normalized)
                0,  # Player 1 health (normalized)
                0,  # Relative Player 2 position x (min)
                -1,  # Relative Player 2 velocity x (normalized)
                0,  # Player 2 health (normalized)
                0,  # Normalized Distance between players
                0, 0,  # Player 1 and Player 2 states (e.g., idle, attacking)
                0,  # Player 1 attack states: light, medium, heavy
                0  # Player 2 attack states: light, medium, heavy
            ]),
            high=np.array([
                1.0,  # Relative Player 1 position x (max normalized to 1)
                1.0,  # Relative Player 1 velocity x (max normalized to 1)
                1.0,  # Player 1 health (max normalized to 1)
                1.0,  # Relative Player 2 position x (max normalized to 1)
                1.0,  # Relative Player 2 velocity x (max normalized to 1)
                1.0,  # Player 2 health (max normalized to 1)
                1.0,  # Normalized Distance between players (max normalized to 1)
                1, 1,  # State encoding (0 = idle, 1 = attacking)
                4,  # Player 1 attack states: 0, 1, 2, 3
                4  # Player 2 attack states: 0, 1, 2, 3
            ]),
            dtype=np.float32
        )

        # Initialize previous state variables for reward calculation
        self.prev_p1_health = 1.0  # Normalized
        self.prev_p2_health = 1.0  # Normalized
        self.prev_distance = 1.0  # Normalized

        # Curriculum Learning Parameters
        self.total_steps = 0  # Initialize step counter

        # Define curriculum phases as a list of tuples (step_threshold, p_random)
        self.curriculum = [
            (50000, 0.8),   # Up to 50,000 steps: 80% random
            (100000, 0.5),  # 50,001 to 100,000 steps: 50% random
            (float('inf'), 0.2)  # Beyond 100,000 steps: 20% random
        ]

        self.current_phase = 0  # Start with the first phase
        self.hits_taken_low_health = 0
        self.hits_given_low_health = 0

    def get_current_mixture_ratio(self):
        """
        Determine the current mixture ratio based on total steps.
        Returns the probability of taking a random action.
        """

        return self.p_random  # Default to the last phase's p_random
    def set_current_mixture_ratio(self,phase):
        """
        Determine the current mixture ratio based on total steps.
        Returns the probability of taking a random action.
        """
        if phase == 0:
            self.p_random = 0.8
        elif phase == 1:
            self.p_random = 0.5
        else:
            self.p_random = 0.2



    def simulate_key_press(self, key_code):
        # Call the wrapper function in Java to handle the key press
        self.game.simulateKeyPress(key_code)

    def start_game_auto(self):
        # Start the game by simulating key presses
        self.simulate_key_press(KeyEvent.VK_ENTER)
        time.sleep(1)  # Wait for the game to initialize
        self.simulate_key_press(KeyEvent.VK_ENTER)
        time.sleep(1)  # Wait for the game to start

        # Character selection
        self.simulate_key_press(KeyEvent.VK_1)
        time.sleep(0.5)
        self.simulate_key_press(KeyEvent.VK_1)
        time.sleep(0.5)

    def step(self, action_p2):
        """
        Execute one step in the environment.

        Parameters:
            action_p2 (int): Action taken by Player 2 (the agent).

        Returns:
            obs (np.array): Observation of the current state.
            reward (float): Reward obtained from taking the action.
            done (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated.
            info (dict): Additional information.
        """
        self.current_action = action_p2  # Store the current action
        self._handle_action(action_p2, player=2)  # Handle Player 2's action
        self.game.requestAdvance()  # Advance the game state
        obs_p1 = self._get_next_frame()  # Get the observation after Player 2's action
        # Determine Player 1's action based on the current mixture ratio
        p_random = 0.2 #self.get_current_mixture_ratio()
        if np.random.rand() < p_random:
            action_p1 = self.action_space.sample()  # Random action
        else:
            action_p1, _ = self.ppo_old.predict(obs_p1, deterministic=True)  # PPO action

        action_p1 = int(action_p1)  # Convert action to integer
        self._handle_action(action_p1, player=1)  # Handle Player 1's action
        self.game.requestAdvance()  # Advance the game state after Player 1's action

        # Get the updated observation
        obs = self._get_next_frame()

        # Compute the reward based on the new observation
        reward = self._compute_reward()

        # Check if the episode is done
        done, result = self._check_done()
        truncated = False  #
        self.cumulative_reward += reward
        info = {"result": result} if result is not None else {}
        self.total_steps += 1

        return obs, reward, done, truncated, info

    def _handle_action(self, action, player=1):
        """
        Handle the action taken by a player.

        Parameters:
            action (int): The action to be taken.
            player (int): The player number (1 or 2).
        """
        action = int(action)
        controls = self.game.getPlayer1Controls() if player == 1 else self.game.getPlayer2Controls()
        actions = {
            0: controls[2],  # Left
            1: controls[3],  # Right
            2: controls[4],  # Light attack
            3: controls[5],  # Medium attack
            4: controls[6],  # Heavy attack
            5: controls[1]   # Crouch (down)
        }
        if action in actions and actions[action] is not None:
            if player == 1:
                self.game.player1Move(actions[action])
            else:
                self.game.player2Move(actions[action])

    def _get_next_frame(self):
        """
        Retrieve the current observation frame.

        Returns:
            np.array: The observation array.
        """
        # Get absolute positions, velocities, and health
        p1_pos = self.game.getPlayer1().getPositionX() / 1400  # Normalize to [0,1]
        p1_vel = self.game.getPlayer1().getVelocityX() / 600    # Normalize to [-1,1]
        p1_health = self.game.getPlayer1().getHealth() / 100    # Normalize to [0,1]
        p2_pos = self.game.getPlayer2().getPositionX() / 1400  # Normalize to [0,1]
        p2_vel = self.game.getPlayer2().getVelocityX() / 600    # Normalize to [-1,1]
        p2_health = self.game.getPlayer2().getHealth() / 100    # Normalize to [0,1]
        distance = abs(self.game.getPlayer1().getPositionX() - self.game.getPlayer2().getPositionX()) / 1800  # Normalize to [0,1]

        # Get states
        combined_observation = self.game.getPlayer1().getObs()
        obs = np.array(combined_observation, dtype=np.float32)



        # Combine all observations
        normalized_continuous_obs = np.array([
            p1_pos,
            p1_vel,
            p1_health,
            p2_pos,
            p2_vel,
            p2_health,
            distance
        ], dtype=np.float32)
        final_obs = np.concatenate([normalized_continuous_obs, obs[7:]])

        return final_obs

    def _compute_reward(self, max_health=100):
        """
        Compute the reward for the current step.

        Parameters:
            max_health (int): The maximum health a player can have.

        Returns:
            float: The computed reward.
        """
        # Get current health values (normalized)
        p1_health = self.game.getPlayer1().getHealth() / max_health
        p2_health = self.game.getPlayer2().getHealth() / max_health

        # Calculate health differences since the last step
        p2_health_diff = self.prev_p2_health - p2_health  # Positive if Player 2 lost health
        p1_health_diff = self.prev_p1_health - p1_health  # Positive if Player 1 lost health

        # Initialize reward
        reward = 0
        reward += (p1_health_diff) * 3  # Reward for damaging Player 1
        reward -= (p2_health_diff) / 3  # Penalty for taking damage

        # Reward for defeating Player 1
        if p1_health <= 0 and p2_health > 0:
            reward += 10.0  # Large reward for winning

        # Penalize for maintaining a large distance
        current_distance = abs(self.game.getPlayer1().getPositionX() - self.game.getPlayer2().getPositionX()) / 1800
        if current_distance > 0.277:  # Equivalent to 500 distance units
            reward -= 0.05 * (current_distance / 0.277)  # Penalty increases with distance

        # Reward for closing distance
        if current_distance < self.prev_distance:
            reward += 0.01 * ((self.prev_distance - current_distance) / 0.277)  # Small reward

        # Penalize for inactivity or not attacking
        if self.current_action not in [2, 3, 4]:  # Not an attack action
            reward -= 0.1  # Small penalty for inactivity
        elif p1_health_diff == 0 and self.current_action in [2, 3, 4]:  # Attack action taken but no damage
            reward -= 0.2

        if self.current_action in [2, 3, 4]:
            reward += 0.05

        if p2_health < 0.5:
            if p2_health_diff > 0:
                reward -= 0.3
            if p1_health_diff > 0:
                reward += 0.5

        self.prev_p1_health = p1_health
        self.prev_p2_health = p2_health
        self.prev_distance = current_distance

        return reward

    def _check_done(self):
        """
        Check if the episode is done.

        Returns:
            tuple: (done (bool), result (str or None))
        """
        p1_health = self.game.getPlayer1().getHealth() / 100
        p2_health = self.game.getPlayer2().getHealth() / 100
        p1_wins = self.game.getP1Wins()
        p2_wins = self.game.getP2Wins()

        result = None
        if p1_health <= 0 and p2_health > 0:
            result = "P1 Lost"  # Player 1 lost
        elif p2_health <= 0 and p1_health > 0:
            result = "P1 Won"  # Player 1 won
        elif p2_health == 0 and p1_health == 0:
            result = "Draw"  # Draw

        # Check if time runs out
        elif self.game.getTimeLeft() <= 2:
            if p1_health > p2_health:
                result = "P1 Won"
            elif p2_health > p1_health:
                result = "P1 Lost"
            else:
                result = "Draw"
            return True, result

        # Check win counters
        elif p1_wins > 0:
            result = "P1 Won"
        elif p2_wins > 0:
            result = "P1 Lost"
        if result is not None:
            # Reseting health and win counters for the next episode
            self.game.getPlayer1().setHealth(100)
            self.game.getPlayer2().setHealth(100)
            self.game.setP2Wins(0)
            self.game.setP1Wins(0)
            self.hits_taken = 0

            return True, result

        return False, None

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.

        Returns:
            tuple: (observation (np.array), info (dict))
        """
        super().reset(seed=seed)
        self.game.resetGame()
        self.game.setP2Wins(0)
        self.game.setP1Wins(0)
        self.cumulative_reward = 0
        self.hits_taken = 0  # Reset hits_taken counter
        self.hits_taken_low_health = 0
        self.hits_given_low_health = 0
        self.prev_p1_health = self.game.getPlayer1().getHealth() / 100
        self.prev_p2_health = self.game.getPlayer2().getHealth() / 100
        self.prev_distance = abs(self.game.getPlayer1().getPositionX() - self.game.getPlayer2().getPositionX()) / 1800
        obs = self._get_next_frame()


        return obs, {}

    def render(self):
        """
        Render the environment. Not implemented.
        """
        pass
# gateway = JavaGateway()
# game_instance = gateway.entry_point.getGame()
# # game_instance.setRenderingEnabled(True)
# env = StreetFighterEnv(game=game_instance)
# print(env.observation_space)