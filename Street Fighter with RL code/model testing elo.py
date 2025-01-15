import time
import matplotlib.pyplot as plt
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO, DQN
from py4j.java_gateway import JavaGateway
from StreetFighter_with_ppo import StreetFighterEnv  # Import your environment class
import seaborn as sns
# Setup the game and environment
gateway = JavaGateway()
game_instance = gateway.entry_point.getGame()
game_instance.setRenderingEnabled(True)
env = StreetFighterEnv(game=game_instance)
check_env(env)
print('Environment created for testing.')
num_test_episodes = 100
# Load the models
model = PPO.load("best_model.zip")
model_old = DQN.load("dqn_sf_lstm_v2.zip")  # Load your DQN model
env.start_game_auto()
K_FACTOR = 32  # K-factor for Elo ratings


# Function to calculate expected outcome
def calculate_expected_outcome(rating_A, rating_B):
    return 1 / (1 + 10 ** ((rating_B - rating_A) / 400))


# Function to update Elo ratings
def update_elo(rating_A, rating_B, score_A, score_B):
    expected_A = calculate_expected_outcome(rating_A, rating_B)
    expected_B = calculate_expected_outcome(rating_B, rating_A)

    new_rating_A = rating_A + K_FACTOR * (score_A - expected_A)
    new_rating_B = rating_B + K_FACTOR * (score_B - expected_B)

    return new_rating_A, new_rating_B


# Function to test a specific Player 2 configuration against Player 1
def test_model_v3(model_p2, model_name, env, num_episodes):
    # Initialize Elo ratings for this Player 2 configuration
    elo_ratings = {"P1": 1200, f"P2_{model_name}": 1000}
    elo_progression = {"P1": [], f"P2_{model_name}": []}

    print(f"Testing Player 2 ({model_name}) vs Player 1...")
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Player 2 takes an action
            action_p2 = env.action_space.sample() if model_p2 is None else model_p2.predict(obs, deterministic=True)[0]
            action_p2 = int(action_p2)

            obs, reward, done, truncated, info = env.step(action_p2)
            total_reward += reward

            if done:
                result = info.get('result')
                if result == "P1 Won":
                    score_p2, score_p1 = 0, 1
                elif result == "P1 Lost":
                    score_p2, score_p1 = 1, 0
                else:  # Draw
                    score_p2, score_p1 = 0.5, 0.5

                # Update Elo ratings
                elo_ratings[f"P2_{model_name}"], elo_ratings["P1"] = update_elo(
                    elo_ratings[f"P2_{model_name}"], elo_ratings["P1"], score_p2, score_p1
                )

                # Track progression
                elo_progression[f"P2_{model_name}"].append(elo_ratings[f"P2_{model_name}"])
                elo_progression["P1"].append(elo_ratings["P1"])
                break

    # Plot Elo ratings for this configuration
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(elo_progression[f"P2_{model_name}"]) + 1), elo_progression[f"P2_{model_name}"],
             label=f"{model_name} (Player 2)", color="blue")
    plt.plot(range(1, len(elo_progression["P1"]) + 1), elo_progression["P1"], label="Player 1 (Environment)",
             color="red", linestyle="--")

    plt.title(f"Elo Ratings Progression: {model_name} vs Player 1")
    plt.xlabel("Match Number")
    plt.ylabel("Elo Rating")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"elo_ratings_{model_name}_vs_P1.png")
    # plt.show()

    return elo_progression


# Test each Player 2 configuration and collect Elo progressions
elo_progression_ppo = test_model_v3(model, "PPO+LSTM", env, num_test_episodes)
elo_progression_dqn = test_model_v3(model_old, "DQN+LSTM", env, num_test_episodes)
elo_progression_random = test_model_v3(None, "Random", env, num_test_episodes)


# Density plot function
def plot_elo_density(elo_ratings_p1, elo_ratings_p2, model_name):
    plt.figure(figsize=(10, 6))

    # KDE plots for Elo ratings
    sns.kdeplot(elo_ratings_p1, label="Player 1 (P1)", color="red", fill=True, alpha=0.4)
    sns.kdeplot(elo_ratings_p2, label=f"Player 2 ({model_name})", color="blue", fill=True, alpha=0.4)

    plt.title(f"Elo Rating Density: P1 vs {model_name}")
    plt.xlabel("Elo Rating")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"elo_density_P1_vs_{model_name}.png")
    # plt.show()


# Plot density after collecting data
plot_elo_density(elo_progression_ppo["P1"], elo_progression_ppo["P2_PPO+LSTM"], "PPO+LSTM")
plot_elo_density(elo_progression_dqn["P1"], elo_progression_dqn["P2_DQN+LSTM"], "DQN+LSTM")
plot_elo_density(elo_progression_random["P1"], elo_progression_random["P2_Random"], "Random")
