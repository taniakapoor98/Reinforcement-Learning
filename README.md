# Street Fighter With RL
successfully integrated a Street Fighter game with reinforcement learning (RL)
by developing game-playing bots. A custom OpenAI Gym environment was designed to
make the game playable via Python and enable the implementation of reinforcement learning.
Proximal Policy Optimisation (PPO) and Deep Q-Networks (DQN) were used to train the
agent. The trained agents were able to learn and optimise their strategies according to the
situation. The agents demonstrated their ability to outperform a random player. Also, a PPOtrained model was used as a benchmark to evaluate the performance of other RL models. The
agents were tested in various setups like PPO vs. DQN and PPO vs. PPO, which
demonstrates PPO’s adaptability to dynamic changes and DQN’s proficiency in exploiting
deterministic reward structures. Additionally, the project included evaluation with humans,
which helped to gather real-time feedback. During the process, critical fixes were applied to
the game code, and a pull request was submitted and merged into the original GitHub
repository. The developed Gym wrapper could also be used for further research, and other
developers can train their own models using it. The work highlights the potential of RL in
gaming and its broader applicability to robotics and autonomous systems.

### Motivation:
Street Fighter presents a complex environment combining dynamic strategies, timing, and spatial awareness.
By tackling the challenges presented by Street Fighter, we can push the boundaries of AI capabilities and contribute to the understanding of complex decision-making.
Insights can be gained for broader applications like robotics and real-time gaming with AI.
### Goals:
Develop a Gym wrapper environment for RL.
Develop RL agents capable of playing Street Fighter better than a random player.
Evaluate performance against random player, other agents and human feedback.

### Stage 1: Beating Idle Player
Model Setup: PPO is initialized with the multi-layer perceptron policy. 
Training: The model is trained for 10000 steps using: model.learn(total_timesteps=num_timesteps, callback=reward_callback)
![image](https://github.com/user-attachments/assets/e0aa8b9c-0403-4cfc-b036-dcbeeadbd565)

Result: Agent successfully beats idle player.
![stg1](https://github.com/user-attachments/assets/9f78b0d4-17cd-4f30-9294-40719ca79425)







