# Implementation of Reinforcement Learning for LunarLander-v2

## Authors
- **Saumya Maurya** - 200552573  
- **Tanveer Singh** - 200554065  
- **Sanket Shreekant Parab** - 200555449  

## Project Description
This project demonstrates the implementation of reinforcement learning (RL) algorithms to solve the LunarLander-v2 problem. The goal is to train an agent to land a spacecraft safely between two flags using state-of-the-art RL methods. The project was implemented using Python and Jupyter Notebook.

## Algorithms Implemented
### 1. Deep Q-Network (DQN)
- **Input Layer**: Accepts an 8-dimensional state vector.
- **Hidden Layers**: Two fully connected layers with 128 and 64 neurons (ReLU activation).
- **Output Layer**: Produces four outputs corresponding to the discrete action space.
- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Loss Function**: Mean Squared Error (MSE).

### 2. Actor-Critic Algorithm
#### Actor Network:
- **Input Layer**: Takes the state as input.
- **Hidden Layers**: Two fully connected layers with 128 and 64 neurons (ReLU activation).
- **Output Layer**: Outputs action probabilities using the Softmax activation function.

#### Critic Network:
- **Architecture**: Similar to the Actor Network but outputs a scalar value representing the state-value function.

## Training Process
### Environment Setup
- **Environment**: LunarLander-v2 initialized using Gymnasium.
- **State**: 8-dimensional vector (position, velocity, orientation).
- **Rewards**: Encourages smooth and safe landings.

### Training Details
- **DQN**: Utilized experience replay and target network updates every 10 episodes.
- **Actor-Critic**: Employed forward-view temporal-difference updates.
- **Hyperparameters**:
  - Episodes: 2000
  - Epsilon-greedy policy (decay from 1.0 to 0.01)
  - Discount Factor (γ): 0.99

### Visualization
- Training rewards tracked and episodes rendered every 100 iterations.
- Performance analyzed via cumulative rewards over episodes.

## Results
- **DQN**: Stable landings achieved after 1000 episodes (average reward of 200).
- **Actor-Critic**: Faster convergence with stable rewards after 600 episodes.

## Challenges and Solutions
- **Sparse Reward Structure**: Addressed with reward shaping.
- **Balancing Exploration and Exploitation**: Solved using decaying epsilon strategy.
- **Stability in DQN Training**: Improved using a target network and experience replay.

## Suggestions for Improvement
- Implement prioritized experience replay and curiosity-driven exploration.
- Experiment with deeper networks or convolutional layers.
- Test in more complex continuous-action environments.

## Instructions for Running the Code
### Prerequisites
- **Libraries**: Install `gymnasium`, `torch`, `numpy`, and `matplotlib`.
- **Environment**: Ensure Python 3.x with Jupyter Notebook support.

### Running the Code
1. Navigate to the directory containing the Jupyter Notebook.
2. Run `LunarLander_Training_and_Visualization_Fixed.ipynb`.

## Conclusion
This project successfully demonstrates the application of forward-view RL algorithms in the LunarLander-v2 environment, achieving promising performance and offering avenues for further enhancement through advanced techniques.
