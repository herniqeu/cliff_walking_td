# Cliff Walking Q-Learning Implementation

## Overview
This repository implements Q-learning algorithms for the Cliff Walking environment, demonstrating the differences between Epsilon-Greedy and Softmax exploration strategies. The implementation includes comprehensive metrics collection, visualization tools, and analysis capabilities.

## Mathematical Foundation

### Q-Learning Update Rule
The Q-learning algorithm updates action-values using the following equation:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t)]$$

where:
- $Q(s_t, a_t)$ is the value of taking action $a_t$ in state $s_t$
- $\alpha$ is the learning rate
- $\gamma$ is the discount factor
- $r_{t+1}$ is the immediate reward
- $\max_{a} Q(s_{t+1}, a)$ is the maximum Q-value for the next state

### Exploration Strategies

#### 1. Epsilon-Greedy Policy
Action selection probability:

$$P(a|s) = \begin{cases} 
1 - \epsilon + \frac{\epsilon}{|A|} & \text{if } a = \argmax_{a'} Q(s,a') \\
\frac{\epsilon}{|A|} & \text{otherwise}
\end{cases}$$

#### 2. Softmax Policy
Action selection probability using Boltzmann distribution:

$$P(a|s) = \frac{\exp(Q(s,a)/\tau)}{\sum_{a'} \exp(Q(s,a')/\tau)}$$

where $\tau$ is the temperature parameter controlling exploration.

## Features

### Environment
- Custom implementation of the Cliff Walking environment
- Grid-based navigation task with negative rewards for falling off the cliff
- State space: $4 \times 12$ grid
- Action space: 4 discrete actions (up, down, left, right)

### Learning Algorithms
1. TD Learning with Epsilon-Greedy exploration
2. TD Learning with Softmax exploration

### Metrics Collection
- Episode rewards and lengths
- TD errors
- State visitation frequencies
- Action selection history
- Value function evolution
- Policy entropy
- Success rate

### Visualization Tools
- Value function heatmaps
- Training progress plots
- Performance comparison metrics

## Requirements
```python
gymnasium
tensorflow
plotly
numpy
pandas
tqdm
```

## Usage

```python
# Initialize environment and policies
env = CliffWalkingEnvironment()
epsilon_greedy = EpsilonGreedyPolicy(env)
softmax = SoftmaxPolicy(env)

# Train using different policies
td_learner = TDLearner(env)
results = td_learner.learn(epsilon_greedy, episodes=500)

# Visualize results
plot_value_function_heatmap(env, results['metrics'])
plot_training_progress(results['metrics'])
```

## Performance Analysis

### Convergence
The value function converges according to the Bellman equation:

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a)[r(s,a,s') + \gamma V^*(s')]$$

### Exploration-Exploitation Trade-off
- Epsilon-Greedy: Linear decay of exploration rate
- Softmax: Temperature-based probabilistic selection

## Implementation Details

### State Value Updates
The state values are updated using TD learning:

$$V(s_t) \leftarrow V(s_t) + \alpha[r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$$

### Metrics Computation
Policy entropy is calculated as:

$$H(\pi) = -\sum_{a} \pi(a|s) \log \pi(a|s)$$