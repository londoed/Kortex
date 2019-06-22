# Gorila
A Chapel framework for distributed deep reinforcement learning.

## Installation

Use this command to install Cortex to your machine:

```yaml
pip install cortex-rl
```

or:

```yaml
conda install cortex-rl
```

## Usage

Example of proper framework usage:

```python
import cortex as cx

# Agent #
agent = cx.DQN(cx.Regressor(), pi, env.info,
               approx_params, batch_size,
               n_approx=1, init_replay_size,
               max_replay_size, target_update_freg)

# Algorithm #
alg = cx.Base(agent, env)
alg.learn(n_steps=init_replay_size,
          n_steps_per_fit=init_replay_size)
#...#
alg.evaluate(n_episodes=10, render=True)
```

Simple chain Q-Learning example:

```python
import cortex as cx

env = cx.generate_simple_chain(state_n=5, goal_states=[2],
                               prob=0.8, rew=1, gamma=0.9)
epsilon = cx.Parameter(value=0.15)
pi = cx.EpsGreedy(epsilon=epsilon)

learning_rate = cx.Parameter(value=0.2)
alg_params = dict(learning_rate=learning_rate)
agent = cx.QLearning(pi, env.info, **alg_params)
alg = cx.Base(agent, env)
alg.learn(n_steps=10000, n_steps_per_fit=1)
```


## Contributing

1. Fork it (<https://github.com/londoed/Gorila/fork>)
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request

## Contributors

- Eric Londo - creator, maintainer
