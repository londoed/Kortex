# Gorila
A Chapel framework for distributed deep reinforcement learning.

## Installation

When completed, Gorila will be available through the Mason package manager.

## Usage

Example of proper framework usage:

```chapel
use Gorila;

// Agent
agent = new DQN(Regressor(), pi, env.info,
               approx_params, batch_size,
               n_approx=1, init_replay_size,
               max_replay_size, target_update_freg); // Initializes new DQN Agent

// Algorithm
alg = new Entity(agent, env); // Creates RL algorithm object
alg.fit(n_steps=init_replay_size,
          n_steps_per_fit=init_replay_size); // Implements RL algorithm
//...//
alg.evaluate(n_episodes=10, render=True); // Evaluates success/failure of algorithm
```

## Contributing

1. Fork it (<https://github.com/londoed/Gorila/fork>)
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request

## Contributors

- Eric Londo - creator, maintainer

## Inspiration
![alt text](https://github.com/londoed/Gorila/blob/master/images/GORILA.png)

This framework is inspired by the original work done by DeepMind Technologies and can be found in this paper:
<https://arxiv.org/pdf/1507.04296.pdf>
