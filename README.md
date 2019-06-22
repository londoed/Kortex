# Gorila
A Chapel framework for distributed deep reinforcement learning.

## Installation

When completed, will be downloadable through Mason.

## Usage

Example of proper framework usage:

```chapel
use Gorila as grla;

# Agent #
agent = new grla.DQN(grla.Regressor(), pi, env.info,
               approx_params, batch_size,
               n_approx=1, init_replay_size,
               max_replay_size, target_update_freg)

# Algorithm #
alg = new grla.Entity(agent, env)
alg.fit(n_steps=init_replay_size,
          n_steps_per_fit=init_replay_size)
#...#
alg.evaluate(n_episodes=10, render=True)
```

## Contributing

1. Fork it (<https://github.com/londoed/Gorila/fork>)
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request

## Contributors

- Eric Londo - creator, maintainer
