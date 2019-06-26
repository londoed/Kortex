# ![alt text](https://github.com/londoed/Kortex/blob/master/images/kortex_logo_grain.png)
A Chapel framework for distributed deep reinforcement learning.

## Note

This work is in pre-pre-alpha stage. Right now, I'm just fleshing out ideas to get to where I want it to be (which looks like the code in the Usage section). Much of the current code probably looks more like Python than Chapel...this is because I'm waiting for tools like Arkouda (Chapel's NumPy interface to progress) to better implement linear algebra functionality, and am still becoming used to Chapel as a programming language.

If you have ideas on how to implement this library in a more efficient "Chapel" way, please don't hesitate
to use my contact information below. Would love feedback and help with parallelizing the code when it comes time.

Thanks!

## Installation

When completed, Kortex will be available through the Mason package manager.

## Usage

Example of proper framework usage:

```chapel
use Kortex;

// Agent
model = new Regressor();
agent = new IMPALA(model, pi, env.info,
                   approx_params, batch_size,
                   n_approx=1, init_replay_size,
                   max_replay_size, target_update_freg); // Initializes new IMPALA Agent

// Algorithm
alg = new Entity(agent, env); // Creates RL algorithm object
alg.fit(n_steps=init_replay_size,
          n_steps_per_fit=init_replay_size); // Implements RL algorithm
//...//
alg.evaluate(n_episodes=10, render=true); // Evaluates success/failure of algorithm
```

Eventually, Kortex will be able to be called through Python while still using Chapel's parallelism.

```python
import kortex as kx
import tensorflow as tf

kx.chpl_setup()

# Agent #
model = kx.Regressor(tf.models.keras.ResNet50())
agent = kx.IMPALA(model, pi, env.info,
                   approx_params, batch_size,
                   n_approx=1, init_replay_size,
                   max_replay_size, target_update_freg)

# Algorithm #
alg = kx.Entity(agent, env)
alg.fit(n_steps=init_replay_size,
          n_steps_per_fit=init_replay_size)

#...#
alg.evaluate(n_episodes=10, render=True)

kx.chpl_cleanup()
```

## TODO

Main Functionality:

- [ ] Parallelize code
- [ ] Implement NumPy functionality
- [ ] Wrappers for environment's like OpenAI's Gym
- [ ] Unit testing
- [ ] Neural network architectures
- [ ] TensorFlow or PyTorch wrappers
- [ ] Integration for use in Python

Advanced Functionality:

- [ ] Population based training
- [ ] Neural Architecture Search
- [ ] Evolutionary Strategies/Genetic Algorithms

Algorithms:

- [ ] DQN
- [ ] DoubleDQN
- [ ] DDPG
- [ ] TRPO
- [ ] PPO
- [ ] A3C
- [ ] QMix
- [ ] MAML & Batched MAML
- [ ] Reptile
- [ ] IMPALA


## Contributing

1. Fork it (<https://github.com/londoed/Kortex/fork>)
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request

## Contributors

- Eric Londo - creator, maintainer

## Inspiration and Credit
![alt text](https://github.com/londoed/Kortex/blob/master/images/GORILA.png)

This framework is inspired by the original work done by DeepMind Technologies LTD and can be found in this paper:
<https://arxiv.org/pdf/1507.04296.pdf>

Also, work by Jeff Dean et al. in this paper:
<https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf>

The initial implementation of Kortex was heavily inspired by work done by Carlo D'Eramo <https://github.com/carloderamo>
and David Tateo <https://github.com/boris-il-forte> in this repository:
<https://github.com/AIRLab-POLIMI/mushroom>.

The paper for their ideas can be found here:
<https://github.com/carloderamo/mushroom_paper/blob/master/mushroom.pdf>
