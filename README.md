# ![alt text](https://github.com/londoed/Kortex/blob/master/images/kortex_logo_grain.png)
A framework for distributed Deep Reinforcement Learning implemented in the Chapel Programming Language.

## Note

This work is in pre-pre-alpha stage. Right now, I'm just fleshing out ideas to get to where I want it to be (which looks like the code in the Usage section). Much of the current code probably looks more like Python than Chapel...this is because I'm waiting for tools like Arkouda (Chapel's NumPy interface) to better implement linear algebra functionality, and am still becoming used to Chapel as a programming language.

If you have ideas on how to implement this library in a more efficient "Chapel" way, please don't hesitate
to use my contact information below. Would love feedback and help with parallelizing the code when it comes time.

Thanks!

## Installation

When completed, Kortex will be available through the [Mason](https://chapel-lang.org/docs/tools/mason/mason.html) package manager.

## Usage

Example of proper framework usage:

```chapel
use Kortex;

// Agent
var model = new Regressor(),
    agent = new IMPALA(model, pi, env.info,
                   approx_params, batch_size,
                   n_approx=1, init_replay_size,
                   max_replay_size, target_update_freg); // Initializes new IMPALA Agent

// Algorithm
var alg = new Entity(agent, env); // Creates RL algorithm object
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
model = kx.Regressor(tf.keras.models.ResNet50())
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
- [ ] Implement NumPy functionality (Arkouda?)
- [ ] Better Error handling
- [ ] Wrappers for environments like OpenAI's Gym
- [ ] Unit testing
- [ ] Neural network architectures
- [ ] TensorFlow or PyTorch wrappers
- [ ] Integration for use in Python

Advanced Functionality:

- [ ] Population based training
- [ ] Neural Architecture Search
- [ ] Evolutionary Strategies/Genetic Algorithms

Algorithms:

- [x] DQN
- [x] DDPG
- [ ] SARSA
- [x] Blackbox Optimization (RWR, PGPE, REPS)
- [x] REINFORCE
- [x] GPOMDP
- [x] eNAC
- [x] SAC
- [ ] TRPO
- [ ] PPO
- [ ] A3C
- [ ] QMix
- [ ] MAML
- [ ] Reptile
- [ ] IMPALA

## Contributing

1. Fork it (<https://github.com/londoed/Kortex/fork>)
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request

## Contributors

- Eric D. Londo (londoed@comcast.net) - creator, maintainer

## Resources & Learning

### Reinforcement Learning

#### Books:

Reinforcement Learning: An Introduction by Richard S. Sutton and Andrew G. Barto\
Algorithms for Reinforcement Learning by Csaba Szepesvari\
Neuro-Dynamic Programming by Dimitri P. Betsekas and John N. Tsitsiklis\
Decision Making Under Uncertainty: Theory and Application by Mykel J. Kochenderfer\
Artificial Intelligence: Foundations of Computational Agents by David Poole and Alan Mackworth\
Deep Reinforcement Learning Hands-On by Maxim Lapan\
Python Reinforcement Learning by Sudharsan Ravichandiran, Sean Saito, Rajalingappaa Shanmugamani and Yang Wenzhuo\
Grokking Deep Reinforcement Learning by Miguel Morales\
Deep Reinforcement Learning in Action by Alexander Zai and Brandon Brown

#### Online Lectures:

Introduction to Reinforcement Learning by David Silver: [Playlist](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)\
Advanced Deep Learning & Reinforcement Learning by DeepMind: [Playlist](https://www.youtube.com/playlist?list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs)\
Move 37 Free Course by School of AI: [Course](https://www.theschool.ai/courses/move-37-course/)

#### Code Repositories:

[Dopamine](https://github.com/google/dopamine) by Google AI\
[Horizon](https://github.com/facebookresearch/Horizon) by Facebook AI Research (FAIR)\
[TensorFlow Agents](https://github.com/tensorflow/agents) by Google Brain\
[TensorFlow Reinforcement Learning (TRFL)](https://github.com/deepmind/trfl) by DeepMind\
[Ray](https://github.com/ray-project/ray) by RISE Lab at UC Berkeley\
[Mushroom](https://github.com/AIRLab-POLIMI/mushroom) by Carlo D'Eramo & David Tateo

### Chapel

See official website:

[Link](https://chapel-lang.org/)\
[Docs](https://chapel-lang.org/docs/)\
[Presentations](https://chapel-lang.org/presentations.html)\
[Tutorials](https://chapel-lang.org/tutorials.html)

## Inspiration and Credit
![alt text](https://github.com/londoed/Kortex/blob/master/images/GORILA.png)

This framework is inspired by original work done by DeepMind Technologies LTD and can be found in this [paper](https://arxiv.org/pdf/1507.04296.pdf).

In addition, work by Jeff Dean et al. in this [paper](https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf) on the DistBelief system as well as the TensorFlow [paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf) by Google Brain.

The initial implementation of Kortex was heavily inspired by work done by [Carlo D'Eramo](https://github.com/carloderamo)
and [David Tateo](https://github.com/boris-il-forte>) in this [repository](https://github.com/AIRLab-POLIMI/mushroom).

The paper for their ideas can be found [here](https://github.com/carloderamo/mushroom_paper/blob/master/mushroom.pdf).
