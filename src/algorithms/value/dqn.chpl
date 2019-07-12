module Kortex {
  // Python functions needed: deepcopy, // (line 155)
  // Numpy functions needed: clip(), any(), reshape(), max(), argmax(), mean()
  // TODO Categorical DQN
  use LinearAlgebra;

  class DQN: Agent {
    /*
      Deep Q-Network algorithm.

      "Human-Level Control Through Deep Reinforcement Learning".
      Mnih V. et al.. 2015.
    */
    proc init(approximator, policy: Policy, env_info: ENVInfo,
              batch_size: int, init_replay_size: int,
              max_replay_size: int, approximator_params: [] real,
              target_update_freq: int, fit_params: []=nil,
              n_approximators: int=1, clip_reward: bool=true) {
      /*
        Constructor.

        Args:
            approximator (object): the approximator to use to fit the
               Q-function.
            batch_size (int): the number of samples in a batch.
            init_replay_size (int): the number of samples to collect before
                starting the learning.
            max_replay_size (int): the maximum number of samples in the replay
                memory.
            approximator_params (associative array): parameters of the approximator
                to build.
            target_update_freq (int): the number of samples collected
                between each update of the target network.
            fit_params (associative array): parameters of the fitting algorithm of
                the approximator.
            n_approximators (int): the number of approximator to use in
                ``AverageDQN``.
            clip_reward (bool): whether to clip the reward or not.
      */
      if fit_params == nil {
        this.fit_params = new AssociativeArray();
      } else {
        this.fit_params = fit_params;
      }

      this.batch_size = batch_size;
      this.n_approximators = n_approximators;
      this.clip_reward = clip_reward;
      this.target_update_freq = target_update_freq;

      this.replay_memory = new ReplayMemory(init_replay_size, max_replay_size);
      this.n_updates = 0;

      apprx_params_train = copy(approximator_params);
      apprx_params_target = copy(approximator_params);
      this.approximator = new Regressor(approximator, apprx_params_train);
      this.target_approximator = new Regressor(approximator,
                                               n_models=this.n_approximators,
                                               apprx_params_target);
      policy.set_q(self.approximator);

      if this.n_approximators == 1 {
        this.target_approximator.model.set_weights(
          this.approximator.model.get_weights())
      } else {
        for i in 0..#this.n_approximators {
          this.target_approximator.model[i].set_weights(
            this.approximator.model.get_weights());
        }
      }
      super().init(policy, env_info);
    }

    proc fit(dataset: []) {
      this.replay_memory.add(dataset);

      if this.replay_memory.initialized {
        var state, action, reward, next_state, absorbing, _ = this.replay_memory.get(this.batch_size);

        if this.clip_reward {
          reward = clip(reward, -1, 1);
        }

        var q_next = next_q(next_state, absorbing),
            q = reward + this.env_info.gamma * q_next;

        this.approximator.fit(state, action, q, this.fit_params);
        this.n_updates += 1;

        if this.n_updates % this.target_update_freq == 0 {
          update_target();
        }
      }
    }

    proc update_target() {
      /*
        Update the target network.
      */
      this.target_approximator.model.set_weights(
        this.approximator.model.get_weights());
    }

    proc next_q(next_state: [], absorbing: []) {
      /*
        Args:
            next_state (array): the states where next action has to be
                evaluated;
            absorbing (array): the absorbing flag for the states in
                ``next_state``.
        Returns:
            Maximum action-value for each state in ``next_state``.
      */
      var q = this.target_approximator.predict(next_state);

      if any(absorbing) {
        q *= 1 - absorbing.reshape(-1, 1);
      }
      return max(q, axis=1);
    }

    proc draw_action(state: []) {
      var action = super(DQN).draw_action(Matrix(state));
      return action;
    }
  }

  class DoubleDQN: DQN {
    /*
      Double DQN algorithm.

      "Deep Reinforcement Learning with Double Q-Learning".
      Hasselt H. V. et al.. 2016.
    */
    proc next_q(next_state: [], absorbing: []) {
      var q = this.approximator.predict(next_state),
          max_a = argmax(q, axis=1),
          double_q = this.target_approximator.predict(next_state, max_a);

      if any(absorbing) {
        double_q *= 1 - absorbing;
      }
      return double_q;
    }
  }

  class AverageDQN: DQN {
    /*
      Averaged-DQN algorithm.

      "Averaged-DQN: Variance Reduction and Stabilization for Deep Reinforcement
      Learning". Anschel O. et al.. 2017.
    */
    proc init(approximator, policy: Policy, env_info: ENVInfo, params: []) {
      super().init(approximator, policy, env_info, params);
      this.n_fitted_target_models = 1;
      assert(this.target_approximator.model.type == Ensemble);
    }

    proc update_target() {
      var idx = this.n_updates / this.target_update_freq % this.n_approximators;
      this.target_approximator.model[idx].set_weights{
        this.approximator.model.get_weights()};

      if this.n_fitted_target_models < this.n_approximators {
        this.n_fitted_target_models += 1;
      }
    }

    proc next_q(next_state: [], absorbing: []) {
      var q = [];

      for idx in 0..#this.n_fitted_target_models {
        q.append(this.target_approximator.predict(next_state, idx=idx));
      }

      q = mean(q, axis=0);

      if any(absorbing) {
        q *= 1 - absorbing.reshape(-1, 1);
      }
      
      return max(q, axis=1);
    }
  }
}
