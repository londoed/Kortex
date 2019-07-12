module Kortex {
  // Numpy functions needed: seterr(), mean(), logical_not(), isfinite(), square(), zeros(), linalg.pinv(), ones()
  // Python functions needed: enumerate()
  use LinearAlgebra;

  class PolicyGradient: Agent {
    /*
      Abstract class to implement a generic Policy Search algorithm using the
      gradient of the policy to update its parameters.

      "A survey on Policy Gradient algorithms for Robotics". Deisenroth M. P. et
      al.. 2011.
    */
    proc init(policy: Policy, env_info: ENVInfo, learning_rate: real, features: AssociatveArray) {
      /*
        Constructor.

        Args:
          learning_rate (real): the learning rate.
      */
      this.learning_rate = learning_rate;
      this.df: real = 1;
      this.J_episode: real = 0;
      super().init(policy, env_info, features);
    }

    proc fit(dataset: []) {
      var J = [];
      this.df = 1.0;
      this.J_episode = 0.0;
      init_update();

      for sample in dataset {
        var x, u, r, xn, _, last = parse(sample);
        step_update(x, u, r);
        this.J_episode += this.df * r;
        this.df *= this.env_info.gamma;

        if last {
          episode_end_update();
          J.append(this.J_episode);
          this.J_episode = 0.0;
          this.df = 1.0;
          init_update();
        }
      }
      update_params(J);
    }

    proc update_params(J: []) {
      /*
        Update the parameters of the policy.

        Args:
          J {array): list of the cumulative discounted rewards for
              each episode in the dataset.
      */
      var res = compute_gradient(J),
          theta = this.policy.get_weights();

      if res.length == 1 {
        var grad = res[0],
            delta = this.learning_rate(grad) * grad;
      } else {
        var grad, nat_grad = res,
            delta = this.learning_rate(grad, nat_grad) * nat_grad;
      }

      var theta_new = theta + delta;
      this.policy.set_weights(theta_new);
    }

    proc init_update() {
      /*
        This function is called, when parsing the dataset, at the beginning
        of each episode. The implementation is dependent on the algorithm (e.g.
        REINFORCE resets some data structure).
      */
      writeln("PolicyGradient is an abstract class, init_update is unavailable.");
    }

    proc step_update(x: Matrix, u: Matrix, r: Matrix) {
      /*
        This function is called when parsing the dataset at each episode step.

        Args:
          x (Matrix): the state at the current step.
          u (Matrix): the action at the current step.
          r (Matrix): the reward at the current step.
      */
      writeln("PolicyGradient is an abstract class, step_update unavailable.");
    }

    proc episode_end_update() {
      /*
        This function is called, when parsing the dataset, at the beginning
        of each episode. The implementation is dependent on the algorithm (e.g.
        REINFORCE updates some data structures).
      */
      writeln("PolicyGradient is an abstract class, episode_end_update is unavailable.");
    }

    proc compute_gradient(J: []) {
      /*
        Return the gradient computed by the algorithm.

        Args:
          J (array): list of the cumulative discounted rewards for each
              episode in the dataset.
      */
      writeln("PolicyGradient is an abstract class, compute_gradient is unavailable.");
    }

    proc parse(sample: []): Tuple {
      /*
        Utility to parse the sample.
        s
        Args:
             sample (array): the current episode step.

        Returns:
            A tuple containing state, action, reward, next state, absorbing and
            last flag. If provided, ``state`` is preprocessed with the features.
      */
      var state = sample[0],
          action = sample[1],
          reward = sample[2],
          next_state = sample[3],
          absorbing = sample[4],
          last = sample[5];

      if this.phi != nil {
        state = this.phi(state);
      }
      return Tuple(state, action, reward, next_state, absorbing, last);
    }
  }

  class REINFORCE: PolicyGradient {
    /*
      REINFORCE algorithm.

      "Simple Statistical Gradient-Following Algorithms for Connectionist
      Reinforcement Learning", Williams R. J.. 1992.
    */
    proc init(policy: Policy, env_info: ENVInfo, learning_rate: real, features: AssociatveArray=nil) {
      super().init(policy, env_info, learning_rate, features);
      this.sum_d_log_pi = nil;
      this.list_sum_d_log_pi = [];
      this.baseline_num = [];
      this.baseline_den = [];

      // Ignore divide by zero
      seterr(division='ignore', invalid='ignores')
    }

    proc compute_gradient(J: []) {
      var baseline = mean(this.baseline_num, axis=0) / mean(this.baseline_den, axis=0);
      baseline[logical_not(isfinite(baseline))] = 0.0;
      var grad_J_episode = [];

      for i, J_episode in enumerate(J) {
        var sum_d_log_pi = this.list_sum_d_log_pi[i];
        grad_J_episode.append(sum_d_log_pi * (J_episode - baseline));
      }

      var grad_J = mean(grad_J_episode, axis=0);
      this.list_sum_d_log_pi = [];
      this.baseline_num = [];
      this.baseline_den = [];

      return grad_J;
    }

    proc step_update(x: Matrix, u: Matrix, r: Matrix) {
      var d_log_pi = this.policy.diff_log(x, u);
      this.sum_d_log_pi += d_log_pi;
    }

    proc episode_end_update() {
      this.list_sum_d_log_pi.append(this.sum_d_log_pi);
      var squared_sum_d_log_pi = square(this.sum_d_log_pi);
      this.baseline_num.append(squared_sum_d_log_pi * this.J_episode);
      this.baseline_den.append(squared_sum_d_log_pi);
    }

    proc init_update() {
      this.sum_d_log_pi = zeros(this.policy.weights_size);
    }
  }

  class GPOMDP: PolicyGradient {
    /*
      GPOMDP algorithm.

      "Infinite-Horizon Policy-Gradient Estimation". Baxter J. and Bartlett P. L..
      2001.
    */
    proc init(policy: Policy, env_info: ENVInfo, learning_rate: real, features: AssociatveArray=nil) {
      super().init(policy, env_info, learning_rate, features);
      this.sum_d_log_pi = nil;
      this.list_sum_d_log_pi = [];
      this.list_sum_d_log_pi_ep = [];
      this.list_reward = [];
      this.list_reward_ep = [];
      this.baseline_num = [];
      this.baseline_den = [];
      this.step_count: int = 0;

      // Ignore division by zero
      seterr(divide='ignore', invalid='ignore');
    }

    proc compute_gradient(J: []) {
      var gradient = zeros(this.policy.weights_size),
          n_episodes = this.list_sum_d_log_pi_ep.length;

      for i in 0..#n_episodes.length {
        var list_sum_d_log_pi = this.list_sum_d_log_pi_ep[i],
            list_reward = this.list_reward_ep[i],
            n_steps = list_sum_d_log_pi.length;

        for t in 0..#n_steps {
          var step_grad = list_sum_d_log_pi[t],
              step_reward = list_reward[t],
              baseline = this.baseline_num[t] / this.baseline_den[t];
          baseline[logical_not(isfinite(baseline))] = 0.0;
          gradient += (step_reward - baseline) * step_grad;
        }
      }
      gradient /= n_episodes;
      this.list_reward_ep = [];
      this.list_sum_d_log_pi_ep = [];
      this.baseline_num = [];
      this.baseline_den = [];

      return gradient;
    }

    proc step_update(x: Matrix, u: Matrix, r: Matrix) {
      var discounted_reward = this.df * r;
      this.list_reward.append(discounted_reward);

      var d_log_pi = this.policy.diff_log(x, u);
      this.sum_d_log_pi += d_log_pi;
      this.list_sum_d_log_pi.append(this.sum_d_log_pi);

      var squared_sum_d_log_pi = square(this.sum_d_log_pi);

      if this.step_count < this.baseline_num.length {
        this.baseline_num[this.step_count] += discounted_reward * squared_sum_d_log_pi;
        this.baseline_den[this.step_count] += squared_sum_d_log_pi;
      } else {
        this.baseline_num.append(discounted_reward * squared_sum_d_log_pi);
        this.baseline_den.append(squared_sum_d_log_pi);
      }
      this.step_count += 1;
    }

    proc episode_end_update() {
      this.list_reward_ep.append(this.list_reward);
      this.list_reward = [];
      this.list_sum_d_log_pi_ep.append(this.list_sum_d_log_pi);
      this.list_sum_d_log_pi = [];
    }

    proc init_update() {
      this.sum_d_log_pi = zeros(this.policy.weights_size);
      this.list_sum_d_log_pi = [];
      this.step_count = 0;
    }
  }

  class eNAC: PolicyGradient {
    /*
      Episodic Natural Actor Critic algorithm.

      "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
      Peters J. 2013.
    */
    proc init(policy: Policy, env_info: ENVInfo, learning_rate: real, features: AssociatveArray=nil,
              critic_features: Features=nil) {
      /*
        Constructor.

        Args:
          critic_features (Features): features used by the critic.
      */
      super().init(policy, env_info, learning_rate, features);
      this.phi_c = critic_features;
      this.sum_grad_log = nil;
      this.psi_ext = nil;
      this.sum_grad_log_list = [];
    }

    proc compute_gradient(J: []) {
      var R = Matrix(J),
          PSI = Matrix(this.sum_grad_log_list),
          w_and_v = pinv(PSI).dot(R),
          nat_grad = w_and_v[..this.policy.weights_size];
      this.sum_grad_log_list = [];

      return nat_grad;
    }

    proc step_update(x: Matrix, u: Matrix, r: Matrix) {
      this.sum_grad_log += this.policy.diff_log(x, u);
      if this.psi_ext == nil {
        if this.phi_c == nil {
          this.phi_ext = ones(1);
        } else {
          this.phi_ext = this.phi_c(x);
        }
      }
    }

    proc episode_end_update() {
      var psi = concatenate((this.sum_grad_log, this.psi_ext));
      this.sum_grad_log_list.append(psi);
    }

    proc init_update() {
      this.psi_ext = nil;
      this.sum_grad_log = zeros(this.policy.weights_size);
    }
  }
}
