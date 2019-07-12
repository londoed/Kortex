module Kortex {
  // Numpy functions needed: max(), random.uniform(), random.choice(), argwhere(), ravel(), maximum(), sqrt(),
  // zeros(), random.normal(), repeat(), argmax(), unique()
  // Python functions needed: copy.deepcopy()
  use LinearAlgebra;

  class TD: Agent {
    /*
      Implements functions to run TD algorithms.
    */
    proc init(approximator: Approximator, policy: Policy, env_info: ENVInfo, learning_rate: Parameter, features: AssociativeArray=nil) {
      /*
        Constructor.

        Args:
            approximator (Approximator): the approximator to use to fit the
               Q-function;
            learning_rate (Parameter): the learning rate.
      */
      this.alpha = learning_rate;
      policy.set_q(approximator);
      this.approximator = approximator;
      super().init(policy, env_info, features);
    }

    proc fit(dataset: []) {
      assert(dataset.length == 1);
      var state, action, reward, next_state, absorbing = parse(dataset);
      update(state, action, reward, next_state, absorbing);
    }

    proc parse(dataset: []) {
      /*
        Utility to parse the dataset that is supposed to contain only a sample.

        Args:
            dataset (Array): the current episode step.
        Returns:
            A tuple containing state, action, reward, next state, absorbing and
            last flag.
      */
      var sample = dataset[0],
          state = sample[0],
          action = sample[1],
          reward = sample[2],
          next_state = sample[3],
          absorbing = sample[4];
      return state, action, reward, next_state, absorbing;
    }

    proc update(state: Matrix, action: Matrix, reward: Matrix, next_state: Matrix, absorbing: Matrix) {
      /*
        Update the Q-table.

        Args:
            state (Matrix): state.
            action (Matrix): action
            reward (Matrix): reward.
            next_state (Matrix): next state.
            absorbing (Matrix): absorbing flag.
      */
      continue;
    }
  }

  class QLearning: TD {
    /*
      Q-Learning algorithm.
      "Learning from Delayed Rewards". Watkins C.J.C.H.. 1989.
    */
    proc init(policy: Policy, env_info: ENVInfo, learning_rate: Parameter) {
      this.Q = new Table(env_info.size);
      super().init(this.Q, policy, env_info, learning_rate);
    }

    proc update(state: Matrix, action: Matrix, reward: Matrix, next_state: Matrix, absorbing: Matrix) {
      var q_current = this.Q[state, action];

      if !absorbing {
        var q_next = max(this.Q[next_state, ..]);
      } else {
        var q_next = 0;
      }

      this.Q[state, action] = q_current + this.alpha(state, action) * (reward + this.env_info.gamma * q_next - q_current);
    }
  }

  class DoubleQLearning: TD {
    /*
      Double Q-Learning algorithm.
      "Double Q-Learning". Hasselt H. V.. 2010.
    */
    proc init(policy: Policy, env_info: ENVInfo, learning_rate: Parameter) {
      this.Q = new EnsembleTable(2, env_info.size);
      super().init(this.Q, policy, env_info, learning_rate);
      this.alpha = [deepcopy(this.alpha), deepcopy(this.alpha)];

      assert(this.Q.length == 2);
    }

    proc update(state: Matrix, action: Matrix, reward: Matrix, next_state: Matrix, absorbing: Matrix) {
      if random.uniform() < 0.5 {
        var approximator_idx: int = 0;
      } else {
        var approximator_idx: int = 1;
      }

      var q_current = this.Q[approximator_idx][state, action];

      if !absorbing {
        var q_ss = this.Q[approximator_idx][next_state, ..],
            max_q = max(q_ss),
            a_n = Matrix([random.choice(argwhere(q_ss == max_q). ravel())]),
            q_next = this.Q[1 - approximator_idx][next_state, a_n];
      } else {
        var q_next = 0;
      }

      var q = q_current + this.alpha[approximator_idx](state, action) * (reward + this.env_info.gamma * q_next - q_current);
      this.Q[approximator_idx][state, action] = q;
    }
  }

  class WeightedQLearning: TD {
    /*
      Weighted Q-Learning algorithm.
      "Estimating the Maximum Expected Value through Gaussian Approximation".
      D'Eramo C. et. al.. 2016.
    */
    proc init(policy: Policy, env_info: ENVInfo, learning_rate: Parameter, sampling: bool=true, precision: int=1000, weighted_policy: bool=false) {
      /*
        Constructor.

        Args:
            sampling (bool): use the approximated version to speed up
                the computation.
            precision (int): number of samples to use in the approximated
                version.
            weighted_policy (bool): whether to use the weighted policy
                or not.
      */
      this.Q = new Table(env_info.size);
      this.sampling = sampling;
      this.precision = precision;
      super().init(this.Q, policy, env_info, learning_rate);
      this.n_updates = new Table(env_info.size);
      this._Q = new Table(env_info.size);
      this.Q2 = new Table(env_info.size);
      this.weights_var = new Table(env_info.size);
      this.use_weighted_policy = weighted_policy;
    }

    proc update(state: Matrix, action: Matrix, reward: Matrix, next_state: Matrix, absorbing: Matrix) {
      var q_current = this.Q[state, action];

      if !absorbing {
        var q_next = next_q(next_state);
      } else {
        var q_next = 0;
      }

      var target = reward + this.env_info.gamma * q_next,
          alpha = this.alpha(state, action);
      this.Q[state, action] = q_current + alpha * (target - q_current);
      this.n_updates[state, action] += 1;
      this._Q[state, action] += (target - this._Q[state, action]) / this.n_updates[state, action];
      this.Q2[state, action] += (target**2 - this.Q2[state, action]) / this.n_updates[state, action];
      this.weights_var[state, action] = (1 - alpha)**2.0 * this.weights_var[state, action] + alpha**2.0;

      if this.n_updates[state, action] > 1 {
        var variance = this.n_updates[state, action] * (this.Q2[state, action] - this._Q[state, action]**2) / (this.n_updates[state, action] - 1.0),
            var_estimator = variance * this.weights_var[state, action];
        var_estimator = maximum(var_estimator, 1e-10);

        this.sigma[state, action] = sqrt(var_estimator);
      }
    }

    proc next_q(next_state: Matrix) {
      /*
        Args:
            next_state (Matrix): the state where next action has to be
                evaluated.
        Returns:
            The weighted estimator value in ``next_state``.
      */
      var means = this.Q[next_state, ..],
          sigmas = zeros(this.Q.shape[-1]);

      for a in 0..#sigmas.size {
        sigmas[a] = this.sigma[next_state, Matrix([a])];
      }

      if this.sampling {
        var samples = random.normal(repeat([means], this.precision, 0), repeat([sigmas], this.precision, 0)),
            max_idx = argmax(samples, axis=1);
        var max_idx, max_count = unique(max_idx, return_counts=true,
            count = zeros(means.size);
        count[max_idx] = max_count;
      } else {
        writeln("Not implemented.")
      }

      if this.use_weighted_policy {
        this.next_action = Matrix([random.choice(this.env_info.action_space.n, p=this.w)]);
      }
      return dot(this.w, means);
    }
  }

  class SpeedyQLearning: TD {
    /*
      Speedy Q-Learning algorithm.
      "Speedy Q-Learning". Ghavamzadeh et. al.. 2011.
    */
    proc init(policy: Policy, env_info: ENVInfo, learning_rate: Parameter) {
      this.Q = new Table(env_info.size);
      this.old_q = deepcopy(this.Q);
      super().init(this.Q, policy, env_info, learning_rate);
    }

    proc update(state: Matrix, action: Matrix, reward: Matrix, next_state: Matrix, absorbing: Matrix) {
      var old_q = deepcopy(this.Q);

      if !absorbing {
        var max_q_cur = max(this.Q[next_state, ..]),
            max_q_old = max(this.old_q[next_state, ..])
      } else {
        var max_q_cur = 0,
            max_q_old = 0;
      }

      var target_cur = reward + this.env_info.gamma * max_q_cur,
          target_old = reward + this.env_info.gamma * max_q_old,
          alpha = this.alpha(state, action),
          q_cur = this.Q[state, action];
      this.Q[state, action] = q_cur + alpha * (target_old - q_cur) + (1.0 - alpha) * (target_cur - target_old);
      this.old_q = old_q;
    }
  }

  class SARSA: TD {
    /*
      SARSA algorithm.
    */
    proc init(policy: Policy, env_info: ENVInfo, learning_rate: Parameter) {
      this.Q = new Table(env_info.size);
      super().init(this.Q, policy, env_info, learning_rate);
    }

    proc update(state: Matrix, action: Matrix, reward: Matrix, next_state: Matrix, absorbing: Matrix) {
      var q_current = this.Q[state, action];
      this.next_action = draw_action(next_state);

      if !absorbing {
        var q_next = this.Q[next_state, this.next_action];
      } else {
        var q_next = 0;
      }

      this.Q[state, action] = q_current + this.alpha(state, action) * (reward + this.env_info.gamma * q_next - q_current);
    }
  }

  class SARSALambdaDiscrete: TD {
    /*
      Discrete version of the SARSA(lambda) algorithm.
    */
    proc init(policy: Policy, env_info: ENVInfo, learning_rate: Parameter, lambda_coeff: real, trace: string="replacing") {
      /*
        Constructor.

        Args:
            lambda_coeff (real): eligibility trace coefficient.
            trace (string): type of eligibility trace to use.
      */
      this.Q = new Table(env_info.size);
      this.lambda = lambda_coeff;
      this.e = new EligibilityTrace(this.Q.shape, trace);
      super().init(this.Q, policy, env_info, learning_rate);
    }

    proc update(state: Matrix, action: Matrix, reward: Matrix, next_state: Matrix, absorbing: Matrix) {
      var q_current = this.Q[state, action];
      this.next_action = draw_action(next_state);

      if !absorbing {
        var q_next = this.Q[next_state, this.next_action];
      } else {
        var q_next = 0;
      }

      var delta = reward + this.env_info.gamma * q_next - q_current;
      this.e.update(state, action);
      this.Q.table += this.alpha(state, action) * delta * this.e.table;
      this.e.table += this.env_info.gamma * this.lambda;
    }

    proc episode_start() {
      this.e.reset();
      super().episode_start();
    }
  }

  class SARSALambdaContinuous: TD {
    /*
      Continuous version of the SARSA(lambda) algorithm.
    */
    proc init(approximator: Approximator, policy: Policy, env_info: ENVInfo, learning_rate: Parameter,
              lambda_coeff: real, features: AssociativeArray, approximator_params: AssociativeArray=nil) {
      /*
      Constructor.

      Args:
          lambda_coeff (real): eligibility trace coefficient.
      */
      if approximator_params == nil {
        this.approximator_params = new AssociativeArray();
      } else {
        this. approximator_params = approximator_params;
      }

      this.Q = new Regressor(approximator, this.approximator_params);
      this.e = zeros(this.Q.weights_size);
      this.lambda = lambda_coeff;
      super().init(this.Q, policy, env_info, learning_rate, features);
    }

    proc update(state: Matrix, action: Matrix, reward: Matrix, next_state: Matrix, absorbing: Matrix) {
      var phi_state = this.phi(state),
          q_current = this.Q.predict(phi_state, action),
          alpha = this.alpha(state, action);
      this.e = this.env_info.gamma * this.lambda * this.e + this.Q.diff(phi_state, action);
      this.next_action = draw_action(next_state);

      var phi_next_state = this.phi(next_state);

      if !absorbing {
        var q_next = this.Q.predict(phi_next_state, this.next_action);
      } else {
        var q_next = 0;
      }

      var delta = reward + this.env_info.gamma * q_next - q_current,
          theta = this.Q.get_weights();
      theta += alpha += alpha * delta * this.e;
      this.Q.set_weights(theta);
    }

    proc episode_start() {
      this.e = zeros(this.Q.weights_size);
      super().episode_start();
    }
  }

  class ExpectedSARSA: TD {
    /*
      Expected SARSA algorithm.

      "A theoretical and empirical analysis of Expected Sarsa". Seijen H. V. et
      al.. 2009.
    */
    proc init(policy: Policy, env_info: ENVInfo, learning_rate: Parameter) {
      this.Q = new Table(env_info.size);
      super().init(this.Q, policy, env_info, learning_rate);
    }

    proc update(state: Matrix, action: Matrix, reward: Matrix, next_state: Matrix, absorbing: Matrix) {
      var q_current = this.Q[state, action];

      if !absorbing {
        var q_next = this.Q[next_state, ..].dot(this.policy(next_state));
      } else {
        var q_next = 0;
      }

      this.Q[state, action] = q_current + this.alpha(state, action) * (reward + this.env_info.gamma * q_next - q_current);
    }
  }

  class TrueOnlineSARSALambda: TD {
    /*
      True Online SARSA(lambda) with linear function approximation.

      "True Online TD(lambda)". Seijen H. V. et al.. 2014.
    */
    proc init(policy: Policy, env_info: ENVInfo, learning_rate: Parameter, lambda_coeff: real,
              features: AssociativeArray, approximator_params: AssociativeArray=nil) {
      /*
        Constructor.

        Args:
            lambda_coeff (real): eligibility trace coefficient.
      */
      if approximator_params == nil {
        this.approximator_params = new AssociativeArray();
      } else {
        this.approximator_params = approximator_params;
      }

      this.Q = new Regressor(LinearApproximator, this.approximator_params);
      this.e = zeros(this.Q.weights_size);
      this.lambda = lambda_coeff;
      this.q_old: Regressor = nil;
      super().init(this.Q, policy, env_info, learning_rate, features);
    }

    proc update(state: Matrix, action: Matrix, reward: Matrix, next_state: Matrix, absorbing: Matrix) {
      var phi_state = this.phi(state),
          phi_state_action = get_action_features(phi_state, action, this.env_info.action_space.n),
          q_current = this.Q.predict(phi_state, action);

      if this.q_old == nil {
        this.q_old = q_current;
      }

      var alpha = this.alpha(state, action),
          e_phi = this.env_info.gamma * this.lambda * this.e + alpha * (1.0 - this.env_info.gamma * this.lambda * e_phi) * phi_state_action;
      this.next_action = draw_action(next_state);
      var phi_next_state = this.phi(next_state);

      if !absorbing {
        var q_next = this.Q.predict(phi_next_state, this.next_action);
      } else {
        var q_next = 0;
      }

      var delta = reward + this.env_info.gamma * q_next - this.q_old,
          theta = this.Q.get_weights();
      theta += delta * this.e + alpha * (this.q_old - q_current) * phi_state_action;
      this.Q.set_weights(theta);
      this.q_old = q_next;
    }

    proc episode_start() {
      this.q_old = nil;
      this.e = zeros(this.Q.weights_size);
      super().episode_start();
    }
  }

  class RLearning: TD {
    /*
      R-Learning algorithm.

      "A Reinforcement Learning Method for Maximizing Undiscounted Rewards".
      Schwartz A.. 1993.
    */
    proc init(policy: Policy, env_info: ENVInfo, learning_rate: Parameter, beta: Parameter) {
      /*
        Constructor.

        Args:
            beta (Parameter): beta coefficient.
      */
      this.Q = new Table(env_info.size);
      this.rho: real = 0.0;
      this.beta = beta;
      super().init(this.Q, policy, env_info, learning_rate);
    }

    proc update(state: Matrix, action: Matrix, reward: Matrix, next_state: Matrix, absorbing: Matrix) {
      var q_current = this.Q[state, action];

      if !absorbing {
        var q_next = max(this.Q[next_state, ..]);
      } else {
        var q_next = 0;
      }

      var delta = reward - this.rho + q_next - q_current,
          q_new = q_current + this.alpha(state, action) * delta;
      this.Q[state, action] = q_new;

      var q_max = max(this.Q[state, ..]);

      if q_new == q_max {
        delta = reward + q_next - q_max - this.rho;
        this.rho += this.beta(state, action) * delta;
      }
    }
  }

  class RQLearning: TD {
    /*
      RQ-Learning algorithm.

      "Exploiting Structure and Uncertainty of Bellman Updates in Markov Decision
      Processes". Tateo D. et al.. 2017.
    */
    proc init(policy: Policy, env_info: ENVInfo, learning_rate: Parameter,
              off_policy: bool=false, beta: Parameter=nil, delta: Parameter=nil) {
      /*
        Constructor.

        Args:
            off_policy (bool): whether to use the off policy setting or
                the online one.
            beta (Parameter): beta coefficient.
            delta (Parameter): delta coefficient.
      */
      this.off_policy = off_policy;

      if delta != nil && beta == nil {
        this.delta = delta;
        this.beta = nil;
      } else if delta == nil && beta != nil {
        this.delta = nil;
        this.beta = beta;
      } else {
        writeln("Delta or beta parameters needed.")
      }

      this.Q = new Table(env_info.size);
      this.Q_tilde = new Table(env_info.size);
      this.R_tilde = new Table(env_info.size);
      super().init(this.Q, policy, env_info, learning_rate);
    }

    proc update(state: Matrix, action: Matrix, reward: Matrix, next_state: Matrix, absorbing: Matrix) {
      var alpha = this.alpha(state, action, target=reward);
      this.R_tilde[state, action] += alpha * (reward - this.R_tilde[state, action]);

      if !absorbing {
        var q_next = next_q(next_state);

        if this.delta != nil {
          beta = alpha * this.delta(state, action, target=q_next, factor=alpha);
        } else {
          beta = this.beta(state, action, target=q_next);
        }
        
        this.Q_tilde[state, action] += beta * (q_next - this.Q_tilde[state, action]);
      }
      this.Q[state, action] = this.R_tilde[state, action] + this.env_info.gamma * this.Q_tilde[state, action];
    }

    proc next_q(next_state: Matrix) {
      /*
        Args:
            next_state (Matrix): the state where next action has to be
                evaluated.
        Returns:
            The weighted estimator value in 'next_state'.
      */
      if this.off_policy {
        return max(this.Q[next_state, ..]);
      } else {
        this.next_action = draw_action(next_state);
        return this.Q[next_state, this.next_action];
      }
    }
  }
}
