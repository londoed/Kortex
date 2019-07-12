module Kortex {
  // Numpy functions needed: any(), reshape(), max(), expand_dims(), argmax(), inf, zeros(), linalg.matrix_rank(),
  // linalg.solve(), ravel(), linalg.pinv, linalg.norm()
  // Python functions needed: //
  // tdqm functions needed: trange()
  use LinearAlgebra, Math;

  class BatchTD: Agent {
    /*
      Abstract class to implement a generic Batch TD algorithm.
    */
    proc init(approximator: Approximator, policy: Policy, env_info: ENVInfo, fit_params: AssociativeArray=nil,
              approximator_params: AssociativeArray=nil, features: AssociativeArray=nil) {
      /*
        Constructor.

        Args:
            approximator (object): approximator used by the algorithm and the
                policy.
            fit_params (AssociativeArray): parameters of the fitting algorithm of the
                approximator.
            approximator_params (AssociativeArray): parameters of the approximator to
                build.
      */
      if fit_params == nil {
        this.fit_params = new AssociativeArray();
      } else {
        this.fit_params = fit_params;
      }

      if approximator_params == nil {
        this.approximator_params = new AssociativeArray();
      } else {
        this.approximator_params = approximator_params;
      }

      this.approximator = new Regressor(approximator, this.approximator_params);
      policy.set_q(this.approximator);
      super().init(policy, env_info, features);
    }
  }

  class FQI: BatchTD {
    /*
      Fitted Q-Iteration algorithm.

      "Tree-Based Batch Mode Reinforcement Learning", Ernst D. et al.. 2005.
    */
    proc init(approximator: Approximator, policy: Policy, env_info: ENVInfo, n_iterations: int,
              fit_params: AssociativeArray=nil, approximator_params: AssociativeArray=nil,
              quiet: bool=false, boosted: bool=false) {
      /*
        Constructor.

        Args:
            n_iterations (int): number of iterations to perform for training.
            quiet (bool): whether to show the progress bar or not.
            boosted (bool): whether to use boosted FQI or not.
      */
      this.n_iterations = n_iterations;
      this.quiet = quiet;
      super().init(approximator, policy, env_info, fit_params, approximator_params);
      this.target = nil;

      // "Boosted Fitted Q-Iteration". Tosatto S. et al.. 2017.
      this.boosted = boosted;

      if this.boosted {
        this.prediction: real = 0.0;
        this.next_q: real = 0.0;
        this.idx: int = 0;
      }
    }

    proc fit(dataset: []) {
      /*
        Fit loop.
      */
      if this.boosted {
        if this.target == nil {
          this.prediction = 0.0;
          this.next_q = 0.0;
          this.idx = 0;
        }
        var fit = fit_boosted();
      } else {
        var fit = single_fit();
      }

      for _ in trange(this.n_iterations, dynamic_ncols=true, disable=this.quiet, leave=false) {
        fit(dataset);
      }
    }

    proc single_fit(x: []) {
      /*
        Single fit iteration.

        Args:
            x (Array): the dataset.
      */
      var state, action, reward, next_state, absorbing, _ = parse_dataset(x);

      if this.target == nil {
        this.target = reward;
      } else {
        var q = this.approximator.predict(next_state);

        if any(absorbing) {
          q *= 1 - absorbing.reshape(-1, 1);
        }

        var max_q = max(q, axis=1);
        this.target = reward + this.env_info.gamma * max_q;
      }
      this.approximator.fit(state, action, this.target, this.fit_params);
    }

    proc fit_boosted(x: []) {
      /*
        Single fit iteration for boosted FQI.

        Args:
            x (Array): the dataset.
      */
      var state, action, reward, next_state, absorbing, _ = parse_dataset(x);

      if this.target == nil {
        this.target = reward;
      } else {
        this.next_q += this.approximator.predict(next_state, idx=this.idx - 1);

        if any(absorbing) {
          this.next_q *= 1 - absorbing.reshape(-1, 1);
        }

        var max_q = max(this.next_q, axis=1);
        this.target = reward + this.env_info.gamma * max_q;
      }
      this.target -= this.prediction;
      this.prediction += this.target;
      this.approximator.fit(state, action, this.target, idx=this.idx, this.fit_params);
      this.idx += 1;
    }
  }

  class DoubleFQI: FQI {
    /*
      Double Fitted Q-Iteration algorithm.

      "Estimating the Maximum Expected Value in Continuous Reinforcement Learning
      Problems". D'Eramo C. et al.. 2017.
    */
    proc init(approximator: Approximator, policy: Policy, env_info: ENVInfo, n_iterations: int,
              fit_params: AssociativeArray=nil, approximator_params: AssociativeArray=nil, quiet: bool=false) {
      super().init(approximator, policy, env_info, n_iterations, fit_params, approximator_params, quiet);
    }

    proc fit(x: []) {
      var state = [],
          action = [],
          reward = [],
          next_state = [],
          absorbing = [],
          half = x.length / 2;

      for i in 0..2 {
        var s, a, r, ss, ab, _ = parse_dataset(x[i * half..(i + 1) * half]);
        state.append(s);
        action.append(a);
        reward.append(r);
        next_state.append(ss);
        absorbing.append(ab);
      }

      if this.target == nil {
        this.target = reward;
      } else {
        for i in 0..2 {
          var q_i = this.approximator.predict(next_state[i], idx=i),
              amax_q = expand_dims(argmax(q_i, axis=1), axis=1),
              max_q = this.approximator.predict(next_state[i], amax_q, idx=1 - i);

          if any(absorbing[i]) {
            max_q *= 1 - absorbing[i];
          }
          this.target[i] = reward[i] + this.env_info.gamma * max_q;
        }
      }
      for i in 0..2 {
        this.approximator.fit(state[i], action[i], this.target[i], idx=i, this.fit_params);
      }
    }
  }

  class LSPI: BatchTD {
    /*
      Least-Squares Policy Iteration algorithm.

      "Least-Squares Policy Iteration". Lagoudakis M. G. and Parr R.. 2003.
    */
    proc init(policy: Policy, env_info: ENVInfo, epsilon: real=1e-2, fit_params: AssociativeArray=nil,
              approximator_params: AssociativeArray=nil, features: AssociativeArray=nil) {
      this.epsilon = epsilon;
      var k = features.size * env_info.action_space.n;
      this.A = zeros((k, k));
      this.b = zeros((k, 1));
      super().init(LinearApproximator, policy, env_info, fit_params, approximator_params, features);
    }

    proc fit(dataset: []) {
      var phi_state, action, reward, phi_next_state, absorbing, _ = parse_dataset(dataset, this.phi),
          phi_state_action = get_action_features(phi_state, action, this.env_info.action_space.n),
          norm = INFINITY;

      while norm > this.epsilon {
        var q = this.approximator.predict(phi_next_state);

        if any(absorbing) {
          q *= 1 - absorbing.reshape(-1, 1);
        }

        var next_action = argmax(q, axis=1).reshape(-1, 1),
            phi_next_state_action = get_action_features(phi_next_state, next_action, this.env_info.action_space.n),
            tmp = phi_state_action - this.env_info.gamma * phi_next_state_action;
        this.A += phi_state_action.T.dot(tmp);
        this.b += (phi_state_action.T.dot(reward)).reshape(-1, 1);

        var old_w = this.approximator.get_weights();

        if matrix_rank(this.A) == this.A.shape[1] {
          var w = solve(this.A, this.b).ravel();
        } else {
          var w = pinv(this.A).dot(this.b).ravel();
        }
        
        this.approximator.set_weights(w);
        norm = norm(w - old_w);
      }
    }
  }
}
