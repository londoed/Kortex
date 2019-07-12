module Kortex {
  // Numpy functions needed: zeros()
  use LinearAlgebra;

  class SAC: Agent {
    /*
      Stochastic Actor critic in the episodic setting as presented in:
      "Model-Free Reinforcement Learning with Continuous Action in Practice".
      Degris T. et al.. 2012.
    */
    proc init(policy: ParametricPolicy, env_info: ENVInfo, alpha_theta: Parameter, alpha_v: Parameter,
              lambda_part: real=0.9, value_function_features: Features=nil, policy_features: Features=nil) {
      /*
        Constructor.

        Args:
            policy (ParametricPolicy): a differentiable stochastic policy.
            env_info (ENVInfo): information about the MDP.
            alpha_theta (Parameter): learning rate for policy update.
            alpha_v (Parameter): learning rate for the value function.
            lambda_par (real): trace decay parameter.
            value_function_features (Features): features used by the value
                function approximator.
            policy_features (Features): features used by the policy.
      */
      this.psi = value_function_features;
      this.alpha_theta = alpha_theta;
      this.alpha_v = alpha_v;
      this.lambda = lambda_par;
      super().init(policy, env_info, policy_features);

      if this.psi != nil {
        var input_shape = (this.psi.size,);
      } else {
        var input_shape = env_info.observation_space.shape;
      }

      this.V = new Regressor(LinearApproximator, input_shape=input_shape, output_shape=(1,));
      this.e_v = zeros(this.V.weights_size);
      this.e_theta = zeros(this.policy.weights_size);
    }

    proc episode_start() {
      this.e_v = zeros(this.V.weights_size);
      this.e_theta = zeros(this.policy.weights_size);
      super().episode_start();
    }

    proc fit(dataset: []) {
      for step in dataset {
        var s, a, r, ss, absorbing, _ = step;

        if this.psi != nil {
          var s_phi = this.phi(s),
              s_psi = this.psi(s),
              ss_psi = this.psi(ss);
        } else {
          var s_phi = s,
              s_psi = s,
              ss_psi = ss;
        }

        if !absorbing {
          var v_next = this.V(ss_psi);
        } else {
          var v_next = 0;
        }

        var delta = r + this.env_info.gamma * v_next - this.V(s_psi);
        this.e_v = this.env_info.gamma * this.lambda * this.e_v + s_psi;
        this.e_theta = this.env_info.gamma * this.lambda * this.e_theta + this.policy.diff_log(s_phi, a);

        var delta_v = this.alpha_v(s, a) * delta * this.e_v,
            v_new = this.V.get_weights() + delta_v;
        this.V.set_weights(v_new);

        var delta_theta = this.alpha_theta(s, a) * delta * this.e_theta,
            theta_new = this.policy.get_weights() + delta_theta;
        this.policy.set_weights(theta_new);
      }
    }
  }

  class SAC_AVG: Agent {
    /*
      Stochastic Actor critic in the average reward setting as presented in:
      "Model-Free Reinforcement Learning with Continuous Action in Practice".
      Degris T. et al.. 2012.
    */
    proc init(policy: ParametricPolicy, env_info: ENVInfo, alpha_theta: Parameter, alpha_v: Parameter,
              alpha_r: Parameter, lambda_par: real=0.9, value_function_features: Features, policy_features: Features) {
      /*
        Constructor.

        Args:
            policy (ParametricPolicy): a differentiable stochastic policy.
            env_info (ENVInfo): information about the MDP.
            alpha_theta (Parameter): learning rate for policy update.
            alpha_v (Parameter): learning rate for the value function.
            alpha_r (Parameter): learning rate for the reward trace.
            lambda_par (real): trace decay parameter.
            value_function_features (Features): features used by the
                value function approximator.
            policy_features (Features): features used by the policy.
      */
      this.psi = value_function_features;
      this.alpha_theta = alpha_theta;
      this.alpha_v = alpha_v;
      this.alpha_r = alpha_r;
      this.lambda = lambda_par;
      super().init(policy, env_info, policy_features);

      if this.psi != nil {
        var input_shape = (this.psi.size,);
      } else {
        var input_shape = env_info.observation_space.shape;
      }
      
      this.V = new Regressor(LinearApproximator, input_shape=input_shape, output_shape=(1,));
      this.e_v = zeros(this.V.weights_size);
      this.e_theta = zeros(this.policy.weights_size);
      this.r_bar = 0;
    }

    proc episode_start() {
      this.e_v = zeros(this.V.weights_size);
      this.e_theta = zeros(this.policy.weights_size);
      super().episode_start();
    }

    proc fit(dataset: []) {
      for step in dataset {
        var s, a, r, ss, absorbing, _ = step;

        if this.phi != nil {
          var s_phi = this.phi(s);
        } else {
          var s_phi = s;
        }

        if this.psi != nil {
          var s_psi = this.psi(s),
              ss_psi = this.psi(ss);
        } else {
          var s_psi = s,
              ss_psi = s;
        }

        if !absorbing {
          var v_next = this.V(ss_phi);
        } else {
          var v_next = 0;
        }

        var delta = r - this.r_bar + v_next - this.V(s_phi);
        this.r_bar += this.alpha_r() * delta;
        this.e_v = this.lambda * this.e_v + s_psi;
        this.e_theta = this.lambda * this.e_theta + this.policy.diff_log(s_phi, a);

        var delta_v = this.alpha_v(s, a) * delta * this.e_v,
            v_new = this.V.get_weights() + delta_v;
        this.V.set_weights(v_new);

        var delta_theta = this.alpha_theta(s, a) * delta * this.e_theta,
            theta_new = this.policy.get_weights() + delta_theta;
        this.policy.set_weights(theta_new);
      }
    }
  }
}
