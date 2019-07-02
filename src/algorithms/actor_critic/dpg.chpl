module Kortex {
  // Numpy functions needed: asscalar(), atleast_2d()
  use LinearAlgebra;

  class COPDAC_Q: Agent {
    /*
      Compatible off-policy deterministic actor-critic algorithm.
      "Deterministic Policy Gradient Algorithms". Silver D. et al.. 2014.
    */
    proc init(policy: Policy: ParametricPolicy, mu, env_info: ENVInfo,
              alpha_theta: Parameter, alpha_omega: Parameter,
              alpha_v: Parameter, value_function_features: Features=nil,
              policy_features: Features=nil) {
      this.mu = mu;
      this.psi = value_function_features;
      this.alpha_theta = alpha_theta;
      this.alpha_omega = alpha_omega;
      this.alpha_v = alpha_v;

      if this.psi != nil {
        var input_shape: Tuple = (this.psi.size,);
      } else {
        var input_shape: Tuple = env_info.observation_space.shape;
      }

      this.V = new Regressor(LinearApproximator, input_shape: Tuple=input_shape,
                             output_shape: Tuple=(1,));
      this.A = new Regressor(LinearApproximator, input_shape: Tuple=(this.mu.weights_size,),
                             output_shape: Tuple=(1,));
      super().init(policy, env_info, policy_features);
    }

    proc fit(dataset: []) {
      for step in dataset {
        var s, a, r, ss, absorbing, _ = step;

        if this.phi != nil {
          var s_phi = this.phi(s);
        } else {
          var s_phi = 0;
        }

        if this.psi != nil {
          var s_psi = this.psi(s),
              ss_psi = this.psi(ss);
        } else {
          var s_psi = s,
              ss_psi = ss;
        }

        if !absorbing {
          var q_next = asscalar(this.V(ss_psi));
        } else {
          var q_next = 0;
        }

        var grad_mu_s = atleast_2d(this.mu.diff(s_phi)),
            omega = this.A.get_weights(),
            delta = r + this.env_info.gamma * q_next - this.Q(s, a),
            delta_theta = this.alpha_theta(s, a) * omega.dot(grad_mu_s.T).dot(grad_mu_s),
            delta_omega = this.alpha_omega(s, a) * delta * nu(s, a),
            delta_v = this.alpha_v(s, a) * delta * s_phi,
            theta_new = this.mu.get_weights() + delta_theta;
        this.mu.set_weights(theta_new);

        var omega_new = omega + delta_omega;
        this.A.set_weights(omega_new);

        var v_new = this.V.get_weights() + delta_v;
        this.V.set_weights(v_new);
      }
    }

    proc Q(state: Matrix, action: Matrix) {
      if this.psi != nil {
        var state_psi = this.psi(state);
      } else {
        var state_psi = state;
      }
      return asscalar(this.V(state_psi)) + asscalar(this.A(nu(state, action)));
    }

    proc nu(state: Matrix, action: Matrix) {
      if this.phi != nil {
        var state_phi = this.phi(state);
      } else {
        var state_phi = state;
      }

      var grad_mu = atleast_2d(this.mu.diff(state_phi)),
          delta = action - this.mu(state_phi);
      return delta.dot(grad_mu);
    }
  }
}
