module Kortex {
  // Numpy functions needed: random.normal(), sqrt(), zeros()
  use LinearAlgebra;

  class OrnsteinUhlenbeckPolicy: ParametricPolicy {
    /*
      Ornstein-Uhlenbeck process as implemented in:
      https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py.

      This policy is commonly used in the Deep Deterministic Policy Gradient
      algorithm.
    */
    proc init(mu: Regressor, sigma: Matrix, theta: real, dt: real, x0: Matrix=nil) {
      /*
        Constructor.

        Args:
            mu (Regressor): the regressor representing the mean w.r.t. the
                state.
            sigma (Matrix): average magnitude of the random flactations per
                square-root time.
            theta (real): rate of mean reversion.
            dt (real): time interval.
            x0 (Matrix): initial values of noise.
        */
        this.approximator = mu;
        this.sigma = sigma;
        this.theta = theta;
        this.dt = dt;
        this.x0 = x0;
    }

    proc call(state: Matrix, action: Matrix) {
      writeln("Call function not implemented.")
    }

    proc draw_action(state: Matrix) {
      mu = this.approximator.predict(state);
      var x = this.x_prev - this.theta * this.x_prev * this.dt + this.sigma * sqrt(this.dt) * random_normal(size=this.approximator.output_shape);
      this.x_prev = x;
      return mu + x;
    }

    proc set_weights(weights: Matrix) {
      this.approximator.set_weights(weights);
    }

    proc get_weights() {
      return this.approximator.get_weights();
    }

    proc weights_size() {
      return this.approximator.weights_size;
    }

    proc reset() {
      if this.x0 != nil {
        this.x_prev = this.x0;
      } else {
        this.x_prev = zeros(this.approximator.output_shape);
      }
    }
  }
}
