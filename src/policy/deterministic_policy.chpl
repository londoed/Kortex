module Kortex {
  // Numpy functions needed: array_equal()
  use LinearAlgebra;

  class DeterministicPolicy: ParametricPolicy {
    /*
      Simple parametric policy representing a deterministic policy. As
      deterministic policies are degenerate probability functions where all
      the probability mass is on the deterministic action, they are not
      differentiable, even if the mean value approximator is differentiable.
    */
    proc init(mu: Regressor) {
      /*
        Constructor.

        Args:
          mu (Regressor): the regressor representing the action to select
              in each state.
      */
      this.approximator = mu;
    }

    proc get_regressor() {
      /*
        Getter.

        Returns:
          The regressor that is used to map state to action/
      */
      return this.approximator;
    }

    proc call(state: Matrix, action: Matrix) {
      var policy_action = this.approximator.predict(state);
      if array_equal(action, policy_action) {
        return 1.0;
      } else {
        return 0.0;
      }

      proc draw_action(state: Matrix) {
        return this.approximator.predict(state);
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
    }
  }
}
