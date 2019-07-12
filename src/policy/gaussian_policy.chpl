module Kortex {
  // Numpy functions needed: linalg.inv(), random.multivariate_normal(), expand_dims(),
  // reshape(), concatenate(), diag(), atleast_2d(), sum(), exp()
  // Scipy functions needed: multivariate_normal()
  use LinearAlgebra;

  class GaussianPolicy: ParametricPolicy {
    /*
      Gaussian policy.
      This is a differentiable policy for continuous action spaces.
      The policy samples an action in every state following a gaussian
      distribution, where the mean is computed in the state and the covariance
      matrix is fixed.
    */
    proc init(mu: Regressor, sigma: Matrix) {
      /*
        Constructor.

        Args:
            mu (Regressor): the regressor representing the mean w.r.t. the
                state.

            sigma (Matrix): a square positive definite matrix representing
                the covariance matrix. The size of this matrix must be n x n,
                where n is the action dimensionality.
      */
      this.approximator = mu;
      this.inv_sigma = inv(sigma);
      this.sigma = sigma;
    }

    proc set_sigma(sigma: Matrix) {
      /*

        Setter.

        Args:
            sigma (np.ndarray): the new covariance matrix. Must be a square
                      positive definite matrix.
      */
      this.sigma = sigma;
      this.inv_sigma = inv(sigma);
    }

    proc call(state: Matrix, action: Matrix) {
      var mu, sigma, _ = compute_multivariate_gaussian(state);
      return multivariate_normal.pdf(action, mu, sigma);
    }

    proc draw_action(state: Matrix) {
      var mu, sigma, _ = compute_multivariate_gaussian(state);
      return random.multivariate_normal(mu, sigma);
    }

    proc diff_log(state: Matrix, action: Matrix) {
      var mu, _, inv_sigma = compute_multivariate_gaussian(state),
          delta = action - mu,
          j_mu = this.approximator.diff(state);

      if j_mu.shape.length == 1 {
        j_mu = expand_dims(j_mu, axis=1);
      }

      var g = 0.5 * j+mu.dot(inv_sigma + inv_sigma.T).dot(delta.T);
      return g;
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

    proc compute_multivariate_gaussian(state: Matrix) {
      var mu = reshape(this.approximator.predict(expand_dims(state, axis=0)), -1);
      return mu, this.sigma, this.inv_sigma;
    }
  }

  class DiagonalGaussianPolicy: ParametricPolicy {
    /*
      Gaussian policy with learnable standard deviation.

      The Covariance matrix is constrained to be a diagonal
      matrix, where the diagonal is the squared standard
      deviation vector. This is a differentiable policy for
      continuous action spaces. This policy is similar to
      the gaussian policy, but the weights includes also
      the standard deviation.
    */
    proc init(mu: Regressor, std: Matrix) {
      /*
        Constructor.

        Args:
            mu (Regressor): the regressor representing the mean w.r.t. the
                state;
            std (Matrix): a vector of standard deviations. The length of
                this vector must be equal to the action dimensionality.
      */
      this.approximator = mu;
      this.std = std;
    }

    proc set_std(std: Matrix) {
      /*
        Setter.

        Args:
            std (Matrix): the new standard deviation. Must be a square
                positive definite matrix.
      */
      this.std = std;
    }

    proc call(state: Matrix, action: Matrix) {
      var mu, sigma, _ = compute_multivariate_gaussian(state);
      return multivariate_normal.pdf(action, mu, sigma);
    }

    proc draw_action(state: Matrix) {
      var mu, sigma, _ = compute_multivariate_gaussian(state);
      return random.multivariate_normal(mu, sigma);
    }

    proc diff_log(state: Matrix, action: Matrix) {
      var mu, _, inv_sigma = compute_multivariate_gaussian(state),
          delta = action - mu,
          // Compute mean derivative
          j_mu = this.approximator.diff(state);

      if j_mu.shape.length == 1 {
        j_mu = expand_dims(j_mu, axis=1);
      }

      var g_mu = 0.5 * j_mu.dot(inv_sigma + inv_sigma.T).dot(delta.T),
          // Compute standard deviation derivative
          g_sigma = -1.0 / this.std + delta**2 / this.std**3;
      return concatenate((g_mu, g_sigma), axis=0);
    }

    proc set_weights(weights: Matrix) {
      this.approximator.set_weights(weights[0..this.approximator.weights_size]);
      this.std = weights[this.approximator.weights_size..];
    }

    proc get_weights() {
      return concatenate((this.approximator.get_weights(), this.std), axis=0);
    }

    proc weights_size() {
      return this.approximator.weights_size + this.std.size;
    }

    proc compute_multivariate_gaussian(state: Matrix) {
      var mu = reshape(this.approximator.predict(expand_dims(state, axis=0)), -1),
          sigma = this.std**2;
      return mu, diag(sigma), diag(1.0 / sigma);
    }
  }

  class StateStdGaussianPolicy: ParametricPolicy {
    /*
      Gaussian policy with learnable standard deviation.

      The Covariance matrix is constrained to be a diagonal matrix, where
      the diagonal is the squared standard deviation, which is computed for
      each state. This is a differentiable policy for continuous action spaces.
      This policy is similar to the diagonal gaussian policy, but a parametric
      regressor is used to compute the standard deviation, so the standard
      deviation depends on the current state.
    */
    proc init(mu: Regressor, std: Regressor, eps: real=1e-6) {
      /*
        Constructor.

        Args:
            mu (Regressor): the regressor representing the mean w.r.t. the
                state.
            std (Regressor): the regressor representing the standard
                deviations w.r.t. the state. The output dimensionality of the
                regressor must be equal to the action dimensionality.
            eps (real): A positive constant added to the variance to
                ensure that is always greater than zero.
      */
      assert(eps > 0);
      this.mu_approximator = mu;
      this.std_approximator = std;
      this.eps = eps;
    }

    proc call(state: Matrix, action: Matrix) {
      var mu, sigma, _ = compute_multivariate_gaussian(state);
      return multivariate_normal.pdf(action, mu, sigma);
    }

    proc draw_action(state: Matrix) {
      var mu, sigma, _ = compute_multivariate_gaussian(state);
      return random.multivariate_normal(mu. sigma);
    }

    proc diff_log(state: Matrix, action: Matrix) {
      var mu, sigma, std = compute_multivariate_gaussian(state),
          diag_sigma = diag(sigma),
          delta = action - mu,
          j_mu = this.mu_approximator.diff(state);

      if j_mu.shape.length == 1 {
        j_mu = expand_dims(j_mu, axis=1);
      }

      var sigma_inv = diag(1/ diag_sigma),
          g_mu = j_mu.dot(sigma_inv).dot(delta.T),
          // Compute variance derivative
          w = (delta**2 - diag_sigma) * std / diag_sigma**2,
          j_sigma = atleast_2d(this.std_approximator.diff(state).T),
          g_sigma = atleast_2d(w.dot(j_sigma));
      return concatenate((g_mu, g_sigma), axis=0);
    }

    proc set_weights(weights: Matrix) {
      var mu_weights = weights[0..this.mu_approximator.weights_size],
          std_weights = weights[this.mu_approximator.weights_size..];
      this.mu_approximator.set_weights(mu_weights);
      this.std_approximator.set_weights(std_weights);
    }

    proc get_weights() {
      var mu_weights = this.mu_approximator.get_weights(),
          std_weights = this.std_approximator.get_weights();
      return concatenate((mu_weights, std_weights), axis=0);
    }

    proc weights_size() {
      return this.mu_approximator.weights_size + this.std_approximator.weights_size;
    }

    proc compute_multivariate_gaussian(state: Matrix) {
      var mu = reshape(this.mu_approximator.predict(expand_dims(state, axis=0)), -1),
          std = reshape(this.std_approximator.predict(expand_dims(state, axis=0)), -1),
          sigma = std**2 + this.eps;
      return mu, diag(sigma), std;
    }
  }

  class StateLogStdGaussianPolicy: ParametricPolicy {
    /*
      Gaussian policy with learnable standard deviation.

      The Covariance matrix is constrained to be a diagonal matrix, the diagonal
      is computed by an exponential transformation of the logarithm of the
      standard deviation computed in each state. This is a differentiable policy
      for continuous action spaces. This policy is similar to the State std
      gaussian policy, but here the regressor represents the logarithm of the
      standard deviation.
    */
    proc init(mu: Regressor, log_std: Regressor) {
      /*
        Constructor.

        Args:
            mu (Regressor): the regressor representing the mean w.r.t. the
                state.
            log_std (Regressor): a regressor representing the logarithm of the
                variance w.r.t. the state. The output dimensionality of the
                regressor must be equal to the action dimensionality.
      */
      this.mu_approximator = mu;
      this.log_std_approximator = log_std;
    }

    proc call(state: Matrix, action: Matrix) {
      var mu, sigma = compute_multivariate_gaussian(state);
      return multivariate_normal.pdf(action, mu, sigma);
    }

    proc draw_action(state: Matrix) {
      var mu, sigma = compute_multivariate_gaussian(state);
      return random.multivariate_normal(mu, sigma);
    }

    proc diff_log(state: Matrix, action: Matrix) {
      var mu, sigma = compute_multivariate_gaussian(state),
          diag_sigma = diag(sigma),
          delta = action - mu,
          // Compute mean derivative
          j_mu = this.mu_approximator.diff(state);

      if j_mu.shape.length == 1 {
        j_mu = expand_dims(j_mu, axis=1);
      }
      
      var sigma_inv = diag(1 / diag_sigma),
          g_mu = j_mu.dot(sigma_inv).dot(delta.T),
          // Compute variance derivative
          w = delta**2 / diag_sigma,
          j_sigma = atleast_2d(this.log_std_approximator.diff(state).T),
          g_sigma = atleast_2d(w.dot(j_sigma)) - sum(j_sigma, axis=0);
      return concatenate((g_mu, g_sigma), axis=0);
    }

    proc set_weights(weights: Matrix) {
      var mu_weights = weights[0..this.mu_approximator.weights_size],
          log_std_weights = weights[this.mu_approximator.weights_size..];
      this.mu_approximator.set_weights(mu_weights);
      this.log_std_approximator.set_weights(log_std_weights);
    }

    proc get_weights() {
      var mu_weights = this.mu_approximator.get_weights(),
          log_std_weights = this.log_std_approximator.get_weights();
      return concatenate((mu_weights, log_std_weights), axis=0);
    }

    proc weights_size() {
      return this.mu_approximator.weights_size + this.log_std_approximator.weights_size;
    }

    proc compute_multivariate_gaussian(state: Matrix) {
      var mu = reshape(this.mu_approximator.predict(expand_dims(state, axis=0)), -1),
          log_std = reshape(this.log_std_approximator.predict(expand_dims(state, axis=0)), -1),
          sigma = exp(log_std)**2;
      return mu, diag(sigma);
    }
  }
}
