module Kortex {
  // Numpy functions needed: linalg.inv(), random.multivariate_normal(), mean(), std(), empty(), cov(), tril_indices()
  // Scipy functions needed: multivariate_normal,
  // Python functions needed: //

  use LinearAlgebra;

  class GaussianDistribution: Distribution {
    /*
      This class implements a gaussian distribution with fixed covariance matrix.
      The parameters vector represents only the mean.
    */
    proc init(mu: Matrix, sigma: Matrix) {
      this.mu = mu;
      this.sigma = sigma;
      this.inv_sigma = inv(sigma);
    }

    proc sample() {
      return multivariate_normal(this.mu, this.sigma);
    }

    proc log_pdf(theta: Matrix) {
      return logpdf(theta, this.mu, this.sigma);
    }

    proc call(theta: Matrix) {
      return pdf(theta, this.mu, this.sigma);
    }

    proc mle(theta: Matrix, weights: Matrix=nil) {
      if weights == nil {
        this.mu = mean(theta, axis=0);
      } else {
        this.mu = weights.dot(theta) / sum(weights);
      }
    }

    proc diff_log(theta: Matrix) {
      var delta = theta - this.mu;,
          g = this.inv_sigma.dot(delta);
      return g;
    }

    proc get_params() {
      return this.mu;
    }

    proc set_params(rho: Matrix) {
      this.mu = rho;
    }

    proc params_size() {
      return this.mu.length;
    }
  }

  class GaussianDiagonalDistribution: Distribution {
    /*
      This class implements a gaussian distribution with diagonal covariance matrix.
      The parameters vector represents the mean and the standard deviation.
    */
    proc init(mu: Matrix, std: Matrix) {
      assert(std.shape.length == 1);
      this.mu = mu;
      this.std = std;
    }

    proc sample() {
      var sigma = diag(this.std**2);
      return multivariate_normal(this.mu, sigma);
    }

    proc log_pdf(theta: Matrix) {
      var sigma = diag(this.std**2);
      return logpdf(theta, this.mu, sigma);
    }

    proc call(theta: Matrix) {
      var sigma = diag(this.std**2);
      return pdf(theta, this.mu, sigma);
    }

    proc mle(theta: Matrix, weights: Matrix=nil) {
      if weights == nil {
        this.mu = mean(theta, axis=0);
        this.std = std(theta, axis=0);
      } else {
        var sum_d = sum(weights),
            sum_d2 = sum(weights**2),
            Z = sum_d - sum_d2 / sum_d;
        this.mu = weights.dot(theta) / sum_d;
        var delta2 = (theta - this.mu)**2;
        this.std = sqrt(weights.dot(delta2) / Z);
      }
    }

    proc diff_log(theta: Matrix) {
      var n_dims = this.mu.length,
          sigma = this.std**2,
          g = empty(this.params_size),
          delta = theta - this.mu,
          g_mean = delta / sigma,
          g_cov = delta**2 / (this.std**3) - 1 / this.std;
      g[..n_dims] = g_mean;
      g[n_dims..] = g_cov;

      return g;
    }

    proc get_params() {
      var rho = empty(this.params_size),
          n_dims = this.mu.length;
      rho[..n_dims] = this.mu;
      rho[n_dims..] = this.std;
      return rho;
    }

    proc set_params(rho: Matrix) {
      var n_dims = this.mu.length;
      this.mu = rho[..n_dims];
      this.std = rho[n_dims..];
    }

    proc params_size() {
      return 2 * this.mu.length;
    }
  }

  class GaussianCholeskyDistribution: Distribution {
    /*
      Gaussian distribution with full covariance matrix. The parameters
      vector represents the mean and the Cholesky decomposition of the
      covariance matrix. This parametrization enforce the covariance matrix to be
      positive definite.
    */
    proc init(mu: Matrix, sigma: Matrix) {
      this.mu = mu;
      this.chol_sigma = cholesky(sigma);
    }

    proc sample() {
      var sigma: Matrix = this.chol_sigma.dot(this.chol_sigma.T);
      return multivariate_normal(this.mu, sigma);
    }

    proc log_pdf(theta: Matrix) {
      var sigma: Matrix = this.chol_sigma.dot(this.chol_sigma.T);
      return logpdf(theta, this.mu, sigma);
    }

    proc call(theta: Matrix) {
      var sigma: Matrix = this.chol_sigma.dot(this.chol_sigma.T);
      return pdf(theta, this.mu, sigma);
    }

    proc mle(theta: Matrix, weights: Matrix=nil) {
      if weights == nil {
        this.mu = mean(theta, axis=0);
        var sigma: Matrix = cov(theta);
      } else {
        var sum_d = sum(weights),
            sum_d2 = sum(weights**2),
            Z = sum_d - sum_d2 / sum_d;
        this.mu = weights.dot(theta) / sum_d;
        var delta = theta - this.mu;
        sigma = delta.T.dot(diag(weights)).dot(delta) / Z;
      }
      this.chol_sigma = cholesky(sigma);
    }

    proc diff_log(theta: Matrix) {
      var n_dims: int = this.mu.length,
          inv_chol = inv(this.chol_sigma),
          inv_sigma = inv_chol.T.dot(inv_chol),
          g = empty(this.params_size);

      var delta = theta - this.mu,
          g_mean = inv_sigma.dot(delta),
          delta_a = reshape(delta, (-1, 1)),
          delta_b = reshape(delta, (1, -1));

      var S = inv_chol.dot(delta_a).dot(delta_b).dot(inv_sigma),
          g_cov = S - diag(diag(inv_chol));
      g[..n_dims] = g_mean;
      g[n_dims..] = g_cov.T[tril_indices(n_dims)];

      return g;
    }

    proc get_params() {
      var rho: Matrix = empty(this.params_size);
          n_dims: int = this.mu.length;
      rho[..n_dims] = this.mu;
      rho[n_dims..] = this.chol_sigma[tril_indices(n_dims)];
      
      return rho;
    }

    proc set_params(rho: Matrix) {
      var n_dims: int = this.mu.length;
      this.mu = rho[..n_dims];
      this.chol_sigma[tril_indices(n_dims)] = rho[n_dims..];
    }

    proc params_size() {
      var n_dims: int = this.mu.length;
      return 2 * n_dims + (n_dims**2 - n_dims) / 2;
    }
  }
}
