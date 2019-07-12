module Kortex {
  // Numpy functions needed: zeros()
  use LinearAlgebra;

  proc numerical_diff_policy(policy: Policy, state: Matrix, action: Matrix, eps: real=1e-6) {
    /*
      Compute the gradient of a policy in (``state``, ``action``) numerically.

      Args:
          policy (Policy): the policy whose gradient has to be returned.
          state (Matrix): the state.
          action (Matrix): the action.
          eps (real): the value of the perturbation.

      Returns:
          The gradient of the provided policy in (``state``, ``action``)
          computed numerically.
    */
    var w_start = policy.get_weights(),
        g = zeros(policy.weights_size);

    for i in 0..#w_start {
      var perturb = zeros(policy.weights_size);
      perturb[i] = eps;
      policy.set_weights(w_start - perturb);
      var v1 = policy(state, action);
      policy.set_weights(w_start + perturb);
      var v2 = policy(state, action);

      g[i] = (v2 - v1) / (2 * eps);
    }
    policy.set_weights(w_start);

    return g;
  }

  proc numerical_diff_dist(dist: Distribution, theta: Matrix, eps: real=1e-6) {
    /*
      Compute the gradient of a distribution in ``theta`` numerically.

      Args:
          dist (Distribution): the distribution whose gradient has to be returned.
          theta (Matrix): the parametrization where to compute the gradient.
          eps (real): the value of the perturbation.

      Returns:
          The gradient of the provided distribution ``theta`` computed
          numerically.
    */
    var rho_start = dist.get_params(),
        g = zeros(dist.params_size);

    for i in 0..#rho_start.length {
      var perturb = zeros(dist.params_size);
      perturb[i] = eps;
      dist.set_params(rho_start - perturb);
      var v1 = dist(theta);
      dist.set_params(rho_start + perturb);
      var v2 = dist(theta);

      g[i] = (v2 - v1) / (2 * eps);
    }
    dist.set_params(rho_start);
    
    return g;
  }
}
