module Kortex {
  // Numpy functions needed: sqrt()
  // Python functions needed: math.ceil()
  use LinearAlgebra, Math;

  proc adam(grad_fn) {
    /*
      Adaptive Momentum Estimation (Adam) algorithm.

      "Adam: A Method for Stochasitc Optimization".
      Kingma, D. P. et al.. 2014.

      Args:
        grad_fn (function): the gradient function to optimize.
    */
    var beta1: real = 0.9,
        beta2: real = 0.999,
        alpha: real = 0.01,
        eps_stable: real = 1e-8,
        theta_0: int = 0,
        m_t: int = 0,
        v_t: int = 0,
        t: int = 0;
    while (1) {
      t += 1;
      var g_t = grad_fn(theta_0);
      m_t = beta1 * m_t + (1 - beta1) * g_t;
      v_t = beta2 * v_t + (1 - beta2) * (g_t**2);
      var m_cap = m_t / (1 - beta1**t),
          v_cap = v_t / (1 - beta2**t),
          theta_0_prev = theta_0;
      theta_0 = theta_0 - (alpha * m_cap) / (sqrt(v_cap) + eps_stable);
      if theta_0 == theta_0_prev {
        break;
      }
    }
  }

  proc sgd(grad_func, theta_0: real, num_iters: int) {
    /*
      Stochasitc Gradient Descent algorithm.

      "A Stochastic Approximation Method".
      Robbins, H .. 1951.

      Args:
          grad_fn (function): the function to optimize.
          theta_0 (real): the initial point to start SGD from.
          num_iters (int): total iterations to run SGD.
    */
    var start_iter = 0,
        theta = theta_0;
    for iteration in (start_iter + 1)..(num_iters + 1) {
      _, grad = grad_fn(theta);
      theta = theta - (alpha * grad);
    }
    return theta;
  }

  proc adagrad(grad_fn, theta_0: real, data: [], args: [], step_size: real=1e-2, fudge_factor: real=1e-6,
               max_iters: int=1000, mini_batch_size: int, mini_batch_ratio: real=0.01) {
    /*
      Adagrad optimization algorithm.

      "Adaptive Subgradient Methods fro Online Learning and Stochasitc Optimization".
      Duchi, J. et al.. 2011.

      Args:
          grad_fn (function): the function to optimize.
          theta_0 (real): the initial parameters for optimization.
          data (Array): list of training data.
          args (Array): list of additional arguments passed to grad_fn.
          step_size (real): the global stepsize for optimization.
          fudge_factor (real): small number to counter numerical instability.
          max_iters (int): number of iterations to run.
          mini_batch_size (int): number of training samples considered for
              each iteration.
          mini_batch_ratio (real): if mini_batch_size is not set, this ratio
              will be used to determine the batch size dependent on the length
              of the training data.
    */
    var gti = zeros(theta_0.shape[0]),
        ld = data.length,
        w = theta_0;
    if mini_batch_size == nil {
      mini_batch_size = (ceil(data.length * mini_batch_ratio));
    }

    for t in 0..max_iters {
      var s = sample(0..ld, mini_batch_size),

      for idx in s {
        var sd = [data[idx]];
      }
      var grad = grad_fn(w, sd, args);
      gti += grad**2;
      var adjusted_grad = grad / (fudge_factor + sqrt(gti));
      w = w - step_size * adjusted_grad;
    }
    return w;
  }
}
