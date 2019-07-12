module Kortex {
  // Numpy functions needed: max(), exp(), mean(), logical_not(), isfinite()
  // Scipy functions needed: optimize.minimize()
  use LinearAlgebra;

  class BlackBoxOptimization: Agent {
    /*
      Base class for black box optimization algorithms.

      These algorithms work on a distribution of policy parameters and often they
      do not rely on stochastic and differentiable policies.
    */
    proc init(distribution: Distribution, policy: ParametricPolicy) {
      /*
        Constructor.

        Args:
          distribution (Distribution): the distribution of the policy parameters.
          policy (ParametricPolicy): the policy to use.
      */
      this.distribution = distribution;
      this.theta_list = [];
      super().init(policy, env_info, features);
    }

    proc episode_start() {
      var theta = this.distribution.sample();
      this.theta_list.append(theta);
      this.policy.set_weights(theta);
      super().episode_start();
    }

    proc fit(dataset: Matrix) {
      var Jep = compute_J(dataset, this.env_info.gamma);
      Jep = Matrix(Jep);
      var theta = Matrix(this.theta_list);
      update(Jep, theta);
      this.theta_list = [];
    }

    proc stop() {
      this.theta_list = [];
    }

    proc update(Jep: Matrix, theta: Matrix) {
      /*
        Function that implements the update routine of distribution parameters.
        Every black box algorithms should implement this function with the
        proper update.

        Args:
            Jep (Matrix): a vector containing the J of the considered
                trajectories;
            theta (Matrix): a matrix of policy parameters of the considered
                trajectories.
      */
      writeln("BlackBoxOptimization is an abstract class, update not available");
    }
  }

  class RWR: BlackBoxOptimization {
    /*
      Reward-Weighted Regression algorithm.

      "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
      Peters J.. 2013.
    */
    proc init(distribution: Distribution, policy: Policy, env_info: ENVInfo, beta: real,
              features: AssociatveArray=nil) {
      /*
        Constructor.

        Args:
            beta (real): the temperature for the exponential reward
                      transformation.
      */
      this.beta = beta;
      super().init(distribution, policy, env_info, features);
    }

    proc update(Jep: Matrix, theta: Matrix) {
      Jep -= max(Jep);
      var d = exp(this.beta * Jep);
      this.distribution.mle(theta, d);
    }
  }

  class PGPE: BlackBoxOptimization {
    /*
      Policy Gradient with Parameter Exploration algorithm.

      "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
      Peters J.. 2013.
    */
    proc init(distribution: Distribution, policy: Policy, env_info: ENVInfo, learning_rate: Parameter,
              features: AssociatveArray=nil) {
      this.learning_rate = learning_rate;
      super().init(distribution, policy, env_info, features);
    }

    proc update(Jep: Matrix, theta: Matrix) {
      var baseline_num_list = [],
          baseline_den_list = [],
          diff_log_dist_list = [];

      for i in 0..#Jep.length {
        var J_i = Jep[i],
            theta_i = theta[i],
            diff_log_dist = this.distribution.diff_log(theta_i),
            diff_log_dist2 = diff_log_dist**2;

        diff_log_dist_list.append(diff_log_dist);
        baseline_num_list.append(J_i * diff_log_dist2);
        baseline_den_list.append(diff_log_dist2);
      }

      var baseline = mean(baseline_num_list, axis=0) / mean(baseline_den_list, axis=0);
      baseline[logical_not(isfinite(baseline))] = 0.0;

      var grad_J_list = [];

      for i in 0..#Jep.length {
        diff_log_dist = diff_log_dist[i];
        J_i = Jep[i];
        grad_J_list.append(diff_log_dist * (J_i - baseline));
      }

      grad_J = mean(grad_J_list, axis=0);
      var omega = this.distribution.get_params();

      omega += this.learning_rate(grad_J) * grad_J;
      this.distribution.set_params(omega);
    }
  }

  class REPS: BlackBoxOptimization {
    /*
      Episodic Relative Entropy Policy Search algorithm.

      "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
      Peters J.. 2013.
    */
    proc init(distribution: Distribution, policy: Policy, env_info: ENVInfo, eps: real,
              features: AssociatveArray=nil) {
      /*
        Constructor.

        Args:
            eps (real): the maximum admissible value for the Kullback-Leibler
                divergence between the new distribution and the
                previous one at each update step.
      */
      this.eps = eps;
      super().init(distribution, policy, env_info, features);
    }

    proc update(Jep: Matrix, theta: Matrix) {
      var eta_start = ones(1),
          res = minimize(REPS.dual_function, eta_start,
                         jac=REPS.dual_function_diff,
                         bounds=((finfo(real).eps, inf),),
                         args=(this.eps, Jep, theta)),
          eta_opt = asscalar(res.x);

      Jep -= max(Jep);
      var d = exp(Jep / eta_opt);
      this.distribution.mle(theta, d);
    }

    proc dual_function(eta_array: Matrix, args: []) {
      var eta = asscalar(eta_array);
      eps, Jep, theta = args;
      var max_J = max(Jep),
          r = Jep - max_j;
          sum1 = mean(exp(r / eta));
      return eta * eps + eta * log(sum1) + max_J;
    }

    proc dual_function_diff(eta_array: Matrix, args: []) {
      var eta = asscalar(eta_array);
      eta, Jep, theta = args;
      var max_J = max(Jep),
          r = Jep - max_J,
          sum1 = mean(exp(r / eta)),
          sum2 = mean(exp(r / eta) * r),
          gradient = eps + log(sum1) - sum2 / (eta * sum1);
      return Matrix(gradient);
    }
  }
}
