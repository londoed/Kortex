module Kortex {
  // Numpy functions needed: zeros(), random.randn(), where(), size, random.choice(), concatenate(), argsort(),
  //   copy(), std()
  use LinearAlgebra;

  class OpenES {
    proc init(num_params: int, sigma_init: real=0.1, sigma_decay: real=0.999, sigma_limit: real=0.01,
              learning_rate: real=0.01, learning_rate_decay: real=0.9999, learning_rate_limit: real=0.0001,
              pop_size: int=256, antithetic: bool=false, weight_decay: real=0.01, rank_fitness: bool=true,
              forget_best: bool=true) {
      this.num_params = num_params;
      this.sigma_decay = sigma_decay;
      this.sigma = sigma_init;
      this.sigma_limit = sigma_limit;
      this.learning_rate = learning_rate;
      this.learning_rate_decay = learning_rate_decay;
      this.learning_rate_limit = learning_rate_limit;
      this.pop_size = pop_size;
      this.antithetic = antithetic;

      if this.antithetic {
        assert(this.pop_size % 2 == 0);
        this.half_pop_size = (this.pop_size / 2): int;
      }

      this.reward = zeros(this.pop_size);
      this.mu = zeros(this.num_params);
      this.best_mu = zeros(this.num_params);
      this.best_reward = 0;
      this.first_iter = true;
      this.forget_best = forget_best;
      this.weight_decay = weight_decay;
      this.rank_fitness = rank_fitness;

      if this.rank_fitness {
        this.forget_best = true;
      }

      this.optimizer = new Adam(learning_rate);
    }

    proc rms_stdev() {
      var sigma = this.sigma;
      return mean(sqrt(sigma**2));
    }

    proc ask() {
      if this.antithetic {
        this.epsilon_half = randn(this.half_pop_size, this.num_params);
        this.epsilon = concatenate([this.epsilon_half, -this.epsilon_half]);
      } else {
        this.epsilon = randn(this.pop_size, this.num_params);
      }

      this.solutions = this.mu.reshape(1, this.num_params) + this.epsilon * this.sigma;

      return this.solutions;
    }

    proc tell(reward_table_result: []) {
      assert(reward_table_result.length == this.pop_size);

      var reward = Matrix(reward_table_result);

      if this.rank_fitness {
        reward = compute_centered_ranks(reward);
      }

      if this.weight_decay > 0 {
        var l2_decay = compute_weight_decay(this.weight_decay, this.solutions);
        reward += l2_decay;
      }

      var idx = argsort(reward)[.., .., -1],
          best_reward = reward[idx[0]],
          best_mu = this.solutions[idx[0]];
      this.current_best_reward = best_reward;
      this.current_best_mu = best_mu;

      if this.first_iter {
        this.first_iter = false;
        this.best_reward = this.current_best_reward;
        this.best_mu = best_mu;
      } else {
        if this.forget_best || (this.current_best_reward > this.best_reward) {
          this.best_mu = best_mu;
          this.best_reward = this.current_best_reward;
        }
      }

      var normalized_reward = (reward - mean(reward) / std(reward)),
          change_mu = 1.0 / (this.pop_size * this.sigma) * this.epsilon.T.dot(normalized_reward);

      this.optimizer.stepsize = this.learning_rate;
      var update_ratio = this.optimizer.update(-change_mu);

      if this.sigma > this.sigma_limit {
        this.sigma *= this.sigma_decay;
      }

      if this.learning_rate > this.learning_rate_limit {
        this.learning_rate *= this.learning_rate_decay;
      }
    }

    proc current_param() {
      return this.current_best_mu;
    }

    proc set_mu(mu: []) {
      this.mu = Matrix(mu);
    }

    proc best_param() {
      return this.best_mu;
    }

    proc result() {
      return new Tuple(this.best_mu, this.best_reward, this.current_best_reward, this.sigma);
    }
  }
}
