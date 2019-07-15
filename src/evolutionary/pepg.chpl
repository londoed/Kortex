module Kortex {
  // Numpy functions needed: zeros(), random.randn(), where(), size, random.choice(), concatenate(), argsort(),
  //   copy(), std()
  use LinearAlgebra;

  class PEPG {
    proc init(num_params: int, sigma_init: real=0.1, sigma_alpha: real=0.2, sigma_decay: real=0.999, sigma_limit: real=0.01,
              sigma_max_change: real=0.2, learning_rate: real=0.01, learning_rate_decay: real=0.9999, learning_rate_limit: real=0.01,
              elite_ratio: int=0, pop_size: int=256, avg_baseline: bool=true, weight_decay: real=0.01, rank_fitness: bool=true,
              forget_best: bool=true) {
      this.num_params = num_params;
      this.sigma_init = sigma_init;
      this.sigma_decay = sigma_decay;
      this.sigma_max_change = sigma_max_change;
      this.learning_rate = learning_rate;
      this.learning_rate_decay = learning_rate_decay;
      this.learning_rate_limit = learning_rate_limit;
      this.pop_size = pop_size;
      this.avg_baseline = avg_baseline;

      if this.avg_baseline {
        assert(this.pop_size % 2 == 0);
        this.batch_size = (this.pop_size / 2): int;
      } else {
        assert(this.pop_size & 1);
        this.batch_size = ((this.pop_size - 1) / 2): int;
      }

      this.elite_ratio = elite_ratio;
      this.elite_pop_size = (this.pop_size * this.elite_ratio): int;
      this.use_elite = false;

      if this.elite_pop_size > 0 {
        this.use_elite = true;
      }

      this.forget_best = forget_best;
      this.batch_reward = zeros(this.batch_size * 2);
      this.mu = zeros(this.num_params);
      this.sigma = ones(this.num_params) * this.sigma_init;
      this.current_best_mu = zeros(this.num_params);
      this.best_mu = 0;
      this.first_iter = true;
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
      this.epsilon = randn(this.batch_size, this.num_params) * this.sigma.reshape(1, this.num_params);
      this.epsilon_full = concatenate([this.epsilon, -this.epsilon]);

      if this.avg_baseline {
        var epsilon = this.epsilon_full;
      } else {
        var epsilon = concatenate([zeros((1, this.num_params)), this.epsilon_full]);
      }

      var solutions = this.mu.reshape(1, this.num_params) + epsilon;
      this.solutions = solutions;

      return solutions;
    }

    proc tell(reward_table_result: []) {
      assert(reward_table_result.length == this.pop_size);

      var reward_table = Matrix(reward_table_result);

      if this.rank_fitness {
        reward_table = compute_centered_ranks(reward_table);
      }

      if this.weight_decay > 0 {
        var l2_decay = compute_weight_decay(this.weight_decay, this.solutions);
        reward_table += l2_decay;
      }

      var reward_offset: int = 1;

      if this.avg_baseline {
        var b = mean(reward_table);
        reward_offset = 0;
      } else {
        var b = reward_table[0];
      }

      var reward = reward_table[reward_offset..];

      if this.use_elite {
        var idx = argsort(reward)[.., .., -1][0..this.elite_pop_size];
      } else {
        var idx = argsort(reward)[.., .., -1];
      }

      var best_reward = reward[idx[0]];

      if best_reward > b || this.avg_baseline {
        var best_mu = this.mu + this.epsilon_full[idx[0]],
            best_reward = reward[idx[0]];
      } else {
        var best_mu = this.mu,
            best_reward = b;
      }

      this.current_best_reward = best_reward;
      this.current_best_mu = best_mu;

      if this.first_iter {
        this.sigma = ones(this.num_params) * this.sigma_init;
        this.first_iter = false;
        this.best_reward = this.current_best_reward;
        this.best_mu = best_mu;
      } else {
        if this.forget_best || this.current_best_reward > this.best_reward {
          this.best_mu = best_mu;
          this.best_reward = this.current_best_reward;
        }
      }

      var epsilon = this.epsilon,
          sigma = this.sigma;

      if this.use_elite {
        this.mu += this.epsilon_full[idx].mean(axis=0);
      } else {
        var rT = (reward[..this.batch_size] - reward[this.batch_size..]),
            change_mu = dot(rT, epsilon);
        this.optimizer.stepsize = this.learning_rate;
        var update_ratio = this.optimizer.update(-change_mu);
      }

      if this.sigma_alpha > 0 {
        var stdev_reward: real = 1.0;

        if !this.rank_fitness {
          stdev_reward = reward.std();
        }

        var S = ((epsilon**2 - (sigma**2).reshape(1, this.num_params)) / sigma.reshape(1, this.num_params)),
            reward_avg = (reward[..this.batch_size] + reward[this.batch_size..]) / 2.0,
            rS = reward_avg - b,
            delta_sigma = rS.dot(S) / (2 * this.batch_size * stdev_reward),
            change_sigma = this.sigma_alpha * delta_sigma;

        change_sigma = minimum(change_sigma, this.sigma_max_change * this.sigma);
        change_sigma = maximum(change_sigma, -this.sigma_max_change * this.sigma);
        this.sigma += change_sigma;
      }

      if this.sigma_decay < 1 {
        this.sigma[this.sigma > this.sigma_limit] *= this.sigma_decay;
      }

      if this.learning_rate_decay < 1 && this.learning_rate > this.learning_rate_limit {
        this.learning_rate *= this.learning_rate_decay;
      }
    }

    proc current_param() {
      return this.current_best_mu;
    }

    proc set_mu() {
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
