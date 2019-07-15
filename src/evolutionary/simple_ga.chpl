module Kortex {
  // Numpy functions needed: zeros(), random.randn(), where(), size, random.choice(), concatenate(), argsort(),
  //   copy(), std()
  use LinearAlgebra;

  class SimpleGeneticAlgorithm {
    proc init(num_params: int, sigma_init: real=0.1, sigma_decay: real=0.999, sigma_limit: real=0.01,
              pop_size: int=256, elite_ratio: real=0.1, forget_best: bool=false, weight_decay: real=0.01) {
      this.num_params = num_params;
      this.sigma_init = sigma_init;
      this.sigma_decay = sigma_decay;
      this.sigma_limit = sigma_limit;
      this.pop_size = pop_size;

      this.elite_ratio = elite_ratio;
      this.elite_pop_size = (this.pop_size * this.elite_ratio): int;

      this.sigma = this.sigma_init;
      this.elite_params = zeros(this.elite_pop_size, this.num_params);
      this.elite_rewards = zeros(this.elite_pop_size);
      this.best_param = zeros(this.num_params);
      this.best_reward = 0;
      this.first_iter = true;
      this.forget_best = forget_best;
      this.weight_decay = weight_decay;
    }

    proc rms_stdev() {
      return this.sigma;
    }

    proc ask() {
      this.epsilon = randn(this.pop_size, this.num_params) * this.sigma;
      var solutions = [];

      proc mate(a: Matrix, b: Matrix) {
        var c = copy(a),
            idx = where(randn((c.size)) > 0.5);
        c[idx] = b[idx];

        return c;
      }

      var elite_range = range(this.elite_pop_size);

      for i in 0..#this.pop_size {
        var idx_a = choice(elite_range),
            idx_b = choice(elite_range),
            child_params = mate(this.elite_params[idx_a], this.elite_params[idx_b]);
        solutions.append(child_params + this.epsilon[i]);
      }

      var solutions = Matrix(solutions);
      this.solutions = solutions;

      return solutions;
    }

    proc tell(reward_table_result: []) {
      assert(reward_table_result.length == this.pop_size);

      var reward_table = Matrix(reward_table_result);

      if this.weight_decay > 0 {
        var l2_decay = compute_weight_decay(this.weight_decay, this.solutions);
        reward_table += l2_decay;
      }

      if this.forget_best || this.first_iter {
        var reward = reward_table,
            solution = this.solutions;
      } else {
        var reward = concatenate([reward_table, this.elite_rewards]),
            solution = concatenate([this.solutions, this.elite_params]);
      }

      var idx = argsort(reward)[.., .., -1][0..this.elite_pop_size];
      this.elite_rewards = reward[idx];
      this.elite_params = solution[idx];
      this.current_best_reward = this.elite_rewards[0];

      if this.first_iter || (this.current_best_reward > this.best_reward) {
        this.first_iter = false;
        this.best_reward = this.elite_rewards[0];
        this.best_param = copy(this.elite_params[0]);
      }

      if (this.sigma > this.sigma_limit) {
        this.sigma *= this.sigma_decay;
      }
    }

    proc current_param() {
      return this.elite_params[0];
    }

    proc set_mu(mu: Matrix) {
      continue;
    }

    proc best_param() {
      return this.best_param;
    }

    proc result() {
      return new Tuple(this.best_param, this.best_reward, this.current_best_reward, this.sigma);
    }
  }
}
