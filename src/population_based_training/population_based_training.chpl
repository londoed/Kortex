module Kortex {
  // Numpy functions needed: sqrt(), copy()
  // Python functions needed:
  use LinearAlgebra;

  class Worker {
    proc init(idx: int, obj, surrogate_obj, h, theta: real, pop_score: int, pop_params: AssociativeArray, asynchronus: bool=false) {
      this.idx = idx;
      this.asynchronus = asynchronus;
      this.obj = obj;
      this.surrogate_obj = surrogate_obj;
      this.theta = theta;
      this.h = h;
      this.score: int = 0;
      this.loss: int = 0;
      this.pop_score = pop_score;
      this.pop_params = pop_params;
      this.rms: real = 0.0;

      update();
    }

    proc step(vanilla: bool=false, rmsprop: bool=false, adam: bool=false, use_loss: bool=true) {
      var decay_rate: real = 0.0,
          alpha: real = 0.01,
          eps: real = 1e-5,
          d_surrogate_obj: real = -2.0 * this.h * this.theta;

      if use_loss {
        this.loss = (this.obj(this.theta) - this.surrogate_obj(this.theta, this.h))**2;
        var d_loss = 2 * (this,obj(this.theta) - this.surrogate_obj(this.theta, this.h)) * d_surrogate_obj;
      } else {
        var d_loss = -d_surrogate_obj;
      }

      if vanilla {
        this.theta -= d_loss * alpha;
      } else {
        this.rms = decay_rate * this.rms + (1 - decay_rate) * d_loss**2;
        this.theta -= alpha * d_loss / (sqrt(this.rms) + eps);
      }
    }

    proc evaluate() {
      this.score = this.obj(this.theta);
      return this.score;
    }

    proc exploit() {
      if this.asynchronus {
        pop_score, pop_params = proxy_sync(pull=true);
      } else {
        pop_score = this.pop_score;
        pop_params = this.pop_params;
      }

      var best_worker_idx = max(pop_score.items(), key=operator.itemgetter(1))[0];

      if best_worker_idx != this.idx {
        var best_worker_theta, best_worker_h = pop_params[best_worker_idx];
        this.theta = copy(best_worker_theta);
        return true;
      }
      return false;
    }

    proc explore() {
      var eps = random.randn(this.h.shape) * 0.1;
      this.h += eps;
    }

    proc update() {
      if !this.asynchronus {
        this.pop_score[this.idx] = this.score;
        this.pop_params[this.idx] = (copy(this.theta), copy(this.h));
      } else {
        proxy_sync(push=true);
      }
    }

    proc proxy_sync(pull: bool=false, push: bool=false) {
      if pull {
        return this.pop_score[0], this.pop_params[0];
      }

      if push {
        var pop_score_1 = this.pop_score[0],
            pop_params_1 = this.pop_params[0];

        pop_score_1[this.idx] = this.score;
        pop_params_1[this.idx] = (copy(this.theta), copy(this.h));

        this.pop_score[0] = pop_score_1;
        this.pop_params[0] = pop_params_1;
      }
    }
  }
}
