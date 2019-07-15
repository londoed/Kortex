module Kortex {
  // Numpy functions needed: linalg.norm(), zeros(), sqrt()
  use LinearAlgebra;

  class Optimizer {
    proc init(pi: Matrix, epsilon: real=1e-08) {
      this.pi = pi;
      this.dimension = pi.num_params;
      this.epsilon = epsilon;
      this.t: int = 0;
    }

    proc update(globalg: Matrix) {
      this.t += 1;
      var step = compute_step(globalg),
          theta = this.pi.mu,
          ratio = norm(step) / (norm(theta) + this.epsilon);
      this.pi.mu = theta + step;

      return ratio;
    }

    proc compute_step(globalg: Matrix) {
      writeln("Optimizer is an abstract class, compute_step unavailable");
    }
  }

  class BasicSGD: Optimizer {
    proc init(pi: Matrix, stepsize: int) {
      Optimizer.init(pi);
      this.stepsize = stepsize;
    }

    proc compute_step(globalg: Matrix) {
      var step = -this.stepsize * globalg;
      return step;
    }
  }

  class SGD: Optimizer {
    proc init(pi: Matrix, stepsize: int, momentum: real=0.9) {
      Optimizer.init(pi);
      this.v = zeros(this.dimension): real;
      this.stepsize = stepsize;
      this.momentum = momentum;
    }

    proc compute_step(globalg: Matrix) {
      this.v = this.momentum * this.v + (1.0 - this.momentum) * globalg;
      var step = -this.stepsize * this.v;
      return step;
    }
  }

  class Adam: Optimizer {
    proc init(pi: Matrix, stepsize: int, beta1: real=0.99, beta2: real=0.999) {
      this.stepsize = stepsize;
      this.beta1 = beta1;
      this.beta2 = beta2;
      this.m = zeros(this.dimension): real;
      this.v = zeros(this.dimension): real;
    }

    proc compute_step(globalg: Matrix) {
      var a = this.stepsize * sqrt(1 - this.beta2**this.t) / (1 - this.beta1**this.t);
      this.m = this.beta1 * this.m + (1 - this.beta1) * globalg;
      this.v = this.beta2 * this.v + (1 - this.beta2) * (globalg**2);
      var step = -a * this.m / (sqrt(this.v) + this.epsilon);

      return step;
    }
  }
}
