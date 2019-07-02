module Kortex {
  // Numpy functions needed: exp(), log(), var{} => calc_var,
  use LinearAlgebra, Math;

  class VarianceParameter {
    /*
      Abstract class to implement variance-dependent parameters.
      A target parameter is expected.
    */
    proc init(value: [], exponential: bool=false, min_value: real, tol: real=1.0, size: Tuple=(1,)) {
      this.exponential = exponential;
      this.tol = tol;
      this.weights_var = new Table(size);
      this.x = new Table(size);
      this.x2 = new Table(size);
      this.parameter_value = new Table(size);
      super().init(value, min_value, size);
    }

    proc compute(idx) {
      return this.parameter_value[idx];
    }

    proc update(idx, kwargs) {
      var x = kwargs['target'],
          factor = kwargs.get('factor', 1.0);

      var n = this.n_updates[idx];
      this.n_updates[idx] += 1;

      if n < 2 {
        parameter_value = this.init_value;
      } else {
        var variance = n * (this.x2[idx] - this.x[idx]**2) / (n - 1.0),
            var_estimator = variance * this.weights_var[idx];
        parameter_value = compute_parameter(var_estimator, sigma_process=var, index=idx);
      }

      this.x[idx] += (x - this.x[idx]) / this.n_updates[idx];
      this.x2[idx] += (x**2 - this.x2[idx]) / this.n_updates[idx];
      this.weights_var[idx] = (
        1.0 - factor * parameter_value)**2 * this.weights_var[idx] + (
        factor * parameter_value)**2;
      this.parameter_value[idx] = parameter_value;
    }

    proc compute_parameter(sigma) {
      writeln("VarianceParameter is an abstract class, compute_parameter unavailable");
    }
  }

  class VarianceIncreasingParameter: VarianceParameter {
    proc compute_parameter(sigma) {
      if this.exponential {
        return 1 - exp(sigma * log(0.5) / this.tol);
      } else {
        return sigma / (sigma + this.tol);
      }
    }
  }

  class VarianceDecreasingParameter: VarianceParameter {
    proc compute_parameter(sigma) {
      if this.exponential {
        return exp(sigma * log(0.5) / this.tol);
      } else {
        return 1.0 / (sigma + this.tol);
      }
    }
  }

  class WindowedVarianceParameter: Parameter {
    proc init(value: [], exponential: bool=false, min_value: real=nil, tol: real=1.0, window: int=100, size: Tuple=(1,)) {
      this.exponential = exponential;
      this.tol = tol;
      this.weights_var = new Table(size);
      this.samples = new Table(size + (window,));
      this.indx = new Table(size): int;
      this.window = window;
      this.parameter_value = new Table(size);
      super(WindowedVarianceParameter).init(value, min_value, size);
    }

    proc compute(idx) {
      return this.parameter_value[idx];
    }

    proc update(idx, kwargs) {
      var x = kwargs['target'],
          factor = kwargs.get('factor', 1.0),
          n = this.n_updates[idx];
      this.n_updates[idx] += 1;

      if n < 2 {
        parameter_value = this.init_value;
      } else {
        samples = this.samples[idx];

        if n < this.window {
          samples = samples[.., n: int];
        }
        var variance = calc_var(samples),
            var_estimator = variance * this.weights_var[idx];
        parameter_value = compute_parameter(var_estimator, sigma_process=variance, index=idx);
      }
      indx = Matrix([this.indx[idx]] dtype=int);
      this.samples[idx + (indx,)] = x;
      this.indx[idx] += 1;
      if this.indx[idx] >= this.window {
        this.indx[idx] = 0;
      }

      this.weights_var[idx] = (
        1.0 - factor * parameter_value)**2 * this.weights_var[idx] + (
        factor * parameter_value)**2;
      this.parameter_value[idx] = parameter_value;
    }

    proc compute_parameter(sigma) {
      writeln("WindowedVarianceParameter is an abstract class, compute_parameter unavailable.");
    }
  }

  class WindowedVarianceIncreasingParameter: WindowedVarianceParameter {
    proc compute_parameter(sigma) {
      if this.exponential {
        return 1 - exp(sigma * log(0.5) / this.tol);
      } else {
        return sigma / (sigma + this.tol);
      }
    }
  }
}
