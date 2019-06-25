module Kortex {
  // Numpy functions needed: maximum(). asscalar(), dot(), sqrt()
  use LinearAlgebra;

  class Parameter {
    /*
      This class implements functionality to manage parameters, such
      as learning rate. It also allows to have a single parameter for
      each state of state-action tuple.
    */
    proc init(value: real, min_value: real=nil, size: Tuple=(1,)) {
      /*
        Constructor.

        Args:
          value (real): Initial value of the parameter.
          min_value (real): The minimum value that it can reach with decreasing.
          size (tuple): Shape of the matrix of parameters. This shape can be
          used to have a single parameter for each state or state-action tuple.
      */
      this._init_value = value;
      this._min_value = min_value;
      this._n_updates = Table(size);
    }

    proc call(idx: []) {
      /*
        Update and return the parameter in the provided index.

        Args:
             idx (array): index of the parameter to return.
        Returns:
            The updated parameter in the provided index.
      */
      if this._n_updates.table.size == 1 {
        idx = [];
      }
      update(idx);
      return get_value(idx);
    }

    proc get_value(idx: []) {
      /*
        Return the current value of the parameter at the provided index.

        Args:
          idx (array): Index of the parameter to return.

        Returns:
          The current value of the parameter at the specified index.
      */
      if this._min_value is nil || new_value >= self._min_value {
        return new_value;
      } else {
        return self._min_value;
      }
    }

    proc compute(idx: []) {
      /*
        Returns:
          The value of the parameter at the specified index.
      */
      return this._init_value;
    }

    proc update(idx: []) {
      /*
        Updates the number of visit of the parameter in the provided index.

        Args:
          idx (array): Index of the parameter whose number of visits has to be
            updated.
      */
      self._n_updates[idx] += 1;
    }

    proc shape() {
      /*
        Returns:
          The shape of the table of parameters.
      */
      return this._n_updates.table.shape;
    }
  }

  class LinearDecayParameter: Parameter {
    /*
      This class implements a linearly decaying parameter according to the
      number of times it has been used.
    */
    proc init(value, min_value, n, size: Tuple={1,}) {
      this.coeff = (min_value - value) / n;
      super(LinearDecayParameter).init(value, min_value, size);
    }

    proc compute(idx) {
      return this.coeff * this.n_updates[idx] + this.init_value;
    }
  }

  class ExponentialDecayParameter: Parameter {
    /*
      This class implements an exponentially decaying parameter according
      to the number of times it has been used.
    */
    proc init(value, decay_exp: real=1.0, min_value: real=nil, size: Tuple=(1,)) {
      this.decay_exp = decay_exp;
      super(ExponentialDecayParameter).init(value, min_value, size);
    }

    proc compute(idx) {
      var n = maximum(this.n_updates[idx], 1);
      return this.init_value / n**this.decay_exp;
    }
  }

  class AdaptiveParameter {
    /*
      This class implements a basic adaptive gradient step. Instead of moving of
      a step proportional to the gradient, takes a step limited by a given metric.
      To specify the metric, the natural gradient has to be provided. If natural
      gradient is not provided, the identity matrix is used.
    */
    proc init(value) {
      this.eps = value;
    }

    proc call(args) {
      return this.get_value(args);
    }

    proc get_value(args) {
      if args.length == 2 {
        var gradient = args[0],
            nat_gradient = args[1],
            tmp = asscalar(gradient.dot(nat_gradient)),
            lambda_v = sqrt(tmp / (4.0 * this.epsilon));
        lambda_v = max(lambda_v, 1e-8);
        var step_length = 1.0 / (2.0 * lambda_v);
        return step_length;
      } else if args.length == 1 {
        return get_value(args[0], args[0]);
      } else {
        return new Error.message("Adaptive parameters need gradient");
      }
    }

    proc shape() {
      return nil;
    }
  }
}
