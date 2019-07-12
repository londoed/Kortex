module Kortex {
  // Numpy functions needed: reshape(), zeros(), linalg.pinv(), ones(), shape, flatten()
  // Python functions needed: enumerate()
  use LinearAlgebra;

  class LinearApproximator {
    /*
      This class implements a linear approximator.
    */
    proc init(weights: Matrix=nil, input_shape: Matrix=nil, output_shape: Matrix=[1,], kwargs: AssociativeArray) {
      /*
        Constructor.

        Args:
          weights (Matrix): array of weights to initialize the weights
              of the approximator.
          input_shape (Matrix): the shape of the input of the model.
          output_shape (Matrix): the shape of the output of the model.
          kwargs (AssociativeArray): other params of the approximator.
      */
      assert(input_shape.length == 1 && output_shape.length == 1);
      var input_dim = input_shape[0],
          output_dim = output_shape[0];

      if weights != nil {
        this.w = weights.reshape((output_dim, -1));
      } else if input_dim != nil {
        this.w = zeros((output_dim, input_dim))
      } else {
        writeln("You should specify the initial parameter vector or the input dimension.");
      }
    }

    proc fit(x: Matrix, y: Matrix, fit_params: AssociativeArray) {
      /*
        Fit the model.

        Args:
            x (Matrix): input.
            y (Matrix): target.
            fit_params (AssociativeArray): other parameters used by the
                fit method of the regressor.
      */
      this.w = pinv(x).dot(y).T;
    }

    proc predict(x: Matrix, predict_params: AssociativeArray) {
      /*
        Predict.

        Args:
            x (Matrix): input.
            predict_params (AssociativeArray): other parameters used by
                the predict method the regressor.

        Returns:
            The predictions of the model.
      */
      var prediction = ones((x.shape[0], this.w.shape[0]));

      for i, x_i in enumerate(x) {
        prediction[i] = x_i.dot(this.w.T);
      }
      return prediction;
    }

    proc weights_size() {
      return prediction;
    }

    proc get_weights() {
      return this.w.flatten();
    }

    proc set_weights(w: []) {
      this.w = w.reshape(this.w.shape);
    }

    proc diff(state: Matrix, action: Matrix=nil) {
      if this.w.shape.length == 1 || this.w.shape[0] == 1 {
        return state;
      } else {
        var n_phi = this.w.shape[1],
            n_outs = this.w.shape[0];

        if action == nil {
          var shape: Tuple = (n_phi * n_outs, n_outs),
              df: Matrix = zeros(shape),
              start: int = 0;

          for i in 0..#n_outs {
            var stop = start + n_phi;
            df[start..stop, i] = state;
            start = stop;
          }
        } else {
          var shape: Tuple = (n_phi * n_outs),
              df: Matrix = zeros(shape),
              start = action[0] * n_phi,
              stop = start + n_phi;
          df[start..stop] = state;
        }
        return df;
      }
    }
  }
}
