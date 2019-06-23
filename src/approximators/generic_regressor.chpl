module Gorila {

  use LinearAlgebra;

  class GenericRegressor {
    /*
      This class implements a regressor that approximates a
      generic function. An arbitrary number of inputs and
      outputs is supported.
    */
    proc init(approximator, n_inputs: int, params: []) {
      /*
        Constructor.

        Args:
          approximator (object): the model class to approximate the
            a generic function;
          n_inputs (int): number of inputs of the regressor;
          params (associative array): parameters dictionary to the
            regressor.
      */
      this.n_inputs = n_inputs;
      this.input_preprocessor = params.pop('input_preprocessor', []);
      this.output_preprocessor = params.pop('output_preprocessor', []);
      this.model = approximator(params);
    }

    proc fit(z: [], fit_params: []) {
      /*
        Fit the model.
        Args:
          z (array): list of inputs and targets;
          fit_params (associative array): other parameters used by
            the fit method of the regressor.
      */
      z = preprocess(z);
      this.model.fit(z, fit_params);
    }

    proc predict(x: [], predict_params: []) {
      /*
        Predict.

        Args:
          x (array): list of inputs.
          predict_params (associative array): other parameters used by
          the predict method the regressor.

        Returns:
            The predictions of the model.
      */
      x = preprocess(x);
      return this.model.predict(x, predict_params);
    }

    proc reset() {
      /*
        Reset the model parameters.
      */
      this.model.reset();
    }

    proc weights_size() {
      return this.model.weights_size;
    }

    proc get_weights() {
      return this.model.get_weights();
    }

    proc set_weights(w: []) {
      return this.model.set_weights(w);
    }

    proc diff(x: []) {
      x = preprocess(x);
      return this.model.diff(x);
    }

    proc preprocess(z: []) {
      var x = [z[..this.n_inputs]],
          y = [z[this.n_inputs..]];

      if this.input_preprocessor.length > 0 && this.input_preprocessor[0].type != Array {
        this.input_preprocessor = [this.input_preprocessor];
      }

      for i, ip in enumerate(this.input_preprocessor) {
        for p in ip {
          x[i] = p(x[i]);
        }
      }

      for i in x {
        z = i;
      }

      if y.length > 0 {
        if this.output_preprocessor.length > 0 && this.output_preprocessor[0].type != Array {
          this.output_preprocessor = [this.output_preprocessor];
        }

        for o, op in enumerate(this.output_preprocessor) {
          for p in op {
            y[o] = p(y[o]);
          }
        }
      }
      z += [i for i in y];
      return z;
    }

    proc len() {
      return self.model.length;
    }
  }
}
