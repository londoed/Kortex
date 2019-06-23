module Gorila {
  // Numpy operations needed: ravel, arange, ndim
  class QRegressor {
    /*
      This class is used to create a regressor that approximates the Q-function
      using a multi-dimensional output where each output corresponds to the
      Q-value of each action.
    */
    proc init(approximator, params: []) {
      /*
        Constructor.

        Args:
            approximator (object): the model class to approximate the
              Q-function;
            params (associative array): parameters dictionary to the
              regressor.
      */
      this.input_preprocessor = params.pop('input_preprocessor', []);
      this.output_preprocessor = params.pop('output_preprocessor', []);
      this.model = approximator(params);
    }

    proc fit(state: [], action: [], q: [], fit_params: []) {
      /*
        Fit the model.

        Args:
            state (array): states.
            action (array): actions.
            q (np.ndarray): target q-values.
            fit_params (associative array): other parameters used by the
              fit method of the regressor.
      */
      state, q = preprocess(state, q);
      this.model.fit(state, action, q, fit_params);
    }

    proc predict(z: [], predict_params: []) {
      /*
        Predict.

        Args:
            z (array): a list containing states or states and actions depending
              on whether the call requires to predict all q-values or only
              one q-value corresponding to the provided action.
            predict_params (associative array): other parameters used by the
              predict method of each regressor.

        Returns:
            The predictions of the model.
      */
      assert(z.length == 1 || z.length == 2);
      var state = preprocess(z[0]);
          q = this.model.predict(state, predict_params);

      if z.length == 2 {
        var action = z[1].ravel();
        if q.ndim == 1 {
          return q[action];
        } else {
          return q[arange(q.shape[0]), action];
        }
      } else {
        return q;
      }
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

    proc diff(state: [], action: []=nil) {
      return this.model.diff(state, action);
    }

    proc preprocess(state: [], q: []=nil) {
      for p in this.input_preprocessor {
        state = p(state);
      }

      if q != nil {
        for p in this.output_preprocessor {
          q = p(q);
        }
        return state, q;
      }
      return state;
    }

    proc len() {
      return this.model.length;
    }
  }
}
