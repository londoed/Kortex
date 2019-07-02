module Kortex {
  // Numpy functions needed: argwhere(), ravel(), flatten(), concatenate(), zeros()
  // Python functions needed: enumerate()
  use LinearAlgebra;

  class ActionRegressor {
    /*
      This class is used to approximate the Q-function with a different
      approximator of the provided class for each action. It is often used in MDPs
      with discrete actions and should not be used in MDPs with continuous
      actions.
    */
    proc init(approximator, n_actions: int, params: AssociatveArray) {
      /*
        Constructor.

        Args:
            approximator (object): the model class to approximate the
                Q-function of each action;
            n_actions (int): number of different actions of the problem. It
                determines the number of different regressors in the action
                regressor;
            params (AssociatveArray): parameters dictionary to create each
                regressor.
      */
      this.model = [];
      this.n_actions = n_actions;
      this.input_preprocessor = params.pop('input_preprocessor', []);
      this.output_preprocessor = params.pop('output_preprocessor', []);

      for i in 0..#n_actions {
        this.model.append(approximator(params));
      }
    }

    proc fit(state: Matrix, action: Matrix, q: Matrix, fit_params: AssociatveArray) {
      /*
        Fit the model.

        Args:
            state (Matrix): states.
            action (Matrix): actions.
            q (Matrix): target q-values
            fit_params (AssociatveArray): other parameters used by the fit method
                of each regressor.
      */
      state, q = preprocess(state, q);
      for i in 0..#this.model.length {
        var idxs = argwhere((action == i)[.., 0]).ravel();
        if idxs.size {
          this.model[i].fit(state[idxs, ..], q[idxs], fit_params);
        }
      }
    }

    proc predict(z: [], predict_params: AssociatveArray) {
      /*
        Predict.

        Args:
            z (array): a list containing states or states and actions depending
                on whether the call requires to predict all q-values or only
                one q-value corresponding to the provided action.
            predict_params (AssociatveArray): other parameters used by the
                predict method of each regressor.

        Returns:
            The predictions of the model.
      */
      assert(z.length == 1 || z.length == 2);
      var state = z[0];
      state = preprocess(state);

      if x.length == 2 {
        var action = z[1],
            q = zeros(state.shape[0]);
        for i in 0..#n_actions {
          var idxs = argwhere((action == 1)[.., 0]).ravel();
          if idxs.size {
            q[idxs] = this.model[i].predict(state[idxs, ..], predict_params);
          }
        }
      } else {
        var q = zeros((state.shape[0], this.n_actions));
        for i in 0..#n_actions {
          q[.., i] = this.model[i].predict(state, predict_params).flatten();
        }
      }
      return q;
    }

    proc reset() {
      /*
        Reset the model parameters.
      */
      try {
        for m in this.model {
          m.reset();
        } catch e {
          writeln("Attempt to reset weights of a non-parametric regressor.");
        }
      }
    }

    proc weights_size() {
      return this.model[0].weights_size * this.model.length;
    }

    proc get_weights() {
      var w = [];
      for m in this.model {
        w.append(m.get_weights());
      }
      return concatenate(w, axis=0);
    }

    proc set_weights(w: []) {
      var size = this.model[0].weights_size;
      for i, m in enumerate(this.model) {
        var start = i * size,
            stop = start + size;
        m.set_weights(w[start..stop]);
      }
    }

    proc diff(state: Matrix, action: Matrix) {
      if action == nil {
        var diff = [];
        for m in this.model {
          diff.append(m.diff(state));
        }
        return diff;
      } else {
        var diff = zeros(state.length * this.model.length),
            a = action[0],
            s = state.length;
        diff[s * a..s * (a + 1)] = this.model[a].diff(state);
        return diff;
      }
    }

    proc preprocess(state: Matrix, q: Matrix=nil) {
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
