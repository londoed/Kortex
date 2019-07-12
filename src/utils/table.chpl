module Kortex {

  use LinearAlgebra;
  use Gorila.apprximators only Ensemble;

  class Table {
    /*
      Table regressor. Used for discrete state and action spaces.
    */
    proc init(shape: Tuple, init_value: int, dtype: []=nil) {
      /*
        Constructor.
        Args:
            shape (tuple): the shape of the tabular regressor.
            initial_value (float, 0.): the initial value for each entry of the
                tabular regressor.
            dtype ([int, float], None): the dtype of the table array.
      */
      this.table = ones(shape, dtype=dtype) * init_value;
    }

    proc get_item(args) {
      if this.table.size == 1 {
        return this.table[0];
      } else {
        for arg in args {
          var idx = new Tuple(arg[0]);
        }
      }
      return this.table[idx];
    }

    proc set_item(args, value) {
      if this.table.size == 1 {
        self.table[0] = value;
      } else {
        for arg in args {
          var idx = new Tuple(arg[0]);
        }
      }
      this.table[idx] = value;
    }

    proc fit(x: int, y: real) {
      /*
        Args:
          x (int): Index of the table to be filled.
          y (real): Value to fill in the table.
      */
      this[x] = y;
    }

    proc predict(z: []) {
      /*
        Predict the output of the table given an input.

        Args:
            *z (list): list of input of the model. If the table is a Q-table,
            this list may contain states or states and actions depending
                on whether the call requires to predict all q-values or only
                one q-value corresponding to the provided action;
        Returns:
            The table prediction.
      */
      if z[0].ndim == 1 {
        for i in z {
          expand_dims(i);
        }
      }

      var state = z[0],
          values = [];

      if z.length == 2 {
        var action = z[1];
        
        for i in 0..state.length {
          var val = self[state[i], action[i]];
          values.append(val);
        }
      } else {
        for i in 0..state.length {
          var val = self[state[i], ..];
          values.append(val);
        }
      }

      if values.length == 1 {
        return values[0];
      } else {
        return [values];
      }
    }

    proc n_actions() {
      /*
        Returns:
          The number of actions considered by the table.
      */
      return this.table.shape[-1];
    }

    proc shape() {
      /*
        Returns:
          The shape of the table.
      */
      return this.table.shape;
    }
  }

  class EnsembleTable: Ensemble {
    /*
      Implements functions to manage table ensembles.
    */
    proc init(n_models: int, shape: [], prediction: string="mean") {
      /*
        Constructor.

        Args:
          n_models (int): number of models in the ensemble.
          shape (array): shape of each table in the ensemble.
          prediction (string): type of prediction to return.
      */
      var approximator_params = ['shape' => shape];
    }

    proc n_actions() {
      return this._model[0].shape[-1];
    }
  }
}
