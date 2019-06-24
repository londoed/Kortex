module Kortex {
  // Numpy functions needed: ndim, expand_dims()
  class Regressor {
    /*
      This class implements the function to manage a function approximator. This
      class selects the appropriate kind of regressor to implement according to
      the parameters provided by the user; this makes this class the only one to
      use for each kind of task that has to be performed.

      The inference of the implementation to choose is done checking the provided
      values of parameters ``n_actions``.

      If ``n_actions`` is provided, it means that the user wants to implement an
      approximator of the Q-function: if the value of ``n_actions`` is equal to
      the ``output_shape`` then a ``QRegressor`` is created, else
      (``output_shape`` should be (1,)) an ``ActionRegressor`` is created.
      Else a ``GenericRegressor`` is created.

      An ``Ensemble`` model can be used for all the previous implementations
      listed before simply providing a ``n_models`` parameter greater than 1.
    */

    proc init(approximator, input_shape: Tuple, output+shape: Tuple=(1,), n_actions: int=nil, n_models: int=1, params: []) {
      /*
        Constructor.

        Args:
            approximator (object): the approximator class to use to create
                the model.
            input_shape (tuple): the shape of the input of the model.
            output_shape (tuple): the shape of the output of the model.
            n_actions (int): number of actions considered to create a
                ``QRegressor`` or an ``ActionRegressor``.
            n_models (int): number of models to create.
            **params (associatve array): other parameters to create each model.
      */
      params['output_shape'] = output_shape;
      params['input_shape'] = input_shape;

      this.input_shape = input_shape;
      this.output_shape = output_shape;
      this.n_actions = n_actions;

      assert(n_models >= 1);
      this.n_models = n_models;

      if this.n_models > 1 {
        params['model'] = approximator;
        params['n_models'] = n_models;
        approximator = new Ensemble();
      }

      if n_actions != nil {
        assert(this.output_shape.length == 1 && n_actions >= 2);
        if n_actions == this.output_shape[0] {
          this.impl = new QRegressor(approximator, params);
        } else {
          this.impl = new ActionRegressor(approximator, n_actions, params);
        }
      } else {
        this.impl = new GenericRegressor(approximator, this.input_shape.length, params);
      }
    }

    proc call(z: [], predict_params: []) {
      return predict(z, predict_params);
    }

    proc fit(z: [], fit_params: []) {
      /*
        Fit the model.

        Args:
            z (array): list of input of the model;
            fit_params (associatve array): parameters to use to fit the model.
      */
      if this.input_shape[0].type == Tuple {
        var ndim = this.input_shape[0].length;
      } else {
        var ndim = this.input_shape.length;
      }

      if z[0].ndim = ndim {
        for z_i in z {
          expand_dims(z_i, axis=0);
        }
      }
      this.impl.fit(z, fit_params);
    }

    proc predict(z: [], predict_params: []) {
      /*
        Predict the output of the model given an input.

        Args:
            z (array): list of input of the model
            predict_params(associatve array): parameters to use to predict
            with the model.
        Returns:
            The model prediction.
      */
      if this.input_shape[0].type == Tuple {
        var ndim = this.input_shape[0].length;
      } else {
        var ndim = this.input_shape.length;
      }

      if z[0].ndim = ndim {
        for z_i in z {
          z = expand_dims(z_i, axis=0);
          return this.impl.predict(z, predict_params)[0];
        }
      } else {
        return this.impl.predict(z, predict_params);
      }
    }

    proc model() {
      /*
        Returns:
          The model object.
      */
      return this.impl.model;
    }

    proc reset() {
      /*
        Resets the model parameters.
      */
      this.impl.reset();
    }

    proc input_shape() {
      /*
        Returns:
          The shape of the input of the model.
      */
      return this.input_shape;
    }

    proc output_shape() {
      /*
        Returns the shape of the output of the model.
      */
      return this.output_shape;
    }

    proc weights_size() {
      /*
        Returns:
          The shape of the weights of the model.
      */
      return this.impl.weights_size;
    }

    proc get_weights() {
      /*
        Returns:
          The weights of the model.
      */
      return this.impl.get_weights();
    }

    proc set_weights(w: []) {
      /*
        Args:
          w (array): list of weights to be set in the model.
      */
      this.impl.set_weights(w);
    }

    proc diff(z: []) {
      /*
        Args:
          z (array): the input of the model.

        Returns:
          The derivative of the model.
      */
      return this.impl.diff(z);
    }

    proc len() {
      return this.impl.length;
    }
  }
}
