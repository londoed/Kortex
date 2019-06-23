module Kortex {

  use LinearAlgebra;

  class Ensemble {
    /*
      This class is used to create an ensemble of regressors.
    */
    proc init(model: Approximator, n_models: int, prediction: string, params: [?dom]) {
      /*
        Constructor.

        Args:
            model (object): the model class to approximate the
                Q-function.
            n_models (int): number of regressors in the ensemble;
            prediction (string): the type of prediction to make. It can
                be a 'mean' of the ensembles, or a 'sum'.
            params (associative array): parameters dictionary to create each regressor.
      */
      this._prediction = prediction;
      this._model = [];

      for _ in 0..n_models {
        this._model.append(model(params));
      }
    }

    proc fit(z: [], fit_params: [?dom]) {
      /*
        Fit the index-th model of the ensemble if index is probided,
        a random model otherwise.

        Args:
          z (array): A list containing the inputs to use to predict with
            each regressor of the ensemble.
          fit_params (associatve array): other params.
      */
      var idx = fit_params.pop('idx', nil);
      /* TODO if idx is None:
          self[np.random.choice(len(self))].fit(*z, **fit_params)
      else:
          self[idx].fit(*z, **fit_params) */
      this[idx].fit(z, fit_params);
    }

    proc predict(z: [], fit_params: [?dom]) {
      /*
        Predict.

        Args:
            z (array): a list containing the inputs to use to predict with each
                regressor of the ensemble;
            predict_params (associatve array): other params.
        Returns:
            The predictions of the model.
      */
      var idx = predict_params.pop('idx', nil);
      if idx == nil {
        var predictions = [];
        for i in 0..this.model.length {
          predictions.append(this[i].predict(z, predict_params));
        }

        if predictions.length == 0 {
          writeln("Model not fitted to dataset.")
        }

        if this._prediction == 'mean' {
          results = mean(predictions);
        } else if this._prediction == 'sum' {
          results = sum(predictions);
        }
      }
      return results;
    }

    proc reset() {
      /*
        Reset the model parameters.
      */
      for m in this.model {
        m.reset();
      }
    }

    proc models() {
      /*
        Returns:
          The list of the models in the ensemble.
      */
      return self._model;
    }
  }
}
