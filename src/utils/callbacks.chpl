module Kortex {
  // Numpy functions needed: mean(), max()
  // Python functions needed: deepcopy()
  use LinearAlgebra, Math;

  class CollectDataset {
    /*
      This callback can be used to collect samples during the learning
      of the agent.
    */
    proc init() {
      this.dataset = [];
    }

    proc add_samples(dataset: []) {
      /*
        Add samples to the samples list.

        Args:
          dataset (array): the samples to collect.
      */
      this.dataset += dataset;
    }

    proc get() {
      /*
        Returns:
          The current samples list.
      */
      return this.dataset;
    }

    proc clean() {
      /*
        Deletes the current dataset.
      */
      this.dataset = [];
    }
  }

  class CollectQ {
    /*
      This callback can be used to collect the action values in all
      states at the current time step.
    */
    proc init(approximator: [Table, EnsembleTable]) {
      /*
        Constructor.

        Args:
          approximator ([Table, EnsembleTable]): the approximator to use to
              predict the action values.
      */
      this.approximator = approximator;
      this.qs = [];
    }

    proc add_values(kwargs: AssociativeArray) {
      /*
        Add action values to the action-value list.

        Args:
          kwargs (associative array): empty associative array.
      */
      if this.approximator.type == EnsembleTable {
        var qs = [];

        for m in this.approximator.model {
          qs.append(m.table);
        }

        this.qs.append(deepcopy(mean(qs, 0)));
      } else {
        this.qs.append(deepcopy(this.approximator.table));
      }
    }

    proc get_values() {
      /*
        Returns:
          The current action-values list.
      */
      return this.qs;
    }
  }

  class CollectMaxQ {
    /*
      This callback can be used to collect the maximum action value in
      a given state at each call.
    */
    proc init(approximator: [Table, EnsembleTable], state: Matrix) {
      /*
        Constructor.

        Args:
          approximator ([Table, EnsembleTable]): the approximator to use.
          state (Matrix): the state to consider.
      */
      this.approximator = approximator;
      this.state = state;
      this.max_qs = [] real;
    }

    proc add_values(kwargs: AssociativeArray) {
      /*
        Add maximum action values to the maximum action-values list.

        Args:
            kwargs (associative array): empty dictionary.
      */
      var q = this.approximator.predict(this.state),
          max_q = max(q);
      this.max_q.append(max_q);
    }

    proc get_values() {
      /*
        Returns:
          The current maximum action-values list.
      */
      return this.max_qs;
    }
  }

  class CollectParameters {
    /*
      This callback can be used to collect the values of a parameter
      during a run of the agent.
    */
    proc init(parameter: Parameter, idx: [] int) {
      /*
        Constructor.

        Args:
            parameter (Parameter): the parameter whose values have to be
                collected.
            idx (array): index of the parameter when the ``parameter`` is
                tabular.
      */
      this.parameter = parameter;
      this.idx = idx;
      this.p = [];
    }

    proc add_values(kwargs: AssociativeArray) {
      /*
        Add the parameter value to the parameter values list.

        Args:
          kwargs (AssociativeArray): an empty associatve array.
      */
      var value = this.parameter.get_value(this.idx);

      if value.type == Matrix {
        value = Matrix(value);
      }
      
      this.p.append(value);
    }

    proc get_values() {
      /*
        Returns:
          The current parameter values list.
      */
      return this.p;
    }
  }
}
