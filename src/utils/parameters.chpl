module Gorila {
  use LinearAlgebra;
  use Gorila.utils.table only Table;

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
}
