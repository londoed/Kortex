module Kortex {
  // Numpy functions needed: ones()
  use LinearAlgebra;

  proc Features(basis_list: []=nil, tilings) {
    /*
      Factory method to build the requested type of features. The types are
      mutually exclusive.

      Args:
          basis_list (Array): list of basis functions.
          tilings (object): single object or list of tilings.

      Returns:
          The class implementing the requested type of features.
    */
    if basis_list != nil && tilings == nil {
      return new BasisFeatures(basis_list);
    } else if basis_list == nil && tilings != nil {
      return new TilesFeatures(tilings);
    } else {
      writeln("You must specify a list of basis or a list of tilings.")
    }
  }

  proc get_action_features(phi_state: Matrix, action: Matrix, n_actions: int) {
    /*
      Compute an array of size ``phi_state.length`` * ``n_actions`` filled with
      zeros, except for elements from ``phi_state.length`` * ``action`` to
      ``phi_state.length`` * (``action`` + 1) that are filled with `phi_state`. This
      is used to compute state-action features.

      Args:
          phi_state (Matrix): the feature of the state.
          action (Matrix): the action whose features have to be computed.
          n_actions (int): the number of actions.

      Returns:
          The state-action features.
    */
    if phi_state.shape.length > 1 {
      assert(phi_state.shape[0] == action.shape[0]);
      var phi = ones((phi_state.shape[0], n_actions * phi_state[0].size)),
          i: int = 0;

      for s, a in zip(phi_state, action) {
        var start = s.size * a[0]: int,
            stop = start + s.size,
            phi_sa = zeros(n_actions * s.size);
        phi_sa[start..stop] = s;
        phi[i] = phi_sa;
        i += 1;
      } else {
        var start = phi_state.size * action[0],
            stop = start + phi_state.size,
            phi = zeros(n_actions * phi_state.size);
        phi[start..stop] = phi_state;
      }
      return phi;
    }
  }
}
