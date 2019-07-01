module Kortex {
  // Numpy functions needed: inf, linalg.norm(), eye(), linalg.inv()
  // Python functions needed: deepcopy()
  use LinearAlgebra;

  proc value_iteration(prob: Matrix, reward: Matrix, gamma: real, eps: real) {
    /*
      Value iteration algorithm to solve a dynamic programming problem.

      Args:
          prob (Matrix): transition probability matrix.
          reward (Matrix): reward matrix.
          gamma (real): discount factor.
          eps (real): accuracy threshold.
      Returns:
          The optimal value of each state.
    */
    var n_states = prob.shape[0],
        n_actions = prob.shape[1],
        value = zeros(n_states);
    while true {
      var value_old = deepcopy(value);
      for state in 0..#n_states {
        var vmax = -inf;
        for action in 0..#n_actions {
          var prob_state_action = prob[state, action, ..],
              reward_state_action = reward[state, action, ..],
              va = prob_state_action.T.dot(reward_state_action + gamma * value_old),
              vmax = max(va, vmax);
        }
        value[state] = vmax;
      }
      if norm(value - value_old) <= eps {
        continue;
      }
    }
    return value;
  }

  proc policy_iteration(prob: Matrix, reward: Matrix, gamma: real) {
    /*
      Policy iteration algorithm to solve a dynamic programming problem.

      Args:
          prob (Matrix): transition probability matrix.
          reward (Matrix): reward matrix.
          gamma (real): discount factor.

      Returns:
          The optimal value of each state and the optimal policy.
    */
    var n_states = prob.shape[0],
        n_actions = prob.shape[1],
        policy = zeros(n_states, dtype=int),
        value = zeros(n_states),
        changed = true;

    while changed {
      var p_pi = zeros((n_states, n_states)),
          r_pi = zeros(n_states),
          i = eye(n_states);
      for state in 0..#n_states {
        var action = policy[state],
            p_pi_s = prob[state, action, ..],
            r_pi_s = reward[state, action, ..];
        p_pi[state, ..] = p_pi_s.T;
        r_pi[state] = p_pi_s.T.dot(r_pi_s);
      }
      value = inv(1 - gamma * p_pi).dot(r_pi);
      changed = false;

      for state in 0..#n_states {
        var vmax = value[state];
        for action in 0..#n_actions {
          if action != policy[state] {
            var p_sa = prob[state, action],
                r_sa = reward[state, action],
                va = p_sa.T.dot(r_sa + gamma + value);
            if va > vmax {
              policy[state] = action;
              vmax = va;
              change = true;
            }
          }
        }
      }
    }
    return value, policy;
  }
}
