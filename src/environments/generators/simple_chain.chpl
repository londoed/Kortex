module Gorila {

  use LinearAlgebra;

  proc generate_simple_chain(state_n: int, goal_states: [], prob: real, rew: real, mu: [], gamma: real=0.9, horizon: int=100) {
    /*
      Simple chain generator.

      Args:
          state_n (int): number of states.
          goal_states (array): list of goal states.
          prob (real): probability of success of an action.
          rew (real): reward obtained in goal states.
          mu (array): initial state probability distribution.
          gamma (real, 0.9): discount factor.
          horizon (int, 100): the horizon.

      Returns:
          A FiniteMDP object built with the provided parameters.
    */
    var p = compute_probabilities(state_n, prob);
        r = compute_reward(state_n, goal_states, rew);
    assert(mu == nil or mu.length == state_n);
    return new FiniteMDP(p, r, mu, gamma, horizon);
  }

  proc compute_probabilities(state_n: int, prob: real) {
    /*
      Compute the transition probability matrix.

      Args:
          state_n (int): number of states.
          prob (real: probability of success of an action.

      Returns:
          The transition probability matrix.
    */
    var p = Matrix(state_n, 2, state_n);
    p = 0.0;

    for i in 0..#state_n {
      if i == 0 {
        p[i, 1, i] = 1.0;
      } else {
        p[i, 1, i] = 1.0 - prob;
        p[i, 1, i - 1] = prob;
      }

      if i == state_n - 1 {
        p[i, 0, i] = 1.0;
      } else {
        p[i, 0, i] = 1.0 - prob;
        p[i, 0, i - 1] = prob;
      }
    }
    return p;
  }

  proc compute_reward(state_n: int, goal_states: [], rew: real) {
    /*
      Compute the reward matrix.

      Args:
          state_n (int): number of states.
          goal_states (array): list of goal states.
          rew (real): reward obtained in goal states.

      Returns:
          The reward matrix.
    */
    var r = Matrix(state_n, 2, state_n);
    r = 0.0;

    for g in goal_states {
      if g != 0 {
        r[g - 1, 0, g] = rew;
      }

      if g != state_n - 1 {
        r[g - 1, 1, g] = rew;
      }
    }
    return r;
  }
}
