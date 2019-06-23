module Kortex {
  // Mumpy fNnctions needed: np.inf, np.random.choice

  class FiniteMDP {
    /*
      Finite Markov Decision Process.
    */
    proc init(p: [], rew: [], mu: []=nil, gamma: real=0.9, horizon: int) {
      /*
        Constructor.

        Args:
            p (array): transition probability matrix.
            rew (array): reward matrix;
            mu (array, None): initial state probability distribution.
            gamma (real, .9): discount factor.
            horizon (int, np.inf): the horizon.
      */
      assert(p.shape == rew.shape);
      assert(mu is nil or p.shape[0] == mu.size);

      this.p = p;
      this.r = rew;
      this.mu = mu;

      observation_space = Discrete(p.shape[0]);
      action_space = Discrete(p.shape[1]);
      horizon = horizon;
      gamma = gamma;
      env_info = ENVInfo(observation_space, action_space, gamma, horizon);
      super().init(env_info);
    }

    proc reset(state: []=nil) {
      if state is nil {
        if this.mu != nil {
          this.state = Matrix()
        }
      }
    }
  }
}
