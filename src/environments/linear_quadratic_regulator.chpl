module Kortex {
  // Numpy functions needed: inf, ones(), random.uniform()
  use LinearAlgebra;

  class LQR(Environment) {
    /*
      This class implements a Linear-Quadratic Regulator.
      This task aims to minimize the undesired deviations from nominal values of
      some controller settings in control problems.

      The system equations in this task are found in:
      "Policy gradient approaches for multi-objective sequential decision making".
      Parisi S., Pirotta M., Smacchia N., Bascetta L., Restelli M.. 2014
    */
    proc init(A: Matrix, B: Matrix, Q: Matrix, B: Matrix, random_init: bool=false, gamma: real=0.9, horizon: int=50) {
      this.A = A;
      this.B = B;
      this.Q = Q;
      this.R = R;
      this.random_init = random_init;

      // MDP Properties
      var high_x = LinearAlgebra.inf * ones(A.shape[0]),
          low_x = -high_x,
          high_u = LinearAlgebra.inf * ones(B.shape[0]),
          low_u = -high_u,
          observation_space = new Box(low=low_x, high=high_x),
          action_space = new Box(low=low_u, high=high_u),
          env_info = new ENVInfo(observation_space, action_space, gamma, horizon);
      super().init(env_info);
    }

    proc generate(dimensions: int, eps: real=0.1, idx: int=0, random_init: bool=false, gamma: real=0.9, horizon: int=50) {
      /*
        Factory method that generates an lqr with identity dynamics and
        symmetric reward matrices.

        Args:
            dimensions (int): number of state-action dimensions.
            eps (real): reward matrix weights specifier.
            idx (int): selector for the principal state.
            random_init (bool): start from a random state.
            gamma (real): discount factor.
            horizon (int): horizon of the mdp.
      */
      assert(dimensions >= 1);
      A = eye(dimensions);
      B = eye(dimensions);
      Q = eps * eye(dimensions);
      R = (1 - eps) * eye(dimensions);

      Q[idx, idx] = 1.0 - eps;
      R[idx, idx] = eps;

      return LQR(A, B, Q, R, random_init, gamma, horizon);
    }

    proc reset(state: Matrix=nil) {
      if state == nil {
        if this.random_init {
          this.state = random_uniform(-3, 3, size=this.A.shape[0]);
        } else {
          this.state = 10.0 * ones(this.A.shape[0]);
        }
      } else {
        this.state = state;
      }
      return this.state;
    }

    proc step(action: Matrix) {
      var x = this.state,
          u = action,
          reward = -(x.dot(this.Q).dot(x) + u.dot(this.R).dot(u));
      this.state = this.A.dot(x) + this.B.dot(u);
      
      return this.state, reward, false, new AssociativeArray();
    }
  }
}
