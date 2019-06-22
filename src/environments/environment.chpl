module Gorila {
  class ENVInfo {
    /*
      This class is used to store the information of the environment.
    */
    proc init(observation_space: [Box, Discrete], action_space: [Box, Discrete], gamma: real, horizon: int) {
      /*
        Constructor.

        Args:
           observation_space ([Box, Discrete]): the state space;
           action_space ([Box, Discrete]): the action space;
           gamma (float): the discount factor;
           horizon (int): the horizon.
      */
      this.observation_space = observation_space;
      this.action_space = action_space;
      this.gamma = gamma;
      this.horizon = horizon;
    }

    proc size() {
      /*
        Returns:
          The sum of the number of discrete states and discrete actions. Only
          works for discrete spaces.
      */
      return this.observation_space.size + this.action_space.size;
    }

    proc shape() {
      /*
        Returns:
          The concatenation of the shape tuple of the state and action spaces.
      */
      return this.observation_space.shape + this.action_space.shape;
    }
  }

  class Environment {
    proc init(env_info: ENVInfo) {
      /*
        Constructor.

        Args:
          env_info (ENVInfo): An object containing the info of the environment.
      */
      this._env_info = env_info;
    }

    proc seed(seed: real) {
      /*
        Set the seed of the environment.

        Args:
          seed (real): The value of the seed.
      */
      this.env.seed(seed);
    }

    proc reset(state: ([], nil)) {
      /*
        Reset the current state.

        Args:
            state (array): the state to set to the current state.
        Returns:
            The current state.
      */
      writeln("Reset function not implemented.")
    }

    proc step(action: []) {
      /*
        Move the agent from its current state according to the action.

        Args:
            action (array): the action to execute.
        Returns:
            The state reached by the agent executing ``action`` in its current
            state.
      */
      writeln("Step function not implemented.")
    }

    proc render() {
      writeln("Render function not implemented.")
    }

    proc info() {
      /*
        Returns:
          An object containing the info of the environment.
      */
      return this._env_info;
    }

    proc bound(x, min_value: real, max_value: real) {
      /*
        Method used to bound state and action variables.

        Args:
            x: the variable to bound;
            min_value: the minimum value;
            max_value: the maximum value;
        Returns:
            The bounded variable.
      */
      return (min_value, (x, max_value).max).max;
    }
  }
}
