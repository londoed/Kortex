module Gorila {
  class Agent {
    /*
      This class implements the functions to manage the agent (e.g. move the agent
      following its policy).
    */
    proc init(policy: Policy, env_info: ENVInfo, features) {
      /*
        Constructor.

          Args:
              policy (Policy): the policy followed by the agent;
              mdp_info (MDPInfo): information about the MDP;
              features (object, None): features to extract from the state.

      */
      this.policy = policy;
      this.env_info = env_info;
      this.phi = features;
      this.next_action = nil;
    }

    proc fit(dataset) {
      /*
        Fit step.

          Args:
              dataset (list): the dataset.
      */
      writeln("Agent is an abstract class, fit cannot be executed.")
    }

    proc draw_action(state: [] real) {
      /*
        Return the action to execute in the given state. It is the action
        returned by the policy or the action set by the algorithm (e.g. in the
        case of SARSA).

        Args:
          state (np.ndarray): the state where the agent is.
        Returns:
          The action to be executed.
      */

      if this.phi != nil {
        var state = this.phi(state)
      }

      if this.next_action == nil {
        return this.policy.draw_action(state);
      } else {
        var action = this.next_action;
        this.next_action = nil;
        return action;
      }
    }

    proc episode_start() {
      /*
        Called by the agent when a new episode starts.
      */
      this.policy.reset();
    }

    proc stop() {
      /*
        Method used to stop an agent. Useful when dealing with real world
        environments, simulators, or to cleanup environment internals after
        an entity fit/evaluate to enforce consistency.
      */
      continue;
    }
  }
}
