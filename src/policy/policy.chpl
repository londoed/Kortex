module Gorila {
  class Policy {
    /*
      Interface for representing a generic policy.

      A policy is a probability distribution that gives the probability
      of taking an action given a specified state.

      A policy is used by Gorila agents to interact with the environment.

    */
    proc call(state_action: []) {
      /*
        Compute the probability of taking action in a certain state following
        the policy.

        Args:
            state_action (array): array containing a state or a state and an action.
        Returns:
            The probability of all actions following the policy in the given
            state if the list contains only the state, else the probability
            of the given action in the given state following the policy. If
            the action space is continuous, state and action must be provided
      */
      writeln("Call function is not implemented.")
    }

    proc draw_action(state: [] real) {
      /*
        Sample an action in ``state`` using the policy.

        Args:
            state (array): the state where the agent is.
        Returns:
            The action sampled from the policy.
      */
      writeln("Draw action function not implemented.")
    }

    proc reset() {
      /*
        Useful when the policy needs a special initialization
        at the beginning of an episode.
      */
      continue;
    }
  }

  class ParametricPolicy: Policy {
    /*
      Interface for a generic parametric policy.
      A parametric policy is a policy that depends on set of parameters,
      called the policy weights.

      If the policy is differentiable, the derivative of the probability for a
      specified state-action pair can be provided.
    */

    proc diff_log(state: [], action: []) {
      /*
        Compute the gradient of the logarithm of the probability density
        function, in the specified state and action pair.

        Args:
            state (array): the state where the gradient is computed.
            action (array): the action where the gradient is computed.
        Returns:
            The gradient of the logarithm of the pdf w.r.t. the policy weights
      */
      writeln("The policy is not differentiable!");
    }

    proc diff(state: [], action: []) {
      /*
        Compute the derivative of the probability density function, in the
        specified state and action pair. Normally it is computed w.r.t. the
        derivative of the logarithm of the probability density function,
        exploiting the likelihood ratio trick.

        Args:
            state (array): the state where the derivative is computed.
            action (array): the action where the derivative is computed.
        Returns:
            The derivative w.r.t. the  policy weights
      */
      return (this.state, this.action) * ParametricPolicy.diff_log(state, action);
    }

    proc set_weights(weights: []) {
      /*
        Setter.

        Args:
          weights (array): The vector of the new weights to be used by the policy.
      */
      writeln("Set weights function is not implemented.")
    }

    proc get_weights() {
      /*
        Getter.

        Returns:
          The current policy weights.
      */
      writeln("Get weights function is not implemented.")
    }

    proc weights_size() {
      /*
        Property,

        Returns:
          The size of the policy weights.
      */
      writeln("Weights size function is not implemented.")
    }
  }
}
