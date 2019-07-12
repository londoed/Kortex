module Kortex {
  // Numpy functions needed: expand_dims(), ravel(), argwhere(), max(), random.uniform(), random.choice(), exp(), sum(), isnan(), isinf()
  // Scipy functions needed: brentq(), logsumexp()
  use LinearAlgebra;

  class TDPolicy: Policy {
    proc init() {
      /*
        Constructor.
      */
      this.approximator = nil;
    }

    proc set_q(approximator) {
      /*
        Args:
          approximator (object): the approximator to use.
      */
      this.approximator = approximator;
    }

    proc get_q() {
      /*
        Returns:
          The approximator used by the policy.
      */
      return this.approximator;
    }
  }

  class EpsilonGreedy: TDPolicy {
    /*
      Epsilon greedy policy.
    */
    proc init(epsilon: Parameter) {
      /*
        Constructor.

        Args:
            epsilon (Parameter): the exploration coefficient. It indicates
                the probability of performing a random actions in the current
                step.
      */
      super().init();
      assert(epsilon.type == Parameter);
      this.epsilon = epsilon;
    }

    proc call(args: Vector) {
      var state = args[0],
          q = this.approximator.predict(expand_dims(state, axis=0)).ravel(),
          max_a = argwhere(q == max(q)).ravel(),
          p = this.epsilon.get_value(state) / this.approximator.n_actions;

      if args.length == 2 {
        var action = args[1];

        if action in max_a {
          return p + (1.0 - this.epsilon.get_value(state)) / max_a.length;
        } else {
          return p;
        }
      } else {
        var probs = ones(this.approximator.n_actions) * p;
        probs[max_a] += (1.0 - this.epsilon.get_value(state)) / max_a.length;

        return probs;
      }
    }

    proc draw_action(state: Matrix) {
      if !random_uniform() < this.epsilon(state) {
        var q = this.approximator.predict(state),
            max_a = argwhere(q == max(q)).ravel();
            
        if max_a.length > 1 {
          max_a = Matrix(random_choice(max_a));
        }
        return max_a;
      }
      return Matrix([random_choice(this.approximator.n_actions)]);
    }

    proc set_epsilon(epsilon: Parameter) {
      /*
      Setter.

      Args:
          epsilon (Parameter): the exploration coefficient. It indicates the
           robability of performing a random actions in the current step.
      */
      assert(epsilon.type == Parameter);
      this.epsilon = epsilon;
    }

    proc update(idx: Vector) {
      /*
        Update the value of the epsilon parameter at the provided index (e.g. in
        case of different values of epsilon for each visited state according to
        the number of visits).

        Args:
            idx (Vector): index of the parameter to be updated.
      */

      this.epsilon.update(idx);
    }
  }

  class Boltzmann: TDPolicy {
    /*
      Boltzmann softmax policy.
    */
    proc init(beta: Parameter) {
      /*
        Constructor.

        Args:
            beta (Parameter): the inverse of the temperature distribution. As
            the temperature approaches infinity, the policy becomes more and
            more random. As the temperature approaches 0.0, the policy becomes
            more and more greedy.
      */
      super().init();
      this.beta = beta;
    }

    proc call(args: Vector) {
      var state = args[0],
          q_beta = this.approximator.predict(state) * this.beta(state);
      q_beta -= q_beta.max();
      var qs = exp(q_beta);

      if args.length == 2 {
        var action = args[1];
        return qs[action] / sum(qs);
      } else {
        return qs / sum(qs);
      }
    }

    proc draw_action(state: Matrix) {
      return Matrix(random_choice(this.approximator.n_actions, p=this(state)));
    }
  }

  class Mellowmax: Boltzmann {
    /*
      Mellowmax policy.
      "An Alternative Softmax Operator for Reinforcement Learning". Asadi K. and
      Littman M.L.. 2017.
    */

    class MellowmaxParameter {
      proc init(outer, omega: Parameter, beta_min: real, beta_max: real) {
        this.omega = omgea;
        this.outer = outer;
        this.beta_min = beta_min;
        this.beta_max = beta_max;
      }

      proc call(state: Matrix) {
        var q = this.outer.approximator.predict(state),
            mm = (logsumexp(q * this.omega(state)) - log(q.size)) / this.omega(state);

        proc f(beta) {
          var v = q - mm,
              beta_v = beta * v;
          beta_v -= beta_v.max();
          return sum(exp(beta_v) * v);

          try {
            beta = brentq(f, a=this.beta_min, b=this.beta_max);
            assert !(isnil(beta) || isinf(beta));
            return beta;
          } catch Error.message("ValueError") {
            return 0;
          }
        }
      }
    }

    proc init(omega: Parameter, beta_min: real=-10.0, beta_max: real=10.0) {
      /*
        Constructor.

        Args:
            omega (Parameter): the omega parameter of the policy from which beta
                of the Boltzmann policy is computed.
            beta_min (real): one end of the bracketing interval for
                minimization with Brent's method.
            beta_max (real): the other end of the bracketing interval for
                minimization with Brent's method.
      */
      var beta_mellow = new MellowmaxParameter(omega, beta_min, beta_max);
      super().init(beta_mellow);
    }
  }
}
