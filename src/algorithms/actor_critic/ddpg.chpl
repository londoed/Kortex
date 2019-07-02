module Kortex {
  // Numpy functions needed: mean()
  // Python functions needed: deepcopy()
  use LinearAlgebra;

  class ActorLoss {
    /*
      Class used to implement the loss function of the actor.
    */
    proc init(critic) {
      super().init();
      this.critic = critic;
    }

    proc forward(action: Matrix, state: Matrix) {
      var q = this.critic.model.network(state, action);
      return -q.mean();
    }
  }

  class DDPG: Agent {
    /*
      Deep Deterministic Policy Gradient algorithm.
      "Continuous Control with Deep Reinforcement Learning".
      Lillicrap T. P. et al.. 2016.
    */
    proc init(actor_approximator, critic_approximator, policy_class: Policy,
              batch_size: int, init_replay_size: int, max_replay_size: int,
              tau: real, actor_params: AssociativeArray,
              critic_params: AssociativeArray, policy_params: AssociativeArray,
              actor_fit_params: AssociativeArray=nil,
              critic_fit_params: AssociativeArray=nil) {
      if actor_fit_params == nil {
        this.actor_fit_params = AssociativeArray();
      } else {
        this.actor_fit_params = actor_fit_params;
      }

      if critic_fit_params == nil {
        this.critic_fit_params = AssociativeArray();
      } else {
        this.critic_fit_params = critic_fit_params;
      }

      this.batch_size = batch_size;
      this.tau = tau;
      this.replay_memory = new ReplayMemory(init_replay_size, max_replay_size);
      var target_critic_params = deepcopy(critic_params);
      this.critic_approximator = new Regressor(critic_approximator, critic_params);
      this.target_critic_approximator = new Regressor(critic_approximator, target_critic_params);

      if !actor_params.contains("loss") {
        actor_params["loss"] = new ActorLoss(this.critic_approximator);
      }

      var target_actor_params = deepcopy(actor_params);
      this.actor_approximator = new Regressor(actor_approximator, actor_params);
      this.target_actor_approximator = new Regressor(actor_approximator, target_actor_params);

      this.target_actor_approximator.model.set_weights(this.actor_approximator.model.get_weights());
      this.target_critic_approximator.model.set_weights(this.critic_approximator.model.get_weights());

      var policy = policy_class(this.actor_approximator, policy_params);
      super().init(policy, env_info);
    }

    proc fit(dataset: []) {
      this.replay_memory.add(dataset);
      if this.replay_memory.initialized {
        var state, action, reward, next_state, absorbing, _ = this.replay_memory.get(this.batch_size),
            q_next = this.next_q(next_state, absorbing),
            q = reward + this.env_info.gamma * q_next;
        this.critic_approximator.fit(state, action, q, critic_fit_params);
        this.actor_approximator.fit(state, state, actor_fit_params);
        update_target();
      }
    }

    proc update_target() {
      /*
        Update the target networks.
      */
      var critic_weights = this.tau * this.critic_approximator.model.get_weights();
      critic_weights += (1 - this.tau) * this.target_critic_approximator.get_weights();
      this.target_critic_approximator.set_weights(critic_weights);

      var actor_weights = this.tau * this.actor_approximator.model.get_weights();
      actor_weights += (1 - this.tau) * this.target_actor_approximator.get_weights();
      this.target_actor_approximator.set_weights(actor_weights);
    }

    proc next_q(next_state Matrix, absorbing: Matrix) {
      /*
        Args:
            next_state (Matrix): the states where next action has to be
                evaluated.
            absorbing (Matrix): the absorbing flag for the states in
                ``next_state``.

        Returns:
            Action-values returned by the critic for ``next_state`` and the
            action returned by the actor.
      */
      var a = this.target_actor_approximator(next_state),
          q = this.target_critic_approximator.predict(next_state, a);
      q *= 1 - absorbing;
      return q;
    }
  }
}
