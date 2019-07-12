module Kortex {
  class Entity {
    /*
      The central class to implement Deep RL algorithms.
    */
    proc init(agent: Agent, env: Environment, callbacks: []=nil) {
      this.agent = agent;
      this.env = env;
      this.state = nil;
      this.total_episodes = 0;
      this.total_steps = 0;
      this.current_episodes = 0;
      this.episode_steps = 0;
      this.n_episodes = nil;
      this.n_steps_per_policy = nil;
      this.n_episodes_per_policy = nil;

      if callbacks != nil {
        this.callbacks = callbacks;
      } else {
        this.callbacks = [];
      }
    }

    proc fit(n_steps: int, n_episoes: int, n_steps_per_policy: int, n_episodes_per_policy : int,
             render: bool=false) {
      /*
        Args:
          n_steps: Number of steps to move the agent.
          n_episodes: Number of episodes to move the agent.
          n_steps_per_policy: Number of steps between each fit of the
          policy.
          n_episodes_per_policy: Number of episodes between each fit of
          the policy.
          render: Whether to render the environment or not.
      */
      assert((n_episodes_per_policy is not None and n_steps_per_policy is None) || (n_episodes_per_policy is None and n_steps_per_policy is not None))

      this.n_steps_per_policy = n_steps_per_policy;
      this.n_episodes_per_policy = n_episodes_per_policy;

      if n_steps_per_policy != nil {
        var policy_cond = this.current_steps_counter >= this.n_steps_per_policy;
      } else {
        var policy_cond = this.current_steps_counter >= this.n_steps_per_policy;
      }
      run(n_steps, n_episodes, policy_cond, render);
    }

    proc evaluate(init_states: [] real, n_steps: int, n_episodes: int, render: bool=false) {
      /*
        This function moves the agent in the environment using its policy.
        The agent is moved for a provided number of steps, episodes, or from
        a set of initial states for the whole episode. By default, the
        environment is reset.

        Args:
            init_states: The starting states of each episode.
            n_steps: Number of steps to move the agent.
            n_episodes: Number of episodes to move the agent.
            render: Whether to render the environment or not.
      */
      var policy_cond: bool = false;
      return run(n_steps, n_episodes, policy_cond, render, init_states);
    }

    proc run(n_steps: int, n_episodes: int, policy_cond: bool, render: bool, init_states: [] real) {
      assert((n_episodes != nil && n_steps == nil && init_states == nil) || (n_episodes == nil && n_steps != nil && init_states == nil) || (n_episodes == nil && n_steps == nil && init_states != nil));

      if init_states != nil {
        this.n_episodes = init_states.length;
      } else {
        this.n_episodes = n_episodes;
      }

      if n_steps != nil {
        var move_cond: bool = this.total_steps < n_steps;
        return run_impl(move_cond, policy_cond, render, init_states);
      }
    }

    proc run_impl(move_cond: bool, policy_cond: bool, render: bool, init_states: [] real) {
      this.total_episodes = 0;
      this.total_steps = 0;
      this.current_episodes = 0;
      this.current_steps = 0;

      var dataset = [];
          end: bool = true;

      while move_cond {
        if end {
          this.reset(init_states);
        }
        
        var sample = Entity.step(render);
        dataset.append(sample);
        this.total_steps += 1;
        this.current_steps += 1;

        if sample[-1] {
          this.total_episodes += 1;
          this.current_episodes += 1;
        }

        if policy_cond {
          this.agent.fit(dataset);
          this.current_episodes = 0;
          this.current_steps = 0;

          for callback in this.callbacks (
            var call_params = ['dataset' => dataset];
            callback(call_params);
          )
          dataset = [];
        }
        end = sample[-1];
      }
      this.agent.stop();
      this.env.stop();
      return dataset;
    }

    proc step(render: bool) {
      /*
        Agent takes a single step.

          Args:
              render: Whether to render or not.
          Returns:
              A tuple containing the previous state, the action sampled by the
              agent, the reward obtained, the state reached, the absorbing flag
              of the reached state, and the last step flag.
      */
      var action = this.agent.draw_action(this.state);
      var next_state, reward, absorbing, _ = this.env.step(action);
      this.episode_steps += 1;

      if render {
        this.env.render();
      }

      var last = !(this.episode_steps < this.env.info.horizon && !absorbing);
      var state = this.state;
      this.state = next_state.copy();
      return state, action, reward, next_state, absorbing, last;
    }

    proc reset(init_states=nil) {
      /*
        Reset the inital state of the agent.
      */
      if init_states == nil || this.total_episodes_counter == this.n_episodes {
        init_states = nil;
      } else {
        init_states = init_states[this.total_episodes_counter];
      }

      this.state = this.env.reset(init_states);
      this.agent.episode_start();
      this.agent.next_action = None;
      this.episode_steps = 0;
    }
  }
}
