use Kortex;

// Single Agent Training \\

var env = generate_simple_chain(state_n=5, goal_states=[2], prob=0.8, rew=1, gamma=0.9),
    epsilon = new Parameter(value=0.15),
    pi = new EpsilonGreedy(epsilon=epsilon),
    learning_rate = new Parameter(value=0.01),
    num_agents = new Parameter(value=100)
    alg_params = ["learning_rate" => learning_rate],
    agent = new QLearning(pi, env.info, alg_params);

var alg = new Entity(agent, env);

alg.fit(n_steps=10_000, n_steps_per_fit=1);
alg.evaluate(n_episodes=100, render=true);


// Multi Agent Training \\

var env = generate_simple_chain(state_n=5, goal_states=[2], prob=0.8, rew=1, gamma=0.9),
    epsilon = new Parameter(value=0.15),
    pi = new EpsilonGreedy(epsilon=epsilon),
    learning_rate = new Parameter(value=0.01),
    num_agents = new Parameter(value=50),
    alg_params = ["learning_rate" => learning_rate, "num_agents" => num_agents],
    agents = new DistributedQLearning(pi, env.info, alg_params);

var alg = new DistributedEntity(agents, env);

alg.fit(n_steps=10_000, n_steps_per_fit=1);
alg.evaluate(n_episodes=100, render=true);
