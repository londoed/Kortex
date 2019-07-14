import kortex as kx

############################################
########## Single Agent Training ###########
############################################

kx.chpl_setup()

env = kx.generate_simple_chain(state_n=5, goal_states=[2],
                               prob=0.8, rew=1, gamma=0.9)

epsilon = kx.Parameter(value=0.15)
pi = kx.EpsilonGreedy(epsilon=epsilon)

learning_rate = kx.Parameter(value=0.01)
alg_params = {'learning_rate': learning_rate}

agent = kx.QLearning(pi, env.info, alg_params)
alg = kx.Entity(agent, env)

alg.fit(n_steps=10000, n_steps_per_fit=1)
alg.evaluate(n_episodes=100, render=True)

kx.chpl_cleanup()

###########################################
########## Multi Agent Training ###########
###########################################

kx.chpl_setup()

env = kx.generate_simple_chain(state_n=5, goal_states=[2],
                               prob=0.8, rew=1, gamma=0.9)

epsilon = kx.Parameter(value=0.15)
pi = kx.EpsilonGreedy(epsilon=epsilon)

learning_rate = kx.Parameter(value=0.01)
num_agents = kx.Parameter(value=50)
alg_params = {'learning_rate': learning_rate, 'num_agents': num_agents}

agents = kx.DistributedQLearning(pi, env.info, alg_params)
alg = kx.DistributedEntity(agents, env)

alg.fit(n_steps=10000, n_steps_per_fit=1)
alg.evaluate(n_episodes=100, render=True)

kx.chpl_cleanup()
