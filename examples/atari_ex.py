import tensorflow as tf
import sonnet as snt
import kortex as kx

# Chapel's NumPy interface #
import arkouda as ak

# Calling from Chapel #
kx.chpl_setup()

# Environment wrapped around Gym's Atari suite #
env = kx.Atari('BreakoutDeterministic-v4', screen_width=84,
                       screen_height=84, ends_at_life=True,
                       history_length=4, max_no_op_actions=30)

# Can use other python libraries and packages for network architecture #
model = snt.Sequential([
    snt.Conv2D(32, 3, 1), tf.nn.relu,
    snt.LSTM(32),
    snt.Conv2D(32, 3, 1) ,tf.nn.relu,
    snt.Flatten(),
    snt.Linear(10)
])

model.optimizer = snt.optimizers.Adam(0.1)

# Building an IMPALA agent that runs on 10 GPUs with 25 agents playing in parallel #
num_agents = 25
agents = kx.IMPALA(num_agents, model, pi, env.info,
                   approx_params, batch_size, n_approx=1,
                   init_replay_size, max_replay_size, 
                   target_update_freq, target_locale='gpu',
                   num_locales=10)

# Building the RL algorithm object #
alg = kx.Entity(agents, env)

n_epochs = 100
gamma_eval = 1.0

# Training iteration #
for epoch in range(n_epochs):
    alg.fit(n_steps=init_replay_size, n_steps_per_fit=init_replay_size)
    alg_eval = alg.evaluate(n_episodes=50, render=False)
    J = kx.compute_J(alg_eval, gamma_eval)
    if epoch % 2 == 0:
        print(f"Epoch: {epoch}, J: {ak.mean(J)}")

# Closes the connection to Chapel library #
kx.chpl_cleanup()
