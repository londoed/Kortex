use Kortex, LinearAlgebra, Arkouda, Random;

class CriticNetwork: NeuralNetworkBase {
  proc init(input_shape, output_shape, n_features, kwargs) {
    super().init();

    var n_input = input_shape[-1],
        n_output = output_shape[0];

    this.network = new Sequential(
      Dense(n_input, n_features, initializer=GlorotUniform()),
      relu(),
      Dense(n_features, n_features, initializer=GlorotUniform()),
      relu(),
      Dense(n_features, n_output, initializer=GlorotUniform()),
      softmax());
  }

  proc call(state, action) {
    var state_action = concatenate(state, action, axis=1),
        q = this.network;
    return squeeze(q);
  }
}

class ActorNetwork: NeuralNetworkBase {
  proc init(input_shape, output_shape, n_features, kwargs) {
    super().init();

    var n_input = input_shape[-1],
        n_output = output_shape[0];

    this.network = new Sequential(
      Dense(n_input, n_features, initializer=GlorotUniform()),
      relu(),
      Dense(n_features, n_features, initializer=GlorotUniform()),
      relu(),
      Dense(n_features, n_output, initializer=GlorotUniform()),
      softmax());
  }

  proc call(state, action) {
    a = this.network;
    return a;
  }
}

proc run(alg, n_epochs, n_steps, n_steps_test) {
  random.seed(91);

  // Environment
  var horizon: int = 500,
      gamma: real = 0.99,
      gamma_eval: real = 1.0,
      env = new Gym("Pendulum-v0", horizon, gamma);

  // Policy
  var policy_class = new OrnsteinUhlenbeckPolicy(),
      policy_params = ["sigma" => ones(1) * 0.2, "theta" => 0.15, "dt" => 1e-2];

  // Settings
  var init_replay_size: int = 500,
      max_replay_size: int = 5000,
      batch_size: int = 156,
      n_features: int = 90;

  // Approximator
  var actor_approximator = new NeuralNetworkApproximator(),
      actor_input_shape = env.info.observation_space.shape,
      actor_params = ["network" => ActorNetwork(), "optimizer" => adam(learning_rate=0.001),
                      "n_features" => n_features, "input_shape" => actor_input_shape,
                      "output_shape" => env.info.action_space.shape];

  var critic_approximator = new NeuralNetworkApproximator(),
      critic_input_shape = (actor_input_shape[0] + env.info.action_space.shape[0],),
      critic_params = ["network" => CriticNetwork, "optimizer" => adam(learning_rate=0.001),
                       "loss" => sparse_categorical_cross_entropy(), "n_features" => n_features,
                       "input_shape" => critic_input_shape, "output_shape" => (1,)];

  // Agent
  var agent = alg(actor_approximator, critic_approximator, policy_class,
                  env.info, batch_size=batch_size, init_replay_size=init_replay_size,
                  max_replay_size=max_replay_size, tau=0.001, actor_params=actor_params,
                  critic_params=critic_params, policy_params=policy_params);

  // Algorithm
  var entity = new Entity(agent, env);
  entity.fit(n_steps=init_replay_size, n_steps_per_fit=init_replay_size);

  var dataset = entity.evaluate(n_steps=n_steps_test, render=false),
      J = compute_J(dataset, gamma_eval);

  writeln("J: ", mean(J));

  for n in 0..#n_epochs {
    writeln("Epoch: ", n);
    entity.fit(n_steps=n_steps, n_steps_per_fit=1);
    dataset = entity.evaluate(n_steps=n_steps_test, render=true);
    J = compute_J(dataset, gamma_eval);
    writeln("J: ", mean(J));
  }
}

proc main() {
  var algs = [DDPG(), IMPALA()];

  for alg in algs {
    writeln("Algorithm: ", alg.name);
    run(alg=alg, n_epochs=50, n_steps=1_000, n_steps_test=2_000);
  }
}
