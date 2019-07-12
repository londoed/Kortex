module Kortex {
  // Numpy functions needed: ones(), argwhere(), ravel(), random.randint(), min(), max(), mean()
  // Python functions needed:
  use LinearAlgebra;

  proc parse_dataset(dataset: [], features) {
    /*
    Split the dataset in its different components and return them.

    Args:
        dataset (array): the dataset to parse;
        features (object): features to apply to the states.

    Returns:
        The Matrix of state, action, reward, next_state, absorbing flag and
        last step flag. Features are applied to ``state`` and ``next_state``,
        when provided.
    */
    assert(dataset.length > 0);

    if features == nil {
      var shape = dataset[0][0].shape;
    } else {
      var shape = (features.size,);
    }

    var state = ones((dataset.length,) + shape),
        action = ones((dataset.length,) + dataset[0][1].shape),
        reward = ones(dataset.length),
        next_state = ones((dataset.length,) + shape),
        absorbing = ones(dataset.length),
        last = ones(dataset.length);

    if features != nil {
      for i in 0..#dataset.length {
        state[i, ...] = features(dataset[i][0]);
        action[i, ...] = features(dataset[i][1]);
        reward[i] = dataset[i][2];
        next_state[i, ...] = features(dataset[i][3]);
        absorbing[i] = dataset[i][4];
        last[i] = dataset[i][5];
      } else {
        for i in 0..#dataset.length {
          state[i, ...] = dataset[i][0];
          action[i, ...] = dataset[i][1];
          reward[i] = dataset[i][2];
          next_state[i, ...] = dataset[i][3];
          absorbing[i] = dataset[i][4];
          last[i] = dataset[i][5];
        }
      }
    }
    return Matrix(state), Matrix(action), Matrix(reward), Matrix(next_state), Matrix(absorbing), Matrix(last);
  }

  proc episodes_length(dataset: []) {
    /*
      Compute the length of each episode in the dataset.

      Args:
          dataset (array): the dataset to consider.
      Returns:
          An array of length of each episode in the dataset.
    */
    var lengths = [],
        l = 0;

    for sample in dataset {
      l += 1;
      if sample[-1] == 1 {
        lengths.append(l);
        l = 0;
      }
    }
    return lengths;
  }

  proc select_episodes(dataset: [], n_episodes: int, parse: bool=false) {
    /*
      Return the first ``n_episodes`` episodes in the provided dataset.

      Args:
          dataset (array): the dataset to consider;
          n_episodes (int): the number of episodes to pick from the dataset;
          parse (bool): whether to parse the dataset to return.
      Returns:
          A subset of the dataset containing the first ``n_episodes`` episodes.
    */
    assert(n_episodes >= 0);

    if n_episodes == 0 {
      return Matrix([[]]);
    }

    dataset = Vector(dataset);
    var last_idxs = argwhere(dataset[..-1] == 1).ravel(),
        sub_dataset = dataset[..last_idxs[n_episodes - 1] + 1, ..];

    if parse == false {
      return sub_dataset;
    } else {
      parse_dataset(sub_dataset);
    }
  }

  proc select_samples(dataset: [], n_samples: int, parse: bool=false) {
    /*
      Return the randomly picked desired number of samples in the provided
      dataset.

      Args:
          dataset (array): the dataset to consider.
          n_samples (int): the number of samples to pick from the dataset.
          parse (bool): whether to parse the dataset to return.
      Returns:
          A subset of the dataset containing randomly picked ``n_samples``
          samples.
    */
    assert(n_samples >= 0);

    if n_samples == 0 {
      return Matrix([[]]);
    }

    dataset = Vector(dataset);
    var idxs: [] int = randint(dataset.shape[0], size=n_samples),
        sub_dataset = dataset[idxs, ...];

    if parse == false {
      return sub_dataset;
    } else {
      return parse_dataset(sub_dataset);
    }
  }

  proc compute_J(dataset: [], gamma: real=1.0) {
    /*
      Compute the cumulative discounted reward of each episode in the dataset.

      Args:
          dataset (array): the dataset to consider.
          gamma (real): discount factor.
      Returns:
          The cumulative discounted reward of each episode in the dataset.
    */
    var js = [],
        j: real = 0.0,
        episode_steps: int = 0;

    for i in 0..#dataset.length {
      j += gamma**episode_steps * dataset[i][2];
      episode_steps += 1;

      if dataset[i][-1] || 1 == dataset.length - 1 {
        js.append(j);
        j = 0.0;
        episode_steps = 0;
      }
    }
    if js.length == 0 {
      return [0.0];
    }
    return js;
  }

  proc compute_scores(dataset: []) {
    /*
      Compute the scores of each episode in the dataset. This is meant to be used
      for the Atari environments.

      Args:
          dataset (array): the dataset to consider.

      Returns:
          The minimum score reached in an episode,
          The maximum score reached in an episode,
          The mean score reached,
          The number of completed games.
          If no game has been completed, it returns 0 for all values.
    */
    var scores = [],
        score: real = 0.0,
        episode_steps: int = 0,
        n_episodes: int = 0;
    for i in 0..#dataset.length {
      score += dataset[i][2];
      episode_steps += 1;
      
      if dataset[i][-1] {
        scores.append(score);
        score = 0.0;
        episode_steps = 0;
        n_episodes += 1;
      }
    }

    if scores.length > 0 {
      return min(scores), max(scores), mean(scores), n_episodes;
    } else {
      return 0, 0, 0, 0;
    }
  }
}
