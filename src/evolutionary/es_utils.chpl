module Kortex {
  // Numpy functions needed: ndim, argsort(), arange(), ravel(), reshape(), shape, mean()
  use LinearAlgebra;

  proc compute_rank(x: Matrix) {
    assert(x.ndim == 1);
    var ranks = Matrix(x.length): int;
    ranks[x.argsort()] = arange(x.length);

    return ranks;
  }

  proc compute_centered_ranks(x: Matrix) {
    var y = compute_rank(x.ravel()).reshape(x.shape): real;
    y /= (x.size - 1);
    y -= 0.5;

    return y;
  }

  proc compute_weight_decay(weight_decay: real, model_param_list: []) {
    var model_param_grid = Matrix(model_param_list);
    return -weight_decay * mean(model_param_grid * model_param_grid, axis=1);
  }
}
