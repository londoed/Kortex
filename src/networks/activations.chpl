module Gorila {
  use LinearAlgebra,
      Norm,
      Random;

  proc relu(x: real) {
    if x < 0 {
      return 0;
    } else {
      return x;
    }
  }

  proc drelu(x: real) {
    if x < 0 {
      return 0;
    } else {
      return 1;
    }
  }

  proc sigmoid(x: real) {
    return (1 / (1 + exp(-x)));
  }

  proc dsigmoid(x: real) {
    return sigmoid(x) * (1 - sigmoid(x));
  }

  proc tanh(x: real) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
  }

  proc dtanh(x: real) {
    return 1 - (tanh(x))**2;
  }
}
