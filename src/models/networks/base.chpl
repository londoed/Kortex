module Kortex {
  // Numpy functions needed: random.randn(), size, shape
  // Python functions needed:
  use LinearAlgebra;

  class NeuralNetworkBase {
    /*
      The base class for a neural network.
    */
    proc init(num_neurons_layers: [], activation: string='relu', opt_fn='adam') {
      /*
        Constructor.

        Args:
          num_neurons_layers (Array): the number of neurons in each respective layer of the list.
          activation (string): the activation function to be used.
          opt_fn (string): the optimization function to be used.
      */
      this.num_layers = num_neurons_layers.length;
      this.num_neurons_layers = num_neurons_layers;
      this.biases_nitem = sum(sizes[1..]);

      for y in sizes[1..] {
        this.biases = randn(y, 1);
      }

      for x,y in zip(sizes[..-1], sizes[1..]) {
        this.weights = randn(y, x);
      }

      for i in 0..#(this.num_layers - 2) {
        this.weight_nitem = sum([this.weights[i].size])
      }
    }

    if activation == 'relu' {
      this.activation = relu();
    } else if activation == 'sigmoid' {
      this.activation = sigmoid();
    } else if activation == 'tanh' {
      this.activation = tanh();
    }

    if opt_fn == 'adam' {
      this.opt_fn = adam();
    } else if opt_fn == 'sgd' {
      this.opt_fn = sgd();
    } else if opt_fn == 'rmsprop' {
      continue;
    } else if opt_fn == 'evolutionary' {
      continue;
    } else if opt_fn == 'adagrad' {
      this.opt_fn = adagrad();
    }

    proc feed_forward(a: Matrix) {
      for b, w in zip(this.biases, this.weights) {
        a = this.activation(w.dot(a) + b);
      }
    }

    proc score(X: Matrix, y: Matrix) {
      var total_score: real = 0.0;
      for i in 0..#X.shape[0] {
        var predicted = feed_forward(X[i].reshape(-1, 1)),
            actual = y[i].reshape(-1, 1);
        total_score += sum(power(predicted - actual, 2) / 2);
      }
      return total_score;
    }

    proc accuracy(X: Matrix, y: Matrix) {
      var accuracy: int = 0.0;
      for i in 0..#X.shape[0] {
        var output = feed_forward(X[i].reshape(-1, 1));
        accuracy += (argmax(output) == argmax(y[i])): int;
      }
      return accuracy / X.shape[0] * 100;
    }
  }
}
