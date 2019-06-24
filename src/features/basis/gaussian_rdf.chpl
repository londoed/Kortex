module Kortex {
  
  use LinearAlgebra;

  class GaussianRBF {
    /*
      Class implementing Gaussian radial basis functions. The value of the feature
      is computed using the formula:

      .. math::
          \sum \dfrac{(X_i - \mu_i)^2}{\sigma_i}

      where X is the input, \mu is the mean vector and \sigma is the scale
      parameter vector.
    */
    proc init(mean: [], scale: [], dimensions: []=nil) {
      /*
        Constructor.

        Args:
            mean (array): the mean vector of the feature.
            scale (array): the scale vector of the feature.
            dimensions (array): list of the dimensions of the input to be
                considered by the feature. The number of dimensions must match
                the dimensionality of ``mean`` and ``scale``.
      */
      this.mean = mean;
      this.scale = scale;
      this.dim = dimensions;
    }

    proc call(x: []) {
      if this.dim != nil {
        x = x[this.dim];
      }
      return exp(-sum((x - this.mean)**2 / this.scale));
    }

    proc str() {
      var name = "GaussianRBF " + this.mean: string + " " + this.scale: string;
      if this.dim != nil {
        name += " " + this.dim: string;
      }
      return name;
    }

    proc generate(n_centers: [], low: [], high: [], dimensions: []=nil) {
      /*

        Factory method to build uniformly spaced gaussian radial basis functions
        with a 25% overlap.

        Args:
          n_centers (list): list of the number of radial basis functions to be
          used for each dimension.
          low (np.ndarray): lowest value for each dimension;
          high (np.ndarray): highest value for each dimension;
          dimensions (list, None): list of the dimensions of the input to be
            considered by the feature. The number of dimensions must match
            the number of elements in ``n_centers`` and ``low``.

        Returns:
          The list of the generated radial basis functions.
      */
      var n_features: int = low.length;
      assert(n_centers.length == n_features);
      assert(low.length == high.length);
      assert(dimensions == nil || n_features == dimensions.length);

      var grid, b = uniform_grid(n_centers, low, high),
          basis = [];
      for i in 0..#grid.length {
        var v = grid[i, ..],
            bf = new GaussianRBF(v, b, dimensions),
            basis.append(bf);
      }
      return basis;
    }
  }
}
