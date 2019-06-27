module Kortex {
  // Numpy functions needed: concatenate(), atleast_2d(), empty()
  // Python functions needed: enumerate()
  use LinearAlgebra;

  class BasisFeatures: FeaturesImplementation {
    proc init(basis: []) {
      this.basis = basis;
    }

    proc call(args: Matrix) {
      if args.length > 1 {
        var x = concat(args, axis=-1);
      } else {
        var x = args[0];
      }

      var y = [];
      x = atleast_2d(x);

      for s in x {
        var out = empty(this.size);

        for i, bf in enumerate(this.basis) {
          out[i] = bf(s);
        }
        y.append(out);
      }

      if y.length == 1 {
        y = y[0];
      } else {
        y = Matrix(y);
      }
      return y;
    }

    proc size() {
      return this.basis.length;
    }
  }
}
