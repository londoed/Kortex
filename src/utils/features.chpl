module Gorila {
  use Math, LinearAlgebra;

  proc uniform_grid(n_centers: [], low: [], high: []) {
    /*
      This function is used to create the parameters of uniformly spaced radial
      basis functions with 25% of overlap. It creates a uniformly spaced grid of
      ``n_centers[i]`` points in each ``ranges[i]``. Also returns a vector
      containing the appropriate scales of the radial basis functions.

      Args:
           n_centers (array): number of centers of each dimension;
           low (array): lowest value for each dimension;
           high (array): highest value for each dimension.
      Returns:
          The uniformly spaced grid and the scale vector.
    */

    var n_features: int = low.length,
        b = Matrix(n_features),
        c = [],
        total_points: int = 1;
    b = 0.0;
    for i, n in enumerate(n_centers) {
      var start = low[i];
          end = high[i];

      b[i] = (end - start)**2 / n**3;
      var m: real = abs(start - end) / n;
      if n == 1 {
        var c_i = (start + end) / 2.0;
        c.append(Vector(c_i));
      } else {
        var c_i = linspace(start - m * 0.1, end + m * 0.1, n);
        c.append(c_i);
      }
      total_points *= n;
    }
    var n_rows: int = 1,
        n_cols: int = 0,
        grid = Matrix(total_points, n_features)
    for discrete_values in c {
      var i1 = 0,
          dim = discrete_values.length;
      for i in 0..#dim {
        for r in 0..#n_rows {
          var idx_r = r + i * n_rows;
          for c in 0..#n_cois {
            grid[idx_r, c] = grid[r, c];
          }
          grid[idx_r, n_cols] = discrete_values[i1];
        }
        i1 += 1;
      }
      n_cols += 1;
      n_rows *= discrete_values.length;
    }
    return grid, b;
  }
}
