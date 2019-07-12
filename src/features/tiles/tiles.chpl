module Kortex {
  // Python functions needed: enumerate()
  // NumPy functions needed: floor(), empty()
  use LinearAlgebra;

  class Tiles {
    /*
      Class implementing rectangular tiling. For each point in the state space,
      this class can be used to compute the index of the corresponding tile.
    */
    proc init(x_range: [], n_tiles: [], state_components: []=nil) {
      /*
        Constructor.

        Args:
            x_range (array): list of two-elements lists specifying the range of
                each state variable;
            n_tiles (array): list of the number of tiles to be used for each
                dimension.
            state_components (array): list of the dimensions of the input
                to be considered by the tiling. The number of elements must
                match the number of elements in ``x_range`` and ``n_tiles``.
      */
      if x_range[0].type == Array {
        this.range = x_range;
      } else {
        this.range = [x_range];
      }

      if n_tiles.type == Array {
        assert(n_tiles.length == this.range.length);
        this.n_tiles = n_tiles;
      } else {
        this.n_tiles = [n_tiles] * this.range.length;
      }

      this.state_components = state_components;

      if this.state_components != nil {
        assert(this.state_components.length == this.range.length);
      }

      this.size: real = 1.0;

      for s in this.n_tiles {
        this.size *= s;
      }
    }

    proc call(x: []) {
      if this.state_components != nil {
        x = x[this.state_components];
      }

      var multiplier: int = 1,
          tile_index: int = 0;

      for i, (r, N) in enumerate(zip(this.range, this.n_tiles)) {
        if r[0] <= x[i] < r[1] {
          var width = r[1] - r[0],
              component_index = floor(N * (x[i] - r[0]) / width): int;
          tile_index += component_index * multiplier;
          multiplier *= N;
        } else {
          tile_index = nil;
          break;
        }
      }
      return tile_index;
    }

    proc generate(n_tilings: int, n_tiles: [], low: [] real, high: [] real, uniform: bool=false) {
      /*
        Factory method to build ``n_tilings`` tilings of ``n_tiles`` tiles with
        a range between ``low`` and ``high`` for each dimension.

        Args:
            n_tilings (int): number of tilings.
            n_tiles (array): number of tiles for each tilings for each dimension.
            low (Matrix): lowest value for each dimension.
            high (Matrix): highest value for each dimension.
            uniform (bool): if True the displacement for each tiling will
                be w/n_tilings, where w is the tile width. Otherwise, the
                displacement will be k*w/n_tilings, where k=2i+1, where i is the
                dimension index.

        Returns:
            The list of the generated tiles.
      */
      assert(n_tiles.length == low.length == high.length);
      low = Matrix(low);
      high = Matrix(high);

      var tilings = [],
          shift = compute_shift(uniform, low.length),
          width = (high - low) / (Matrix(n_tiles) * n_tilings - shift * n_tilings + shift),
          offset = width;

      for i in 0..#n_tilings {
        var x_min = low - (n_tilings - 1 - i) * offset * shift,
            x_max = high + i * offset;
        for x, y in zip(x_min, x_max) {
          x_range = [x, y];
        }
        tilings.append(Tiles(x_range, n_tiles));
      }
      return tilings;
    }

    proc compute_shift(uniform: bool, n_dims: Matrix(real)) {
      if uniform {
        return 1;
      } else {
        shift = empty(n_dims);

        for i in 0..#n_dims {
          shift[i] = 2 * i + 1;
        }

        return shift;
      }
    }

    proc size() {
      return this.size;
    }
  }
}
