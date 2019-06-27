module Kortex {
// Numpy functions needed: concatenate(), zeros()
  use LinearAlgebra;

  class TilesFeatures: FeaturesImplementation {
    proc init(tiles: Matrix) {
      if tiles.type == Array {
        this. tiles = tiles;
      } else {
        this.tiles = [tiles];
      }
      this.size = 0;

      for tiling in this.tiles {
        this.size += tiling.size;
      }
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
        var out = zeros(),
            offset = 0;
        for tiling in this.tiles {
          var idx = tiling(s);
          if idx != nil {
            out[idx + offset] = 1.0;
          }
          offset += tiling.size;
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
      return this.size;
    }
  }
}
