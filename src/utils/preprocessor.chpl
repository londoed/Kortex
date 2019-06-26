module Kortex {
  class Preprocessor {
    /*
      This is the interface class of the preprocessors.
    */
    proc process(x: Matrix) {
      /*
        Compute the preprocessing of the given input according to the type
        of preprocessor.

        Args:
          x {Matrix}: the array to preprocess.

        Returns:
          The preprocessed input data array.
      */
      if x.type == Array {
        assert(x[0].type == Matrix);
        x[0] = compute(x[0]);
      } else {
        assert(x.type == Matrix);
        x = this.compute(x);
      }
      return x;
    }

    proc compute(x: Matrix) {
      /*
        Args:
          x (Matrix): input data.

        Returns:
          The preprocessed data array.
      */
      return Error.message("Preprocessor is an abstract class, compute not available");
    }
  }

  class Scalar: Preprocessor {
    /*
      This class implements the function to binarize the values of
      a matrix according to a provided threshold value.
    */
    proc init(threshold: real, geq: bool=true) {
      /*
        Constructor.

        Args:
            threshold (real): the coefficient to use to scale input data.
            geq (bool): whether the threshold include equal elements
                or not.
      */
      this.threshold = threshold;
      this.geq = geq;
    }

    proc compute(x: Matrix) {
      if this.geq {
        return Matrix(x >= this.threshold): real;
      } else {
        return Matrix(x > this.threshold]: real;
      }
    }
  }

  class Filter: Preprocessor {
    /*
      This class implements the function to filter the values of a
      matrix according to a provided array of indexes.
    */
    proc init(idxs: real) {
      /*
        Constructor.

        Args:
          idxs (real): The array of idxs to use to filter input data.
      */
      this.idxs = idxs;
    }

    proc compute(x: real) {
      return x[this.idxs];
    }
  }
}
