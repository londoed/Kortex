module Kortex {
  // Numpy functions needed: isscalar, zeros, arange
  class Box {
  /*
    This class implements functions to manage continuous states and action
    spaces.
  */
    proc init(low, high, shape: []=nil) {
      /*
        Constructor.

        Args:
            low ([float, array]): the minimum value of each dimension of
                the space. If a scalar value is provided, this value is
                considered as the minimum one for each dimension. If an
                array is provided, each i-th element is considered the
                minimum value of the i-th dimension.
            high ([float, np.array]): the maximum value of dimensions of the
                space. If a scalar value is provided, this value is considered
                as the maximum one for each dimension. If an array is
                provided, each i-th element is considered the maximum value
                of the i-th dimension;.
            shape (array, None): the dimension of the space. Must match
                the shape of ``low`` and ``high``, if they are arrays.
      */
      if shape == nil {
        this.low = low;
        this.high = high;
        this.shape = low.shape;
      } else {
        this.low = low;
        this.high = high;
        this.shape = shape;
        if isscalar(low) && isscalar(high) {
          this.low += ZerosMatrix(shape);
          this.high += ZerosMatrix(shape);
        }
      }
      assert(this.low.shape == this.high.shape);
    }

    proc low() {
      /*
        Returns:
          The minimum value of each dimension of the space.
      */
      return this.low;
    }

    proc high() {
      /*
        Returns:
          The maximum value of each dimension of the space.
      */
      return this.high;
    }

    proc shape() {
      /*
        Returns:
          The dimensions of the space.
      */
      return this.shape;
    }
  }

  class Discrete {
    /*
      This class implements functions to manage discrete states and
      action spaces.
    */
    proc init(n: int) {
      /*
        Constructor.

        Args:
          n (int): The number of values of the space.
      */
      this.values = arange(n);
      this.n = n;
    }

    proc size() {
      /*
        Returns:
          The number of elements of the space.
      */
      return this.n;
    }

    proc shape() {
      /*
        Returns:
          The shape of the space that is always (1,).
      */
      return (1,);
    }
  }
}
