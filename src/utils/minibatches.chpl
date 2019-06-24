module Kortex {
  // Numpy functions needed: arange()
  use LinearAlgebra, Random;

  proc minibatch_number(size: int, batch_size: int) {
    /*
      Function to retrieve the number of batches, given a batch sizes.

      Args:
          size (int): size of the dataset;
          batch_size (int): size of the batches.

      Returns:
          The number of minibatches in the dataset.
    */
    return ceil(size / batch_size): int;
  }

  proc minibatch_generator(batch_size: int, dataset: [] ?t) {
    /*
      Generator that creates a minibatch from the full dataset.

      Args:
          batch_size (int): the maximum size of each minibatch;
          dataset: the dataset to be splitted.

      Returns:
          The current minibatch.
    */
    var size: int = dataset[0].length;
        num_batches = minibatch_number(size, batch_size);
        indexes = arange(0, size, 1);
        shuffle(indexes);

    for i in 0..#num_batches {
      var batches = new Tuple(i * batch_size, min(size, (i + 1) * batch_size));
    }

    for (batch_start, batch_end) in batches {
      var batch = [];
      for i in 0..#dataset.length {
        batch.append(dataset[i][indexes[batch_start..batch_end]]);
      }
      yield batch;
    }
  }
}
