module Kortex {
  // Numpy functions needed: random.choice()

  use LinearAlgebra;

  class ReplayMemory {
    /*
      This class implements functions to manage replay memory as the one
      used in "Human-Level Control Through Deep Reinforcement Learning"
      by Mnih V. et al..
    */
    proc init(init_size: int, max_size: int) {
      /*
        Constructor.

        Args:
            init_size (int): initial number of elements in the replay memory.
            max_size (int): maximum number of elements that the replay memory
                can contain.
      */
      this.init_size = init_size;
      this.max_size = max_size;
      reset();
    }

    proc add(dataset: []) {
      /*
        Add elements to the replay memory.

        Args:
            dataset (array): list of elements to add to the replay memory.
      */
      for i in 0..#dataset.length {
        this.states[this.idx] = dataset[i][0];
        this.actions[this.idx] = dataset[i][1];
        this.rewards[this.idx] = dataset[i][2];
        this.next_states[this.idx] = dataset[i][3];
        this.absorbing[this.idx] = dataset[i][4];
        this.last[this.idx] = dataset[i][5];
        this.idx += 1;

        if this.idx == this.max_size {
          this.full = true;
          this.idx = 0;
        }
      }
    }

    proc get(n_samples: int) {
      /*
        Returns the provided number of states from the replay memory.

        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
      */
      if this.current_sample_idx + n_samples >= this.sample_idxs {
        this.sample_idxs = random_choice(this.size, this.size, replace=false);
        this.current_sample_idx = 0;
      }

      var start = this.current_sample_idx,
          stop = start + n_samples;

      this.current_sample_idx = stop;

      var s = [],
          a = [],
          r = [],
          ss = [],
          ab = [],
          last = [];

      for i in this.sample_idxs[start..stop] {
        s.append(Matrix(this.states[i]));
        a.append(this.actions[i]);
        r.append(this.rewards[i]);
        ss.append(Matrix(this.next_states[i]));
        ab.append(this.absorbing[i]);
        last.append(this.last[i]);
      }
      return Tuple(Matrix(s), Matrix(a), Matrix(r), Matrix(ss), Matrix(ab), Matrix(last));
    }

    proc reset() {
      /*
        Reset the replay memory.
      */
      this.idx = 0;
      this.full = false;
      this.states = [nil for _ in 0..#this.max_size];
      this.actions = [nil for _ in 0..#this.max_size];
      this.rewards = [nil for _ in 0..#this.max_size];
      this.next_states = [nil for _ in 0..#this.max_size];
      this.absorbing = [nil for _ in 0..#this.max_size];
      this.last = [nil for _ in 0..#this.max_size];
      this.sample_idxs = random_choice(this.init_size, this.init_size, replace=false);
      this.current_sample_idx = 0;
    }

    proc initialized() {
      /*
        Returns:
          Whether the replay memory has reached the number of elements that
          allows it to be used.
      */
      return this.size > this.init_size;
    }

    proc size() {
      /*
        Returns:
          The number of elements contained in the replay memory.
      */
      if !this.full {
        return this.idx;
      } else {
        return this.max_size;
      }
    }
  }
}
