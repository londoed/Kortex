module Kortex {
  proc eligibility_trace(shape: [] int, name: string="replacing") {
    /*
      Factory method to create an eligibility trace of the provided type.

      Args:
          shape (array): shape of the eligibility trace table.
          name (string): type of the eligibility trace.

      Returns:
          The eligibility trace table of the provided shape and type.
    */
    if name == "replacing" {
      return ReplacingTrace(shape);
    } else if name == "accumulating" {
      return AccumulatingTrace(shape);
    } else {
      return Error.message("Unknown type of trace.");
    }
  }

  class ReplacingTrace: Table {
    proc reset() {
      this.table = 0.0;
    }

    proc update(state, action) {
      this.table[state, action] = 1.0;
    }
  }

  class AccumulatingTrace: Table {
    proc reset() {
      this.table = 0.0;
    }

    proc update(state, action) {
      this.table[state, action] += 1;
    }
  }
}
