module Gorila {
  proc relu(x) {
    return x * (x > 0);
  }

  proc drelu(x) {
    return 1.0 * (x > 0);
  }
}
