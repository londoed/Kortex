module Kortex {
  // Numpy functions needed:
  // Math functions needed: fmod()
  use LinearAlgebra, Math;

  proc normalize_angle_positive(angle: real) {
    /*
      Wrap the angle between 0 and 2 * pi.

      Args:
          angle (real): angle to wrap.

      Returns:
           The wrapped angle.
    */
    var pi_2 = 2.0 * PI;
    return fmod(fmod(angle, pi_2) + pi_2, pi_2);
  }

  proc normalize_angle(angle: real) {
    /*
      Wraps the angle between -pi and pi.

      Args:
          angle (real): angle to wrap.

      Returns:
          The wrapped angle.
    */
    var a = normalize_angle_positive(angle);
    
    if a > PI {
      a -= 2.0 * PI;
    }
    return a;
  }

  proc shortest_angular_distance(from_angle: real, to_angle: real) {
    return normalize_angle(to_angle - from_angle);
  }
}
