module Kortex {
  class Distribution {
    /*
      Interface for Distributions to represent a generic probability distribution.
      Probability distributions are often used by black box optimization
      algorithms in order to perform exploration in parameter space. In
      literature, they are also known as high level policies.
    */
    proc sample() {
      /*
        Draw a sample from the distribution.

        Returns:
          A random vector sampled from the distribution.
      */
      writeln("Distribution is an abstract class, sample is not available.");
    }

    proc log_pdf(theta: []) {
      /*
        Compute the logarithm of the probability density function in the
        specified point.

        Args:
            theta (array): the point where the log pdf is calculated.
        Returns:
            The value of the log pdf in the specified point.
      */
      writeln("Distribution is an abstract class, log_pdf is not available.");
    }

    proc mle(theta: [], weights: []=nil) {
      /*
        Compute the (weighted) maximum likelihood estimate of the points,
        and update the distribution accordingly.

        Args:
            theta (array): a set of points, every row is a sample
            weights (array): a vector of weights. If specified
              the weighted maximum likelihood estimate is computed
              instead of the plain maximum likelihood. The number of
              elements of this vector must be equal to the number of
              rows of the theta matrix.
      */
      writeln("Distribution is an abstract class, mle is not available.");
    }

    proc diff_log(theta: []) {
      /*
        Compute the derivative of the gradient of the probability denstity
        function in the specified point.

        Args:
            theta (array): the point where the gradient of the log pdf is
            calculated
        Returns:
            The gradient of the log pdf in the specified point.
      */
      writeln("Distribution is an abstract class, diff_log is not available.");
    }

    proc diff(theta: []) {
      /*
        Compute the derivative of the probability density function, in the
        specified point. Normally it is computed w.r.t. the
        derivative of the logarithm of the probability density function,
        exploiting the likelihood ratio trick.

        Args:
            theta (array): the point where the gradient of the pdf is
              calculated.
        Returns:
            The gradient of the pdf in the specified point.
      */
      return this(theta) * diff_log(theta);
    }

    proc get_parameters() {
      /*
        Getter.

        Returns:
          The current distribution parameters.
      */
      writeln("Distribution is an abstract class, get_parameters is not available.");
    }

    proc set_parameters(rho: []) {
      /*
        Setter.

        Args:
          rho (array): The vector of the new parameters to be used by
            the distribution.
      */
      writeln("Distribution is an abstract class, set_parameters is not available.")
    }

    proc parameters_size() {
      /*
        Property.

        Returns:
          The size of the distribution parameters.
      */
      writeln("Distribution is an abstract class, parameters_size is not available.")
    }
  }
}
