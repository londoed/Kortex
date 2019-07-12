module Kortex {
  // Numpy functions needed: array_equal(), inf, random.choice()
  // Python functions needed: // (tried casting the division into an int and subtracting 1)
  use LinearAlgebra;

  class AbstractGridWorld: Environment {
    /*
      Abstract class to build a grid world.
    */
    proc init(env_info: ENVInfo, height: int, width: int, start: Tuple, goal: Tuple) {
      /*
        Constructor.

        Args:
            height (int): height of the grid.
            width (int): width of the grid.
            start (tuple): x-y coordinates of the goal.
            goal (tuple): x-y coordinates of the goal.
      */
      assert(!array_equal(start, goal));
      assert(goal[0] < height && goal[1] < width);
      this.state = nil;
      this.height = height;
      this.width = width;
      this.start = start;
      this.goal = goal;

      // Visualization
      this.viewer = new Viewer(this.width, this.height, 500, (this.height * 500 / this.width): int - 1);

      super().init(env_info);
    }

    proc reset(state: Matrix=nil) {
      if state == nil {
        state = convert_to_int(new_state, this.width);
      }

      this.state = state;
      return this.state;
    }

    proc step(action: Matrix) {
      for row in 1..this.height {
        for col in 1..this.width {
          this.viewer.line(Matrix(col, 0), Matrix(col, this.height));
          this.viewer.line(Matrix(0, row), Matrix(this.width, row));
        }
      }
      
      var goal_center = Matrix(0.5 + this.goal[1], this.height - (0.5 + this.goal[0]))
      this.viewer.square(goal_center, 0, 1, (0, 255, 0));

      var start_grid = convert_to_grid(this.start, this.width),
          start_center = Matrix(0.5 + start_grid[1], this.height - (0.5 + start_grid[0]));
      this.viewer.square(start_center, 0, 1, (255, 0, 0));

      var state_grid = convert_to_grid(this.state, this.width).
          state_center = Matrix(0.5 + state_grid[1], this.height - (0.5 + state_grid[0]));
      this.viewer.circle(state_center, 0.4, (0, 0, 255));
      this.viewer.display(0.1);
    }

    proc step(state: Matrix, action: Matrix) {
      writeln("AbstractGridWorld is an abstract class, step is not available");
    }

    proc convert_to_grid(state: Matrix, width: int) {
      return Matrix((state[0] / width): int - 1, state[0] % width);
    }

    proc convert_to_int(state: Matrix, width: int) {
      return Matrix(state[0] * width + state[1]);
    }
  }

  class GridWorld: AbstractGridWorld {
    /*
      Class to implement a standard grid world.
    */
    proc init(height: int, width: int, goal: Matrix, start: Tuple=(0, 0)) {
      // MDP Properties
      var observation_space = Discrete(height * width).
          action_space = Discrete(4),
          horizon: int = 100,
          gamma: real = 0.9,
          env_info = new ENVInfo(observation_space, action_space, gamma, horizon);
      super().init(env_info, height, width, start, goal);
    }

    proc step(state: Matrix, action: int) {
      if action == 0 {
        if state[0] > 0 {
          state[0] -= 1;
        }
      } else if action == 1 {
        if state[0] + 1 < this.height {
          state[0] += 1;
        }
      } else if action == 2 {
        if state[1] > 0 {
          state[1] -= 1;
        }
      } else if action == 3 {
        if state[1] + 1 < this.width {
          state[1] += 1;
        }
      }

      if array_equal(state, this.goal) {
        var reward = 10,
            absorbing: bool = true;
      } else {
        var reward = 0,
            absorbing: bool = false;
      }
      return state, reward, absorbing, new AssociativeArray();
    }
  }

  class GridWorldVanHasselt: AbstractGridWorld {
    /*
      A variant of the grid world as presented in:
      "Double Q-Learning". Hasselt H. V.. 2010.
    */
    proc init(height: int=3, width: int=3, goal: Tuple=(0, 2), start: Tuple=(2, 0)) {
      // MDP Properties
      var observation_space = Discrete(height * width),
          action_space = Discrete(4),
          horizon = INFINITY;
          gamma: real = 0.95,
          env_info = new ENVInfo(observation_space, action_space, gamma, horizon);
      super().init(env_info, height, width, start, goal);
    }

    proc step(state: Matrix, action: int) {
      if array_equal(state, this.goal) {
        var reward: int = 5,
            absorbing: bool = true;
      } else {
        if action == 0 {
          if state[0] > 0 {
            state[0] -= 1;
          }
        } else if action == 1 {
          if state[0] + 1 < this.height {
            state[0] += 1;
          }
        } else if action == 2 {
          if state[1] > 0 {
            state[1] -= 1;
          }
        } else if action == 3 {
          if state[1] + 1 < this.width {
            state[1] += 1;
          }
        }
        reward = random_choice([-12, 10]);
        absorbing: bool = false;
      }
      return state, reward, absorbing, new AssociativeArray();
    }
  }
}
