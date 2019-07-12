module Kortex {
  // Numpy functions needed: zeros(), arange(), random.uniform(), argmax(), average()
  // Bitstring functions needed: BitArray
  // Scipy functions needed: stats.rv_discrete()
  // Python functions needed: random.seed(), enumerate() random.randint(), all()
  use LinearAlgebra, Random, Math;

  class GeneticAlgorithm {
    /*
      Base class for implementing a genetic algorithm.
    */
    proc init(args) {
      /*
        Constructor.

        Args:
            l (int): the size of the bit strings (each bit is a gene).
            N (int): the size of the population.
            G (int): the number of generations.
            pr_mutation (real): the probability of mutation among genes.
            pr_crossover (real): the probability of crossover among genes.
            population (Array): a list of bit strings of size N.
            current_offspring (Array): current generation of children.
            n_runs (int): number of times to run the algorithm.
            learn (bool): whether to learn the offspring or not.
            NG (int): number of guesses to use when learning the offspring.
            ff (function): the fitness function to use.
            ce (bool): whether to inflict sudden change of environment on
                the final population.
            rs: seed for the random number generator.
      */

      // Algorithm parameters
      this.args = args;
      this.l: int = args.l;
      this.N: int = args.N;
      this.G: int = args.G;
      this.pr_mutation: real = args.pm;
      this.pr_crossover: real = args.pc;
      seed(args.rs);
      this.population = [];
      this.current_offspring: [];
      this.n_runs: int = args.n_runs;
      this.NG: int = args.NG;
      this.learn: bool = args.learn;
      this.ff = args.ff;
      this.ce: bool = args.ce;
      this.max_recovery: int = 100;

      // Helper objects
      this.orig_fitness_vals = zeros((this.G + this.max_recovery, this.N));
      this.norm_fitness_vals = zeros((this.G + this.max_recovery, this.N));
      this.total_fitness_vals = zeros((this.G + this.max_recovery, this.N));
      this.pr_mut_dist: [] = nil;
      this.pr_cr_dist: [] = nil;
      this.env_state: int = 0;

      // Statistics objects
      this.avg_fitness_vals = zeros((this.n_runs, 2, this.G + this.max_recovery));
      this.best_fitness_vals = zeros((this.n_runs, 2, this.G + this.max_recovery));
      this.num_correct_bits = zeros((this.n_runs, 2, this.G + this.max_recovery));
      this.recovery_time: int = 0;
    }

    proc init_population() {
      this.population = [];
      this.current_offspring = [];
      for i in 0..#this.N {
        var tmp_bitstring = ''.join(random.choice("01") for _ in 0..#this.l);
            tmp_bitstring = new BitArray(bin=tmp_bitstring);
        this.population.append(tmp_bitstring);
      }
    }

    proc init_environment() {
      if this.env_state != 0 {
        this.recovery_time = 0;
      }
    }

    proc generate_probability_distributions() {
      var xk = arange(2),
          pk1 = (1 - this.pr_mutation, this.pr_mutation),
          pk2 = (1 - this.pr_crossover, this.pr_crossover);
      this.pr_mut_dist = rv_discrete(name="pr_mut_dist", values=(xk, pk1));
      this.pr_cr_dist = rv_discrete(name="pr_cr_dist", values=(xk, pk2));
    }

    proc fitness(g: int, n_run: int, ff: FitnessFunction) {
      var total_fitness: int = 0;

      for i, bitstring in enumerate(this.population) {
        var bit_sum = bitstring.uint,
            fitness_val = this.ff(bit_sum, this.l);
        this.orig_fitness_vals[g][i] = fitness_val;
        total_fitness += fitness_val;
      }

      if total_fitness > 0 {
        this.norm_fitness_vals[g] = this.orig_fitness_vals[g] / total_fitness;
      }

      var prev_norm_fitness_val = 0;

      for i in 0..#this.N {
        this.total_fitness_vals[g][i] = (this.norm_fitness_vals[g][i] + prev_norm_fitness_val);
        prev_norm_fitness_val = this.total_fitness_vals[g][i];
      }
    }

    proc select_individual(g: int) {
      var rand_nums = random.uniform(0, 1, 2),
          prev_individual_fit = 0,
          j: int = 0;

      while true {
        if j >= this.N {
          j = 0;
          rand_nums = random.uniform(0, 1, 2);
        }

        var individual_fit = this.total_fitness_vals[g][j];

        if rand_nums[0] < this.total_fitness_vals[g][0] {
          this.parents[g][0] = this.population[0];
        }

        if j != 0 {
          if prev_individual_fit <= rand_nums[0] <= individual_fit {
            this.parents[g][0] = this.population[j];
            break;
          }
        }

        prev_individual_fit = individual_fit;
        j += 1;
      }

      prev_individual_fit = 0;
      j = 0;
      var cycles = 0;

      while true {
        if j >= this.N {
          cycles += j;

          if cycles >= 100 {
            this.parents[g][1] = this.parents[g][0];
            break;
          } else {
            j = 0;
            rand_nums = random.uniform(0, 1, 2);
          }

          individual_fit = this.total_fitness_vals[g][j];

          if rand_nums[1] < this.total_fitness_vals[g][0] {
            this.parents[g][1] = this.population[0];
            break;
          }

          if j != 0 {
            if prev_individual_fit <= rand_nums[1] <= individual_fit {
              if this.population[j] != this.parents[g][0] {
                this.parents[g][1] = this.population[j];
                break;
              }
            }
          }
          prev_individual_fit = individual_fit;
          j += 1;
        }
      }
    }

    proc mutate(g: int) {
      for parent in this.parents[g] {
        for index_bit in 0..this.l {
          var to_mutate = this.pr_mut_dist.rvs(size=1);

          if to_mutate {
            parent.invert(index_bit);
          }
        }
      }
    }

    proc crossover(g: int) {
      var to_crossover = this.pr_cr_dist.rvs(size=1);

      if to_crossover {
        var c1 = new BitArray(length=this.l),
            c2 = new BitArray(length=this.l),
            crossover_bit = randint(0, this.l);
        c1.overwrite(this.parents[g][0][..crossover_bit], 0);
        c1.overwrite(this.parents[g][1][..(this.l - crossover_bit)], crossover_bit);
        c2.overwrite(this.parents[g][0][..crossover_bit], 0);
        c2.overwrite(this.parents[g][1][..(this.l - crossover_bit)], crossover_bit);
        this.current_offspring.append(c1);
        this.current_offspring.append(c2);
      } else {
        this.current_offspring.append(this.parents[g][0]);
        this.current_offspring.append(this.parents[g][0]);
      }
    }

    proc learn_offspring(g: int, ff: FitnessFunction) {
      for child in this.current_offspring {
        for guess in 0..this.NG {
          var current_child = child.copy(),
              max_fitness = 0;
          for ibit in 0..this.l {
            if random.choice([0, 1]) {
              if current_child[ibit] {
                current_child.set(false, ibit);
              } else {
                current_child.set(true, ibit);
              }
            }
          }
          var bit_sum = current_child.uint,
              current_fitness = ff(bit_sum, this.l);
          max_fitness = max(current_fitness, max_fitness);
          if current_fitness == max_fitness {
            child = current_child;
          }
        }
      }
    }

    proc compute_statistics(g: int, n_run: int) {
      var index_bi = this.orig_fitness_vals[g].argmax(),
          bi_bitstring = this.population[index_bi],
          individual_num_correct_bits = bi_bitstring.count(1);
      this.num_correct_bits[n_run][this.env_state][g] = individual_num_correct_bits;
      this.best_fitness_vals[n_run][this.env_state][g] = this.orig_fitness_vals[g][index_bi];
      this.avg_fitness_vals[n_run][this.env_state][g] = average(this.orig_fitness_vals[g]);
    }

    proc check_population_recovery(g: int, n_run: int) {
      var checks = [];
      checks.append((this.best_fitness_vals[n_run][1][g] > this.best_fitness_vals[n_run][0][this.G - 1]));
      checks.append((this.avg_fitness_vals[n_run][1][g] > this.avg_fitness_vals[n_run][0][this.G - 1]));
      for check in checks {
        writeln(check);
      }
      if all(checks) {
        return true;
      }
    }

    proc reproduce(n_run: int, g: int) {
      this.fitness(g, n_run, this.ff);
      for i in 0..#(this.N / 2) {
        select_individual(g);
        crossover(g);
        mutate(g);
      }

      if this.learn {
        learn_offspring(g, this.ff);
      }
      compute_statistics(g, n_run);
      this.population = this.current_offspring;
      this.current_offspring = [];
    }

    proc run() {
      for n_run in 0..#this.n_runs {
        init_population();
        this.env_state = 0;
        this.ff = fitness_function1();

        for g in 0..#this.G {
          reproduce(n_run, g);
        }

        if this.ce {
          this.env_state = 1;
          init_env();
          //this.ff = fitness_function2();
          while true {
            g += 1;
            this.recovery_time += 1;
            reproduce(n_run, g);
          }
        }
      }
      return new Tuple(this.avg_fitness_vals, this.best_fitness_vals, this.num_correct_bits);
    }
  }
}
