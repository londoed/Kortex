module Kortex {
  // Numpy functions needed;
  use LinearAlgebra;

  class GeneticOptimizer {
    proc init(n_populations: int, network_size: int, mutation_rate: real, crossover_rate: real, retain_rate: real,
              x_data: [], y_data: []) {
      this.n_populations = n_populations;
      this.network_size = network_size;
      this.nets = [Network(this.network_size) for i in 0..this.n_populations];
      this.mutation_rate = mutation_rate;
      this.crossover_rate = crossover_rate;
      this.retain_rate = retain_rate;
      this.x_data = x_data;
      this.y_data = y_data;
    }

    proc get_random_point(point_type: string) {
      var nn = this.nets[0],
          layer_index, point_index = random.randint(0, nn.num_layers - 2), 0;
      if point_type == "weight" {
        var row = random.randint(0, nn.weights[layer_index].shape[0] - 1),
            col = random.randint(0, nn.weights[layer_index].shape[1] - 1),
            point_index = (row, col);
      } else if point_type == "bias" {
        var point_index = random.randint(0, nn.biases[layer_index].size - 1);
      }
      return (layer_index, point_index);
    }

    proc get_all_scores() {
      for net in this.nets {
        return [net.score(this.x_data, this.y_data)];
      }
    }

    proc get_all_accuracy() {
      for net in this.nets {
        return [net.accuracy(this.x_data, this.y_data)];
      }
    }

    proc crossover(father: Individual, mother: Individual) {
      var nn = deepcopy(father);

      for _ in 0..#this.nets[0].bias_nitem {
        var layer, point = get_random_point("bias");
        if random.uniform(0, 1) < this.crossover_rate {
          nn.biases[layer][point] = mother.biases[layer][point];
        }
      }

      for _ in 0..#this.nets[0].weight_nitem {
        var layer, point = get_random_point("weight");
        if random.uniform(0, 1) < this.crossover_rate {
          nn.weights[layer][point] = mother.weights[layer][point]
        }
      }
      return nn;
    }

    proc mutation(child: Child) {
      var nn = deepcopy(child);

      for _ in 0..#this.nets[0].bias_nitem {
        var layer, point = get_random_point("bias");
        if random.uniform(0, 1) < this.mutation_rate {
          nn.biases[layer][point] += random.uniform(-0.5, 0.5);
        }


      for _ in 0..#this.nets[0].weight_nitem {
        var layer, point = get_random_point("weight");
        if random.uniform(0, 1) < this.mutation_rate {
          nn.weights[layer][point[0], point[1]] += random.uniform(-0.5, 0.5);
        }
      }
    }
    return nn;
  }
}
