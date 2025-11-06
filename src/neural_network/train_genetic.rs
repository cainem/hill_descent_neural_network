use super::genetic_fitness::GeneticFitness;
use super::NeuralNetwork;
use hill_descent_lib::{setup_world, GlobalConstants, TrainingData};
use ndarray::Array2;
use std::ops::RangeInclusive;
use std::sync::Arc;
use std::time::Instant;

impl NeuralNetwork {
    /// Trains the neural network using a genetic algorithm instead of backpropagation.
    ///
    /// This method uses the hill_descent_lib genetic algorithm to optimize network
    /// parameters without computing gradients. The algorithm explores the parameter
    /// space through evolution: organisms (parameter sets) compete based on fitness
    /// (loss), with better organisms producing more offspring.
    ///
    /// Key differences from `train()`:
    /// - No gradient computation (no backpropagation)
    /// - No rigid layer structure requirement
    /// - Much slower but explores broader solution space
    /// - Uses random subset of training data for fitness evaluation
    ///
    /// # Arguments
    /// * `x_train` - Training images as 2D array (rows=examples, cols=784 pixels)
    /// * `y_train` - Training labels as 2D array (rows=examples, cols=10 one-hot)
    /// * `generations` - Number of evolutionary generations to run
    /// * `population_size` - Number of organisms per generation (recommend 500)
    /// * `subset_size` - Number of training examples to evaluate per fitness (recommend 1000)
    ///
    /// # Returns
    /// Tuple of (final_loss, training_time_seconds)
    ///
    /// # Example
    /// ```no_run
    /// use neural_network_scratch::NeuralNetwork;
    /// use ndarray::Array2;
    ///
    /// let mut nn = NeuralNetwork::new(784, 64, 10);
    /// let x_train = Array2::from_shape_fn((1000, 784), |(_, _)| 0.5);
    /// let y_train = Array2::from_shape_fn((1000, 10), |(i, j)| if j == i % 10 { 1.0 } else { 0.0 });
    ///
    /// let (final_loss, training_time) = nn.train_genetic(
    ///     &x_train,
    ///     &y_train,
    ///     100,   // generations
    ///     500,   // population
    ///     1000,  // subset size
    /// );
    /// ```
    ///
    /// # Performance Notes
    /// - Each generation evaluates population_size × subset_size examples
    /// - Much slower than backpropagation but explores more freely
    /// - Use smaller subset_size for faster iterations
    /// - Parallel evaluation via Rayon (automatically used by hill_descent_lib)
    #[allow(clippy::too_many_arguments)]
    pub fn train_genetic(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &Array2<f64>,
        generations: usize,
        population_size: usize,
        subset_size: usize,
    ) -> (f64, f64) {
        println!("Starting genetic algorithm training...");
        println!(
            "Architecture: {}-{}-{}",
            self.input_size, self.hidden_size, self.output_size
        );
        println!("Population size: {}", population_size);
        println!("Generations: {}", generations);
        println!("Subset size: {} examples per evaluation", subset_size);
        println!("Total parameters: {}", self.parameter_count());

        let start_time = Instant::now();

        // Create fitness function
        let architecture = (self.input_size, self.hidden_size, self.output_size);
        let fitness = GeneticFitness::new(
            architecture,
            Arc::new(x_train.clone()),
            Arc::new(y_train.clone()),
            subset_size,
            100, // Regenerate subset every 100 evaluations
        );

        // Define parameter bounds
        // Using [-3.0, 3.0] range for weights/biases (wider than typical initialization)
        let param_count = self.parameter_count();
        let bounds: Vec<RangeInclusive<f64>> = vec![-3.0..=3.0; param_count];

        // Configure genetic algorithm
        let constants = GlobalConstants::new(
            population_size,
            10, // Number of spatial regions for the algorithm
        );

        // Initialize the world with random organisms
        let mut world = setup_world(&bounds, constants, Box::new(fitness));

        println!("Initial best score: {:.6}", world.get_best_score());

        // Run evolutionary generations
        for generation in 1..=generations {
            // Perform one generation of evolution
            // The library handles: fitness evaluation, selection, reproduction, mutation
            world.training_run(TrainingData::None { floor_value: 0.0 });

            let best_score = world.get_best_score();

            // Print progress every 10 generations
            if generation % 10 == 0 || generation == 1 {
                let elapsed = start_time.elapsed().as_secs_f64();
                println!(
                    "Generation {}/{}: Best Loss = {:.6} (Time: {:.1}s)",
                    generation, generations, best_score, elapsed
                );
            }
        }

        // Extract best organism's parameters and update the network
        let best_organism = world.get_best_organism(TrainingData::None { floor_value: 0.0 });
        let best_params = best_organism.phenotype().expression_problem_values();
        self.unflatten_parameters(best_params);

        let training_time = start_time.elapsed().as_secs_f64();
        let final_loss = world.get_best_score();

        println!("\nGenetic training complete!");
        println!("Final best loss: {:.6}", final_loss);
        println!("Total training time: {:.2}s", training_time);
        println!(
            "Evaluations per second: {:.0}",
            (generations * population_size) as f64 / training_time
        );

        (final_loss, training_time)
    }

    /// Evaluates the network's performance on the full training set after genetic training.
    ///
    /// Since genetic training uses random subsets, this provides an accurate assessment
    /// of how the evolved network performs on all training data.
    ///
    /// # Arguments
    /// * `x_train` - Full training images
    /// * `y_train` - Full training labels
    ///
    /// # Returns
    /// Average loss over the entire training set
    pub fn evaluate_loss(&self, x_train: &Array2<f64>, y_train: &Array2<f64>) -> f64 {
        let mut total_loss = 0.0;

        for i in 0..x_train.nrows() {
            let x = x_train.row(i).to_owned();
            let y = y_train.row(i).to_owned();

            if let Ok((_z1, _a1, _z2, a2)) = self.feed_forward(x) {
                total_loss += self.loss_function(&y, &a2);
            }
        }

        total_loss / x_train.nrows() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn given_small_network_when_train_genetic_then_completes() {
        let mut nn = NeuralNetwork::new(3, 2, 2);

        let x_train = arr2(&[
            [0.5, 0.3, 0.8],
            [0.2, 0.7, 0.4],
            [0.9, 0.1, 0.6],
            [0.4, 0.8, 0.2],
            [0.6, 0.5, 0.7],
            [0.1, 0.9, 0.3],
        ]);
        let y_train = arr2(&[
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]);

        // Run very short training to verify it works
        let (final_loss, training_time) = nn.train_genetic(
            &x_train, &y_train, 5,  // Just 5 generations for testing
            10, // Small population
            3,  // Small subset
        );

        // Verify it ran and produced valid results
        assert!(final_loss > 0.0);
        assert!(final_loss < f64::INFINITY);
        assert!(training_time > 0.0);
    }

    #[test]
    fn given_network_when_evaluate_loss_then_returns_valid_value() {
        let nn = NeuralNetwork::new(3, 2, 2);

        let x_train = arr2(&[[0.5, 0.3, 0.8], [0.2, 0.7, 0.4]]);
        let y_train = arr2(&[[1.0, 0.0], [0.0, 1.0]]);

        let loss = nn.evaluate_loss(&x_train, &y_train);

        assert!(loss > 0.0);
        assert!(loss < f64::INFINITY);
    }

    #[test]
    fn given_trained_network_when_parameters_changed_then_different_from_initial() {
        let mut nn = NeuralNetwork::new(3, 2, 2);

        let original_params = nn.flatten_parameters();

        let x_train = arr2(&[
            [0.5, 0.3, 0.8],
            [0.2, 0.7, 0.4],
            [0.9, 0.1, 0.6],
            [0.4, 0.8, 0.2],
        ]);
        let y_train = arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]);

        nn.train_genetic(&x_train, &y_train, 3, 10, 3);

        let trained_params = nn.flatten_parameters();

        // Parameters should have changed during training
        assert_ne!(original_params, trained_params);
    }
}
