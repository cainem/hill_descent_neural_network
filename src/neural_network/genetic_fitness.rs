use super::NeuralNetwork;
use hill_descent_lib::SingleValuedFunction;
use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::sync::{Arc, Mutex};

/// Fitness function for genetic algorithm training of neural networks.
///
/// Implements the `SingleValuedFunction` trait from hill_descent_lib to evaluate
/// candidate network parameter sets. Lower fitness values are better (minimization).
///
/// The fitness function:
/// 1. Takes a flat parameter vector from the genetic algorithm
/// 2. Reconstructs a neural network with those parameters
/// 3. Evaluates the network on a random subset of training data
/// 4. Returns the average loss (binary cross-entropy)
///
/// # Performance Optimization
/// Uses a random subset of training examples (default 1000) rather than the full
/// training set to make fitness evaluation tractable. This is necessary because
/// genetic algorithms require many fitness evaluations per generation.
#[derive(Debug)]
pub struct GeneticFitness {
    /// Network architecture specification (input_size, hidden_size, output_size)
    architecture: (usize, usize, usize),
    /// Full training data (features)
    x_train: Arc<Array2<f64>>,
    /// Full training labels (one-hot encoded)
    y_train: Arc<Array2<f64>>,
    /// Number of random examples to evaluate per fitness calculation
    subset_size: usize,
    /// Thread-safe random subset indices (regenerated periodically)
    subset_indices: Arc<Mutex<Vec<usize>>>,
    /// Counter for when to regenerate subset (every N evaluations)
    eval_counter: Arc<Mutex<usize>>,
    /// How often to regenerate the random subset
    regenerate_frequency: usize,
}

impl GeneticFitness {
    /// Creates a new fitness function for genetic algorithm training.
    ///
    /// # Arguments
    /// * `architecture` - Network dimensions (input_size, hidden_size, output_size)
    /// * `x_train` - Training images as 2D array (rows=examples, cols=784 pixels)
    /// * `y_train` - Training labels as 2D array (rows=examples, cols=10 one-hot)
    /// * `subset_size` - Number of random examples to evaluate per fitness call
    /// * `regenerate_frequency` - How many evaluations before picking new random subset
    ///
    /// # Example
    /// ```
    /// let fitness = GeneticFitness::new(
    ///     (784, 64, 10),
    ///     Arc::new(x_train),
    ///     Arc::new(y_train),
    ///     1000,  // Evaluate on 1000 random examples
    ///     100,   // Pick new examples every 100 evaluations
    /// );
    /// ```
    pub fn new(
        architecture: (usize, usize, usize),
        x_train: Arc<Array2<f64>>,
        y_train: Arc<Array2<f64>>,
        subset_size: usize,
        regenerate_frequency: usize,
    ) -> Self {
        let total_examples = x_train.nrows();
        let initial_subset = Self::generate_random_indices(total_examples, subset_size);

        GeneticFitness {
            architecture,
            x_train,
            y_train,
            subset_size,
            subset_indices: Arc::new(Mutex::new(initial_subset)),
            eval_counter: Arc::new(Mutex::new(0)),
            regenerate_frequency,
        }
    }

    /// Generates a random subset of indices without replacement.
    fn generate_random_indices(total: usize, count: usize) -> Vec<usize> {
        let mut rng = rand::rngs::StdRng::from_entropy();
        let mut indices: Vec<usize> = (0..total).collect();
        indices.shuffle(&mut rng);
        indices.truncate(count);
        indices
    }

    /// Checks if it's time to regenerate the random subset and does so if needed.
    fn maybe_regenerate_subset(&self) {
        let mut counter = self.eval_counter.lock().unwrap();
        *counter += 1;

        if (*counter).is_multiple_of(self.regenerate_frequency) {
            let total_examples = self.x_train.nrows();
            let new_indices = Self::generate_random_indices(total_examples, self.subset_size);
            let mut indices = self.subset_indices.lock().unwrap();
            *indices = new_indices;
        }
    }
}

impl SingleValuedFunction for GeneticFitness {
    /// Evaluates the fitness of a candidate parameter set.
    ///
    /// This is called by the genetic algorithm for each organism in each generation.
    /// The function must be thread-safe as hill_descent_lib may parallelize evaluations.
    ///
    /// # Arguments
    /// * `params` - Flat vector of network parameters from genetic algorithm
    ///
    /// # Returns
    /// Average loss (lower is better). Binary cross-entropy over the random subset.
    ///
    /// # Implementation Notes
    /// - Creates a temporary network for evaluation (not modifying any shared state)
    /// - Evaluates only on a random subset for performance
    /// - Returns infinity if network construction or evaluation fails
    fn single_run(&self, params: &[f64]) -> f64 {
        // Regenerate subset periodically to avoid overfitting to specific examples
        self.maybe_regenerate_subset();

        // Create a temporary network with the candidate parameters
        let (input_size, hidden_size, output_size) = self.architecture;
        let mut nn = NeuralNetwork::new(input_size, hidden_size, output_size);
        nn.unflatten_parameters(params);

        // Get current subset indices
        let indices = self.subset_indices.lock().unwrap().clone();

        // Evaluate loss on the random subset
        let mut total_loss = 0.0;
        for &idx in &indices {
            let x = self.x_train.row(idx).to_owned();
            let y = self.y_train.row(idx).to_owned();

            // Forward pass to get prediction
            match nn.feed_forward(x) {
                Ok((_z1, _a1, _z2, a2)) => {
                    total_loss += nn.loss_function(&y, &a2);
                }
                Err(_) => {
                    // If forward pass fails, return high penalty
                    return f64::INFINITY;
                }
            }
        }

        // Return average loss
        total_loss / indices.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn given_fitness_function_when_single_run_then_returns_valid_loss() {
        let x_train = Arc::new(arr2(&[
            [0.5, 0.3, 0.8],
            [0.2, 0.7, 0.4],
            [0.9, 0.1, 0.6],
            [0.4, 0.8, 0.2],
        ]));
        let y_train = Arc::new(arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]));

        let fitness = GeneticFitness::new((3, 2, 2), x_train, y_train, 2, 10);

        let nn = NeuralNetwork::new(3, 2, 2);
        let params = nn.flatten_parameters();

        let loss = fitness.single_run(&params);

        // Loss should be a valid positive number
        assert!(loss > 0.0);
        assert!(loss < f64::INFINITY);
    }

    #[test]
    fn given_random_indices_when_generate_then_correct_count() {
        let indices = GeneticFitness::generate_random_indices(1000, 100);
        assert_eq!(indices.len(), 100);
    }

    #[test]
    fn given_random_indices_when_generate_then_all_unique() {
        let indices = GeneticFitness::generate_random_indices(1000, 100);
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), indices.len());
    }

    #[test]
    fn given_random_indices_when_generate_then_in_valid_range() {
        let indices = GeneticFitness::generate_random_indices(1000, 100);
        assert!(indices.iter().all(|&i| i < 1000));
    }

    #[test]
    fn given_subset_regenerate_when_called_multiple_times_then_eventually_changes() {
        let x_train = Arc::new(arr2(&[[0.5, 0.3], [0.2, 0.7], [0.9, 0.1], [0.4, 0.8]]));
        let y_train = Arc::new(arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]));

        let fitness = GeneticFitness::new((2, 2, 2), x_train, y_train, 2, 5);

        let nn = NeuralNetwork::new(2, 2, 2);
        let params = nn.flatten_parameters();

        // Get initial indices
        let initial_indices = fitness.subset_indices.lock().unwrap().clone();

        // Call single_run multiple times to trigger regeneration
        for _ in 0..10 {
            fitness.single_run(&params);
        }

        // Indices should have changed after regenerate_frequency calls
        let new_indices = fitness.subset_indices.lock().unwrap().clone();
        assert_ne!(initial_indices, new_indices);
    }
}
