use hill_descent_lib::SingleValuedFunction;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Thread-local scratchpad for genetic fitness evaluation to avoid allocations.
struct FastEvaluator {
    a1_buffer: Array1<f64>,
    a2_buffer: Array1<f64>,
}

impl FastEvaluator {
    fn new(hidden_size: usize, output_size: usize) -> Self {
        Self {
            a1_buffer: Array1::zeros(hidden_size),
            a2_buffer: Array1::zeros(output_size),
        }
    }

    /// Performs zero-allocation forward pass and loss calculation.
    fn evaluate(
        &mut self,
        x: ArrayView1<f64>,
        y_true: ArrayView1<f64>,
        w1: &ArrayView2<f64>,
        b1: &ArrayView1<f64>,
        w2: &ArrayView2<f64>,
        b2: &ArrayView1<f64>,
    ) -> f64 {
        // === LAYER 1 (Hidden) ===
        // z1 = x * W1 + b1
        self.a1_buffer.assign(b1);
        // Using general_mat_vec_mul for in-place matrix-vector product
        // Note: x (1xN) * W1 (NxM) is same as W1.t() (MxN) * x (Nx1)
        ndarray::linalg::general_mat_vec_mul(1.0, &w1.t(), &x, 1.0, &mut self.a1_buffer);
        // In-place sigmoid
        self.a1_buffer
            .mapv_inplace(|v| 1.0 / (1.0 + (-v).exp()));

        // === LAYER 2 (Output) ===
        // z2 = a1 * W2 + b2
        self.a2_buffer.assign(b2);
        ndarray::linalg::general_mat_vec_mul(
            1.0,
            &w2.t(),
            &self.a1_buffer.view(),
            1.0,
            &mut self.a2_buffer,
        );
        // In-place sigmoid
        self.a2_buffer
            .mapv_inplace(|v| 1.0 / (1.0 + (-v).exp()));

        // === LOSS CALCULATION ===
        // Manual loop to avoid allocations in loss_function
        let epsilon = 1e-15;
        let mut sum_loss = 0.0;
        for i in 0..self.a2_buffer.len() {
            let p = self.a2_buffer[i].max(epsilon).min(1.0 - epsilon);
            let t = y_true[i];
            // Binary cross-entropy: -[t*ln(p) + (1-t)*ln(1-p)]
            sum_loss += t * p.ln() + (1.0 - t) * (1.0 - p).ln();
        }

        -sum_loss / self.a2_buffer.len() as f64
    }
}

thread_local! {
    /// Each thread keeps its own buffers to avoid reallocation and synchronization.
    static EVALUATOR: RefCell<Option<FastEvaluator>> = const { RefCell::new(None) };
}

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
    /// Thread-safe random subset indices - uses Arc<RwLock> for safe updates across threads
    subset_indices: Arc<RwLock<Vec<usize>>>,
    /// Counter for when to regenerate subset (every N evaluations) - uses atomic for lock-free access
    eval_counter: Arc<AtomicUsize>,
    /// How often to regenerate the random subset
    regenerate_frequency: usize,
}

impl GeneticFitness {
    /// Creates a new fitness function for genetic algorithm training.
    ///
    /// Internal use only - called by `NeuralNetwork::train_genetic()`.
    ///
    /// # Arguments
    /// * `architecture` - Network dimensions (input_size, hidden_size, output_size)
    /// * `x_train` - Training images as 2D array (rows=examples, cols=784 pixels)
    /// * `y_train` - Training labels as 2D array (rows=examples, cols=10 one-hot)
    /// * `subset_size` - Number of random examples to evaluate per fitness call
    /// * `regenerate_frequency` - How many evaluations before picking new random subset
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
            subset_indices: Arc::new(RwLock::new(initial_subset)),
            eval_counter: Arc::new(AtomicUsize::new(0)),
            regenerate_frequency,
        }
    }

    /// Returns a shared handle to the subset indices.
    ///
    /// This allows external code to trigger subset rotation.
    pub fn subset_handle(&self) -> Arc<RwLock<Vec<usize>>> {
        self.subset_indices.clone()
    }

    /// Forces a regeneration of the random training subset.
    ///
    /// The next evaluations will use a different set of 1000 examples.
    pub fn regenerate_subset(&self) {
        let mut indices = self.subset_indices.write().unwrap();
        *indices = Self::generate_random_indices(self.x_train.nrows(), self.subset_size);
    }

    /// Generates a random subset of indices without replacement.
    fn generate_random_indices(total: usize, count: usize) -> Vec<usize> {
        let mut rng = rand::rngs::StdRng::from_os_rng();
        let mut indices: Vec<usize> = (0..total).collect();
        indices.shuffle(&mut rng);
        indices.truncate(count);
        indices
    }

    /// Checks if it's time to regenerate the random subset and does so if needed.
    fn maybe_regenerate_subset(&self) {
        // Increment counter
        let count = self.eval_counter.fetch_add(1, Ordering::Relaxed);

        // Regenerate subset if frequency is reached
        if self.regenerate_frequency > 0 && count > 0 && count.is_multiple_of(self.regenerate_frequency) {
            self.regenerate_subset();
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
    fn single_run(&self, params: &[f64]) -> f64 {
        // Regenerate subset periodically to avoid overfitting to specific examples
        self.maybe_regenerate_subset();

        let (input_size, hidden_size, output_size) = self.architecture;

        // Slice parameters into views to avoid any allocations or copies
        let w1_len = input_size * hidden_size;
        let b1_len = hidden_size;
        let w2_len = hidden_size * output_size;
        // let b2_len = output_size; // b2 starts after w2

        let w1_slice = &params[0..w1_len];
        let b1_slice = &params[w1_len..w1_len + b1_len];
        let w2_slice = &params[w1_len + b1_len..w1_len + b1_len + w2_len];
        let b2_slice = &params[w1_len + b1_len + w2_len..];

        // Map slices to views
        let w1 = ArrayView2::from_shape((input_size, hidden_size), w1_slice).unwrap();
        let b1 = ArrayView1::from_shape(b1_len, b1_slice).unwrap();
        let w2 = ArrayView2::from_shape((hidden_size, output_size), w2_slice).unwrap();
        let b2 = ArrayView1::from_shape(output_size, b2_slice).unwrap();

        // Get current subset indices
        let indices_lock = self.subset_indices.read().unwrap();
        let indices = &*indices_lock;

        // Use thread-local evaluator for zero-allocation performance
        let mut total_loss = 0.0;
        EVALUATOR.with(|cell| {
            let mut opt = cell.borrow_mut();
            // Initialize or resize evaluator if needed
            if opt.is_none() || opt.as_ref().unwrap().a1_buffer.len() != hidden_size {
                *opt = Some(FastEvaluator::new(hidden_size, output_size));
            }
            let evaluator = opt.as_mut().unwrap();

            for &idx in indices {
                let x = self.x_train.row(idx);
                let y = self.y_train.row(idx);
                total_loss += evaluator.evaluate(x, y, &w1, &b1, &w2, &b2);
            }
        });

        // Return average loss
        total_loss / indices.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use crate::NeuralNetwork;

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
    fn given_subset_when_used_multiple_times_then_remains_stable() {
        let x_train = Arc::new(arr2(&[[0.5, 0.3], [0.2, 0.7], [0.9, 0.1], [0.4, 0.8]]));
        let y_train = Arc::new(arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]));

        let fitness = GeneticFitness::new((2, 2, 2), x_train, y_train, 2, 5);

        let nn = NeuralNetwork::new(2, 2, 2);
        let params = nn.flatten_parameters();

        // Get initial indices
        let initial_indices = fitness.subset_handle().read().unwrap().clone();

        // Call single_run multiple times
        for _ in 0..3 {
            fitness.single_run(&params);
        }

        // Indices should remain the same (frequency is 5, we did 3 + 1 initial = 4 calls)
        let final_indices = fitness.subset_handle().read().unwrap().clone();
        assert_eq!(initial_indices, final_indices);
    }
}
