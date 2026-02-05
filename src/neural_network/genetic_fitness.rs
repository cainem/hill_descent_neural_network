use hill_descent_lib::SingleValuedFunction;
use ndarray::{Array2, ArrayView1, ArrayView2};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Pre-allocated batch of training data to enable GEMM (matrix-matrix) operations.
#[derive(Debug)]
struct BatchData {
    x: Array2<f64>,
    y: Array2<f64>,
}

/// Thread-local scratchpad for genetic fitness evaluation using GEMM.
struct FastEvaluator {
    /// Buffer for hidden layer activations (BatchSize x HiddenSize)
    a1_buffer: Array2<f64>,
    /// Buffer for output layer activations (BatchSize x OutputSize)
    a2_buffer: Array2<f64>,
}

impl FastEvaluator {
    fn new(batch_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            a1_buffer: Array2::zeros((batch_size, hidden_size)),
            a2_buffer: Array2::zeros((batch_size, output_size)),
        }
    }

    /// Performs zero-allocation batch forward pass and loss calculation using GEMM.
    fn evaluate_batch(
        &mut self,
        x_batch: &Array2<f64>,
        y_true_batch: &Array2<f64>,
        w1: &ArrayView2<f64>,
        b1: &ArrayView1<f64>,
        w2: &ArrayView2<f64>,
        b2: &ArrayView1<f64>,
    ) -> f64 {
        // === LAYER 1 (Hidden) ===
        // Z1 = X * W1 + B1
        // We use beta=0 to clear the buffer first, then add B1 via broadcasting
        ndarray::linalg::general_mat_mul(1.0, x_batch, w1, 0.0, &mut self.a1_buffer);
        self.a1_buffer += b1;
        
        // In-place sigmoid
        self.a1_buffer
            .mapv_inplace(|v| 1.0 / (1.0 + (-v).exp()));

        // === LAYER 2 (Output) ===
        // Z2 = A1 * W2 + B2
        ndarray::linalg::general_mat_mul(1.0, &self.a1_buffer, w2, 0.0, &mut self.a2_buffer);
        self.a2_buffer += b2;
        
        // In-place sigmoid
        self.a2_buffer
            .mapv_inplace(|v| 1.0 / (1.0 + (-v).exp()));

        // === BATCH LOSS CALCULATION ===
        let epsilon = 1e-15;
        let mut total_loss = 0.0;
        let batch_size = self.a2_buffer.nrows();
        let output_size = self.a2_buffer.ncols();

        // Manual loop over the matrix to avoid any intermediate allocations
        for i in 0..batch_size {
            let mut row_loss = 0.0;
            for j in 0..output_size {
                let p = self.a2_buffer[[i, j]].max(epsilon).min(1.0 - epsilon);
                let t = y_true_batch[[i, j]];
                row_loss += t * p.ln() + (1.0 - t) * (1.0 - p).ln();
            }
            total_loss -= row_loss / output_size as f64;
        }

        total_loss / batch_size as f64
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
/// # Performance Optimization: GEMM (Matrix-Matrix Multiplication)
/// Instead of evaluating examples one by one (GEMV), this version processes the
/// entire subset as a single batch matrix multiplication. This significantly
/// improves cache locality and allows the BLAS/math library to use SIMD more effectively.
#[derive(Debug, Clone)]
pub struct GeneticFitness {
    /// Network architecture specification (input_size, hidden_size, output_size)
    architecture: (usize, usize, usize),
    /// Full training data (features)
    x_train: Arc<Array2<f64>>,
    /// Full training labels (one-hot encoded)
    y_train: Arc<Array2<f64>>,
    /// Number of random examples to evaluate per fitness calculation
    subset_size: usize,
    /// Current subset of data, pre-built as contiguous matrices for GEMM
    batch_data: Arc<RwLock<Arc<BatchData>>>,
    /// Counter for when to regenerate subset (every N evaluations)
    eval_counter: Arc<AtomicUsize>,
    /// How often to regenerate the random subset
    regenerate_frequency: usize,
}

impl GeneticFitness {
    /// Creates a new fitness function for genetic algorithm training.
    pub fn new(
        architecture: (usize, usize, usize),
        x_train: Arc<Array2<f64>>,
        y_train: Arc<Array2<f64>>,
        subset_size: usize,
        regenerate_frequency: usize,
    ) -> Self {
        let total_examples = x_train.nrows();
        let indices = Self::generate_random_indices(total_examples, subset_size);
        let batch = Self::build_batch(&x_train, &y_train, &indices);

        GeneticFitness {
            architecture,
            x_train,
            y_train,
            subset_size,
            batch_data: Arc::new(RwLock::new(Arc::new(batch))),
            eval_counter: Arc::new(AtomicUsize::new(0)),
            regenerate_frequency,
        }
    }

    /// Efficiently builds contiguous matrices for the selected subset of indices.
    fn build_batch(x_full: &Array2<f64>, y_full: &Array2<f64>, indices: &[usize]) -> BatchData {
        let subset_size = indices.len();
        let input_size = x_full.ncols();
        let output_size = y_full.ncols();

        let mut x_batch = Array2::zeros((subset_size, input_size));
        let mut y_batch = Array2::zeros((subset_size, output_size));

        for (i, &idx) in indices.iter().enumerate() {
            x_batch.row_mut(i).assign(&x_full.row(idx));
            y_batch.row_mut(i).assign(&y_full.row(idx));
        }

        BatchData { x: x_batch, y: y_batch }
    }

    /// Forces a regeneration of the random training subset and its batch matrices.
    pub fn regenerate_subset(&self) {
        let total_examples = self.x_train.nrows();
        let indices = Self::generate_random_indices(total_examples, self.subset_size);
        let batch = Self::build_batch(&self.x_train, &self.y_train, &indices);
        
        let mut lock = self.batch_data.write().unwrap();
        *lock = Arc::new(batch);
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
        let count = self.eval_counter.fetch_add(1, Ordering::Relaxed);
        if self.regenerate_frequency > 0 && count > 0 && count.is_multiple_of(self.regenerate_frequency) {
            self.regenerate_subset();
        }
    }
}

impl SingleValuedFunction for GeneticFitness {
    /// Evaluates the fitness of a candidate parameter set using batch matrix operations.
    fn single_run(&self, params: &[f64]) -> f64 {
        self.maybe_regenerate_subset();

        let (input_size, hidden_size, output_size) = self.architecture;

        // Map flat params to views
        let w1_len = input_size * hidden_size;
        let b1_len = hidden_size;
        let w2_len = hidden_size * output_size;

        let w1 = ArrayView2::from_shape((input_size, hidden_size), &params[0..w1_len]).unwrap();
        let b1 = ArrayView1::from_shape(b1_len, &params[w1_len..w1_len + b1_len]).unwrap();
        let w2 = ArrayView2::from_shape((hidden_size, output_size), &params[w1_len + b1_len..w1_len + b1_len + w2_len]).unwrap();
        let b2 = ArrayView1::from_shape(output_size, &params[w1_len + b1_len + w2_len..]).unwrap();

        // Get the current batch (read lock is cheap, Arc clone is near-zero cost)
        let batch = {
            let lock = self.batch_data.read().unwrap();
            lock.clone()
        };

        // Use thread-local evaluator for zero-allocation performance
        let mut loss = 0.0;
        EVALUATOR.with(|cell| {
            let mut opt = cell.borrow_mut();
            if opt.is_none() || opt.as_ref().unwrap().a1_buffer.nrows() != batch.x.nrows() {
                *opt = Some(FastEvaluator::new(batch.x.nrows(), hidden_size, output_size));
            }
            let evaluator = opt.as_mut().unwrap();
            loss = evaluator.evaluate_batch(&batch.x, &batch.y, &w1, &b1, &w2, &b2);
        });

        loss
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
    fn given_subset_when_regenerated_then_changes_data() {
        let x_train = Arc::new(arr2(&[[0.5, 0.3], [0.2, 0.7], [0.9, 0.1], [0.4, 0.8]]));
        let y_train = Arc::new(arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]));

        let fitness = GeneticFitness::new((2, 2, 2), x_train, y_train, 2, 0);

        let initial_data = fitness.batch_data.read().unwrap().clone();
        
        // Regenerate multiple times to ensure we likely get a different subset (small dataset)
        let mut changed = false;
        for _ in 0..10 {
            fitness.regenerate_subset();
            let new_data = fitness.batch_data.read().unwrap().clone();
            if !Arc::ptr_eq(&initial_data, &new_data) {
                changed = true;
                break;
            }
        }
        
        assert!(changed, "Batch data should have been replaced");
    }
}
