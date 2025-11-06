// Simple profiling example for genetic training
// Run with: cargo flamegraph --example profile_genetic --release
//
// This runs a single genetic training session that can be profiled
// to generate a flamegraph showing where time is spent.

use ndarray::Array2;
use neural_network_scratch::NeuralNetwork;

fn main() {
    println!("Starting genetic training profiling run...");
    println!("This will run 40 generations with 784-16-10 network\n");

    // Create training data
    let n_examples = 2000;
    let x_train = Array2::from_shape_fn((n_examples, 784), |(i, j)| ((i + j) % 256) as f64 / 255.0);

    let y_train = Array2::from_shape_fn(
        (n_examples, 10),
        |(i, j)| {
            if j == i % 10 {
                1.0
            } else {
                0.0
            }
        },
    );

    // Train with genetic algorithm
    let mut nn = NeuralNetwork::new(784, 16, 10);
    let (final_loss, training_time) = nn.train_genetic(
        &x_train, &y_train, 40,   // generations (enough to profile, not too many)
        500,  // population size
        1000, // subset size
    );

    println!("\nProfiling complete!");
    println!("Final loss: {:.6}", final_loss);
    println!("Training time: {:.2}s", training_time);
    println!("\nFlamegraph should be generated in the project root.");
}
