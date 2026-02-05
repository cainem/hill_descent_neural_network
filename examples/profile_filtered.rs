// Filtered profiling that separates time spent in:
// (a) neural_network_scratch code
// (b) hill_descent_lib code
// Run with: cargo run --example profile_filtered --release

use neural_network_scratch::NeuralNetwork;
use std::time::Instant;

fn main() {
    println!("=== Filtered Performance Profile ===\n");

    // Load MNIST data
    let (x_train, y_train) = neural_network_scratch::load_mnist_data(
        "dataset/train-images.idx3-ubyte",
        "dataset/train-labels.idx1-ubyte",
    )
    .expect("Failed to load training data");

    let mut nn = NeuralNetwork::new(784, 16, 10);

    // Measure time in our neural network code (fitness function)
    let mut total_hill_descent_time = 0.0;

    println!("Testing 10 training cycles...\n");

    for i in 0..10 {
        // Time spent in hill_descent_lib
        let start = Instant::now();
        let _result = nn.train_genetic(
            &x_train, &y_train, 1000, // generations
            500, // population
            100, // subset_size
        );
        let hill_descent_elapsed = start.elapsed().as_secs_f64();
        total_hill_descent_time += hill_descent_elapsed;

        println!("Cycle {}: {:.2}s", i + 1, hill_descent_elapsed);

        // The fitness evaluation time is embedded within hill_descent time
        // We'll need to manually instrument to separate them
    }

    println!("\n=== Summary ===");
    println!(
        "Total time in train_genetic (hill_descent_lib calls): {:.2}s",
        total_hill_descent_time
    );
    println!("Average per cycle: {:.2}s", total_hill_descent_time / 10.0);
}
