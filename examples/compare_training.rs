// Example: Comparing Backpropagation vs Genetic Algorithm Training
//
// This example demonstrates side-by-side training of two identical neural networks:
// 1. Traditional gradient descent with backpropagation
// 2. Gradient-free genetic algorithm optimization
//
// Run with: cargo run --example compare_training --release
//
// Note: Use --release mode for reasonable performance, especially for genetic training

use ndarray::arr2;
use neural_network_scratch::NeuralNetwork;

fn main() {
    println!("=== Neural Network Training Comparison ===\n");
    println!("Comparing backpropagation vs genetic algorithm");
    println!("Network architecture: 4-3-2 (tiny for demonstration)\n");

    // Create simple training data (XOR-like problem)
    let x_train = arr2(&[
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5],
        [0.2, 0.8, 0.3, 0.7],
    ]);

    let y_train = arr2(&[
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.5, 0.5],
        [0.3, 0.7],
    ]);

    let x_test = arr2(&[[0.1, 0.9, 0.2, 0.8], [0.9, 0.1, 0.8, 0.2]]);

    let y_test = arr2(&[[0.0, 1.0], [1.0, 0.0]]);

    // === METHOD 1: Traditional Backpropagation ===
    println!("--- Method 1: Backpropagation (Gradient Descent) ---");
    let mut nn_backprop = NeuralNetwork::new(4, 3, 2);

    let initial_loss_backprop = nn_backprop.evaluate_loss(&x_train, &y_train);
    println!("Initial loss: {:.6}", initial_loss_backprop);

    nn_backprop.train(&x_train, &y_train, 50, 0.1);

    let final_loss_backprop = nn_backprop.evaluate_loss(&x_train, &y_train);
    let test_accuracy_backprop = nn_backprop.accuracy(&x_test, &y_test);

    println!("Final loss: {:.6}", final_loss_backprop);
    println!("Test accuracy: {:.2}%", test_accuracy_backprop);
    println!(
        "Loss reduction: {:.1}%\n",
        ((initial_loss_backprop - final_loss_backprop) / initial_loss_backprop * 100.0)
    );

    // === METHOD 2: Genetic Algorithm ===
    println!("--- Method 2: Genetic Algorithm (Gradient-Free) ---");
    let mut nn_genetic = NeuralNetwork::new(4, 3, 2);

    let initial_loss_genetic = nn_genetic.evaluate_loss(&x_train, &y_train);
    println!("Initial loss: {:.6}", initial_loss_genetic);

    let (final_loss_genetic, training_time) = nn_genetic.train_genetic(
        &x_train, &y_train, 50, // generations (comparable to epochs)
        50, // population size
        6,  // evaluate on all 6 training examples
    );

    let test_accuracy_genetic = nn_genetic.accuracy(&x_test, &y_test);

    println!("\nFinal loss: {:.6}", final_loss_genetic);
    println!("Test accuracy: {:.2}%", test_accuracy_genetic);
    println!(
        "Loss reduction: {:.1}%",
        ((initial_loss_genetic - final_loss_genetic) / initial_loss_genetic * 100.0)
    );
    println!("Training time: {:.2}s", training_time);

    // === COMPARISON SUMMARY ===
    println!("\n=== Summary ===");
    println!("Backpropagation:");
    println!("  Final loss: {:.6}", final_loss_backprop);
    println!("  Test accuracy: {:.2}%", test_accuracy_backprop);

    println!("\nGenetic Algorithm:");
    println!("  Final loss: {:.6}", final_loss_genetic);
    println!("  Test accuracy: {:.2}%", test_accuracy_genetic);
    println!("  Training time: {:.2}s", training_time);

    println!("\nKey Insights:");
    println!("- Backpropagation uses calculus (chain rule) - fast and precise");
    println!("- Genetic algorithm is gradient-free - slower but more flexible");
    println!("- Genetic approach could handle non-differentiable network structures");
    println!("- Trade-off: architectural freedom vs computational cost");
}
