// Full-Scale MNIST Comparison: Backpropagation vs Genetic Algorithm
//
// This example performs a comprehensive comparison of two training methods
// on the MNIST digit classification task:
// 1. Traditional gradient descent with backpropagation
// 2. Gradient-free genetic algorithm optimization
//
// Run with: cargo run --example mnist_comparison --release
//
// IMPORTANT: Use --release mode! Debug mode is ~10x slower.

use neural_network_scratch::NeuralNetwork;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;
use std::time::Instant;

/// Loads MNIST dataset from IDX file format
fn load_mnist_data() -> Result<
    (
        ndarray::Array2<f64>,
        ndarray::Array2<f64>,
        ndarray::Array2<f64>,
        ndarray::Array2<f64>,
    ),
    Box<dyn std::error::Error>,
> {
    // Load training images
    let train_images_path = PathBuf::from("dataset/train-images.idx3-ubyte");
    let mut file = BufReader::new(File::open(&train_images_path)?);
    let mut buffer = [0u8; 4];

    // Read magic number and dimensions
    file.read_exact(&mut buffer)?;
    file.read_exact(&mut buffer)?;
    let n_images = u32::from_be_bytes(buffer);
    file.read_exact(&mut buffer)?;
    file.read_exact(&mut buffer)?;

    // Read image data and normalize
    let mut image_data = vec![0u8; (n_images * 28 * 28) as usize];
    file.read_exact(&mut image_data)?;
    let x_train: Vec<f64> = image_data.iter().map(|&x| x as f64 / 255.0).collect();
    let x_train = ndarray::Array2::from_shape_vec((n_images as usize, 784), x_train)?;

    // Load training labels
    let train_labels_path = PathBuf::from("dataset/train-labels.idx1-ubyte");
    let mut file = BufReader::new(File::open(&train_labels_path)?);
    file.read_exact(&mut buffer)?;
    file.read_exact(&mut buffer)?;

    let mut label_data = vec![0u8; n_images as usize];
    file.read_exact(&mut label_data)?;

    // Convert to one-hot encoding
    let mut y_train_vec = vec![0.0; (n_images * 10) as usize];
    for (i, &label) in label_data.iter().enumerate() {
        y_train_vec[i * 10 + label as usize] = 1.0;
    }
    let y_train = ndarray::Array2::from_shape_vec((n_images as usize, 10), y_train_vec)?;

    // Load test images
    let test_images_path = PathBuf::from("dataset/t10k-images.idx3-ubyte");
    let mut file = BufReader::new(File::open(&test_images_path)?);
    file.read_exact(&mut buffer)?;
    file.read_exact(&mut buffer)?;
    let n_images = u32::from_be_bytes(buffer);
    file.read_exact(&mut buffer)?;
    file.read_exact(&mut buffer)?;

    let mut image_data = vec![0u8; (n_images * 28 * 28) as usize];
    file.read_exact(&mut image_data)?;
    let x_test: Vec<f64> = image_data.iter().map(|&x| x as f64 / 255.0).collect();
    let x_test = ndarray::Array2::from_shape_vec((n_images as usize, 784), x_test)?;

    // Load test labels
    let test_labels_path = PathBuf::from("dataset/t10k-labels.idx1-ubyte");
    let mut file = BufReader::new(File::open(&test_labels_path)?);
    file.read_exact(&mut buffer)?;
    file.read_exact(&mut buffer)?;

    let mut label_data = vec![0u8; n_images as usize];
    file.read_exact(&mut label_data)?;

    let mut y_test_vec = vec![0.0; (n_images * 10) as usize];
    for (i, &label) in label_data.iter().enumerate() {
        y_test_vec[i * 10 + label as usize] = 1.0;
    }
    let y_test = ndarray::Array2::from_shape_vec((n_images as usize, 10), y_test_vec)?;

    Ok((x_train, y_train, x_test, y_test))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MNIST Training Comparison ===\n");
    println!("Loading MNIST dataset...");

    let (x_train, y_train, x_test, y_test) = load_mnist_data()?;

    println!("Training set: {} images", x_train.nrows());
    println!("Test set: {} images\n", x_test.nrows());

    // Test with smaller network first (784-16-10) as recommended
    let hidden_size = 16;
    println!("Testing with smaller network: 784-{}-10", hidden_size);
    println!(
        "Total parameters: {}\n",
        784 * hidden_size + hidden_size + hidden_size * 10 + 10
    );

    // === METHOD 1: Backpropagation ===
    println!("╔════════════════════════════════════════════╗");
    println!("║  Method 1: Backpropagation (Calculus)     ║");
    println!("╚════════════════════════════════════════════╝\n");

    let mut nn_backprop = NeuralNetwork::new(784, hidden_size, 10);
    let initial_accuracy_backprop = nn_backprop.accuracy(&x_test, &y_test);
    println!("Initial test accuracy: {:.2}%", initial_accuracy_backprop);

    let backprop_start = Instant::now();
    let epochs = 10;
    nn_backprop.train(&x_train, &y_train, epochs, 0.01);
    let backprop_time = backprop_start.elapsed().as_secs_f64();

    let final_accuracy_backprop = nn_backprop.accuracy(&x_test, &y_test);
    let final_loss_backprop = nn_backprop.evaluate_loss(&x_test, &y_test);

    println!("\n--- Backpropagation Results ---");
    println!("Training time: {:.2}s", backprop_time);
    println!("Final test accuracy: {:.2}%", final_accuracy_backprop);
    println!("Final test loss: {:.6}", final_loss_backprop);
    println!(
        "Improvement: +{:.2}%\n",
        final_accuracy_backprop - initial_accuracy_backprop
    );

    // === METHOD 2: Genetic Algorithm ===
    println!("╔════════════════════════════════════════════╗");
    println!("║  Method 2: Genetic Algorithm (Evolution)  ║");
    println!("╚════════════════════════════════════════════╝\n");

    let mut nn_genetic = NeuralNetwork::new(784, hidden_size, 10);
    let initial_accuracy_genetic = nn_genetic.accuracy(&x_test, &y_test);
    println!("Initial test accuracy: {:.2}%", initial_accuracy_genetic);

    // Genetic algorithm parameters
    let generations = 500_000;
    let population_size = 500;
    let subset_size = 1000; // Evaluate on 1000 random training examples per fitness

    println!("\nStarting genetic training...");
    println!("Generations: {}", generations);
    println!("Population: {}", population_size);
    println!("Subset size: {} examples\n", subset_size);

    let (final_loss_genetic, genetic_time) = nn_genetic.train_genetic(
        &x_train,
        &y_train,
        generations,
        population_size,
        subset_size,
    );

    let final_accuracy_genetic = nn_genetic.accuracy(&x_test, &y_test);

    println!("\n--- Genetic Algorithm Results ---");
    println!("Training time: {:.2}s", genetic_time);
    println!("Final test accuracy: {:.2}%", final_accuracy_genetic);
    println!("Final training loss: {:.6}", final_loss_genetic);
    println!(
        "Improvement: +{:.2}%\n",
        final_accuracy_genetic - initial_accuracy_genetic
    );

    // === COMPARISON SUMMARY ===
    println!("╔════════════════════════════════════════════╗");
    println!("║           Comparison Summary               ║");
    println!("╚════════════════════════════════════════════╝\n");

    println!("┌─────────────────────┬──────────────┬──────────────┐");
    println!("│ Metric              │ Backprop     │ Genetic      │");
    println!("├─────────────────────┼──────────────┼──────────────┤");
    println!(
        "│ Training Time       │ {:>8.2}s    │ {:>8.2}s    │",
        backprop_time, genetic_time
    );
    println!(
        "│ Final Accuracy      │ {:>8.2}%    │ {:>8.2}%    │",
        final_accuracy_backprop, final_accuracy_genetic
    );
    println!(
        "│ Accuracy Gain       │ {:>+8.2}%    │ {:>+8.2}%    │",
        final_accuracy_backprop - initial_accuracy_backprop,
        final_accuracy_genetic - initial_accuracy_genetic
    );
    println!(
        "│ Time Ratio          │ {:>8.1}x    │ {:>8.1}x    │",
        1.0,
        genetic_time / backprop_time
    );
    println!("└─────────────────────┴──────────────┴──────────────┘\n");

    println!("Key Findings:");
    println!("• Backpropagation:");
    println!("  - Uses calculus (chain rule) for precise gradient computation");
    println!("  - Fast and efficient for standard network architectures");
    println!("  - Requires differentiable activation functions");
    println!("  - Expected accuracy: ~96-97% on MNIST\n");

    println!("• Genetic Algorithm:");
    println!("  - Gradient-free evolutionary optimization");
    println!("  - Can handle non-differentiable network structures");
    println!(
        "  - {:.0}x slower due to population-based search",
        genetic_time / backprop_time
    );
    println!("  - Explores parameter space more broadly\n");

    println!("Trade-off Analysis:");
    if final_accuracy_genetic > final_accuracy_backprop {
        println!(
            "✓ Genetic algorithm achieved HIGHER accuracy (+{:.2}%)",
            final_accuracy_genetic - final_accuracy_backprop
        );
        println!("  This suggests the broader exploration found a better solution.");
    } else {
        println!(
            "✗ Backpropagation achieved HIGHER accuracy (+{:.2}%)",
            final_accuracy_backprop - final_accuracy_genetic
        );
        println!("  Gradient-based optimization was more effective for this problem.");
    }

    println!("\nConclusion:");
    println!("For standard feedforward networks on MNIST:");
    println!("• Backpropagation is typically more efficient");
    println!("• Genetic algorithms offer architectural flexibility");
    println!(
        "• The {:.0}x time cost may be worthwhile for:",
        genetic_time / backprop_time
    );
    println!("  - Non-differentiable network components");
    println!("  - Novel architecture exploration");
    println!("  - Avoiding gradient-related issues (vanishing/exploding)");

    Ok(())
}
