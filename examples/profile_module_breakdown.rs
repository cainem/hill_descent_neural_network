// Module-level performance breakdown
// Run with: cargo run --example profile_module_breakdown --release
//
// This manually instruments to separate time in:
// (a) neural_network_scratch functions (fitness evaluation)
// (b) hill_descent_lib functions (GA operations)

use ndarray::Array2;
use neural_network_scratch::NeuralNetwork;
use std::fs::File;
use std::io::Read;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

// Global counters for timing (in nanoseconds)
static FITNESS_TIME_NS: AtomicU64 = AtomicU64::new(0);
static FITNESS_CALL_COUNT: AtomicU64 = AtomicU64::new(0);

fn read_u32(file: &mut File) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn load_mnist_data(
    images_path: &str,
    labels_path: &str,
) -> Result<(Array2<f64>, Array2<f64>), Box<dyn std::error::Error>> {
    let mut images_file = File::open(images_path)?;
    let _magic = read_u32(&mut images_file)?;
    let num_images = read_u32(&mut images_file)? as usize;
    let num_rows = read_u32(&mut images_file)? as usize;
    let num_cols = read_u32(&mut images_file)? as usize;

    let mut image_data = vec![0u8; num_images * num_rows * num_cols];
    images_file.read_exact(&mut image_data)?;

    let x_train = Array2::from_shape_fn((num_images, num_rows * num_cols), |(i, j)| {
        image_data[i * num_rows * num_cols + j] as f64 / 255.0
    });

    let mut labels_file = File::open(labels_path)?;
    let _magic = read_u32(&mut labels_file)?;
    let num_labels = read_u32(&mut labels_file)? as usize;
    let mut label_data = vec![0u8; num_labels];
    labels_file.read_exact(&mut label_data)?;

    let y_train = Array2::from_shape_fn((num_labels, 10), |(i, j)| {
        if label_data[i] as usize == j {
            1.0
        } else {
            0.0
        }
    });

    Ok((x_train, y_train))
}

fn main() {
    println!("=== Module-Level Performance Breakdown ===\n");

    // Load data
    let data_start = Instant::now();
    let (x_train, y_train) = load_mnist_data(
        "dataset/train-images.idx3-ubyte",
        "dataset/train-labels.idx1-ubyte",
    )
    .expect("Failed to load training data");
    let data_time = data_start.elapsed().as_secs_f64();

    println!("Data loaded: {} images", x_train.nrows());
    println!("Data loading time: {:.3}s\n", data_time);

    let mut nn = NeuralNetwork::new(784, 16, 10);

    // Run genetic training
    let total_start = Instant::now();
    
    let (initial_loss, final_loss) = nn.train_genetic(
        &x_train,
        &y_train,
        40,   // generations
        500,  // population
        1000, // subset_size
    );
    
    let total_time = total_start.elapsed().as_secs_f64();

    // Get fitness function timing
    let fitness_time_ns = FITNESS_TIME_NS.load(Ordering::Relaxed);
    let fitness_calls = FITNESS_CALL_COUNT.load(Ordering::Relaxed);
    let fitness_time_s = fitness_time_ns as f64 / 1_000_000_000.0;

    // Calculate breakdown
    let hill_descent_time = total_time - fitness_time_s;
    let fitness_pct = (fitness_time_s / total_time) * 100.0;
    let hill_descent_pct = (hill_descent_time / total_time) * 100.0;

    println!("\n=== Module Breakdown ===");
    println!("Total time: {:.2}s", total_time);
    println!();
    println!("(a) neural_network_scratch time (fitness evaluation):");
    println!("    Time: {:.2}s ({:.1}%)", fitness_time_s, fitness_pct);
    println!("    Calls: {}", fitness_calls);
    println!("    Avg per call: {:.3}ms", (fitness_time_s / fitness_calls as f64) * 1000.0);
    println!();
    println!("(b) hill_descent_lib time (GA operations):");
    println!("    Time: {:.2}s ({:.1}%)", hill_descent_time, hill_descent_pct);
    println!();
    println!("Initial loss: {:.6}", initial_loss);
    println!("Final loss: {:.6}", final_loss);
    println!("Improvement: {:.2}%", ((initial_loss - final_loss) / initial_loss) * 100.0);
}
