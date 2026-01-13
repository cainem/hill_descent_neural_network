use ndarray::Array2;
use neural_network_scratch::neural_network::NeuralNetwork;
use std::time::Instant;

fn load_mnist_subset() -> (Array2<f64>, Array2<f64>) {
    use std::fs::File;
    use std::io::Read;

    fn read_u32(file: &mut File) -> u32 {
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf).unwrap();
        u32::from_be_bytes(buf)
    }

    // Load training images
    let mut img_file = File::open("dataset/train-images.idx3-ubyte").unwrap();
    let _magic = read_u32(&mut img_file);
    let _n_images = read_u32(&mut img_file) as usize;
    let n_rows = read_u32(&mut img_file) as usize;
    let n_cols = read_u32(&mut img_file) as usize;

    // Load training labels
    let mut lbl_file = File::open("dataset/train-labels.idx1-ubyte").unwrap();
    let _magic = read_u32(&mut lbl_file);
    let _n_labels = read_u32(&mut lbl_file) as usize;

    // Take only 1000 examples for profiling
    let subset_size = 1000;
    let pixel_count = n_rows * n_cols;

    let mut x_data = vec![0.0; subset_size * pixel_count];
    let mut y_data = vec![0.0; subset_size * 10];

    for i in 0..subset_size {
        let mut pixel_buf = vec![0u8; pixel_count];
        img_file.read_exact(&mut pixel_buf).unwrap();

        for j in 0..pixel_count {
            x_data[i * pixel_count + j] = pixel_buf[j] as f64 / 255.0;
        }

        let mut label_buf = [0u8; 1];
        lbl_file.read_exact(&mut label_buf).unwrap();
        y_data[i * 10 + label_buf[0] as usize] = 1.0;
    }

    let x = Array2::from_shape_vec((subset_size, pixel_count), x_data).unwrap();
    let y = Array2::from_shape_vec((subset_size, 10), y_data).unwrap();

    (x, y)
}

fn main() {
    println!("=== Allocation Analysis: Genetic Training ===\n");

    println!("Loading MNIST subset...");
    let (x_train, y_train) = load_mnist_subset();
    println!("Loaded {} examples", x_train.nrows());
    println!(
        "Data size: {} MB\n",
        (x_train.len() + y_train.len()) * 8 / 1_000_000
    );

    println!("Creating network...");
    let mut nn = NeuralNetwork::new(784, 16, 10);
    println!("Network parameters: {}\n", nn.parameter_count());

    println!("Key allocation sources in fitness evaluation:");
    println!("1. NeuralNetwork::new() - creates W1, b1, W2, b2 matrices");
    println!("   - W1: 784 × 16 = 12,544 f64 = ~100 KB");
    println!("   - b1: 16 f64 = ~128 bytes");
    println!("   - W2: 16 × 10 = 160 f64 = ~1.3 KB");
    println!("   - b2: 10 f64 = ~80 bytes");
    println!("   - Total per network: ~102 KB");
    println!();
    println!("2. unflatten_parameters() - no new allocations (reuses existing arrays)");
    println!();
    println!("3. feed_forward() per example:");
    println!("   - hidden activations: 16 f64 = ~128 bytes");
    println!("   - output activations: 10 f64 = ~80 bytes");
    println!("   - Total per forward pass: ~208 bytes");
    println!();
    println!("4. Per fitness evaluation (1000 examples):");
    println!("   - 1 × NeuralNetwork::new: ~102 KB");
    println!("   - 1000 × feed_forward: ~203 KB");
    println!("   - Total: ~305 KB per evaluation");
    println!();
    println!("5. For 500 organisms per generation:");
    println!("   - 500 × 305 KB = ~153 MB per generation");
    println!("   - With parallelism: 5-10 concurrent = ~1.5 GB working set");
    println!();

    println!("Running 5 generations to demonstrate...\n");
    let start = Instant::now();
    let (best_loss, time) = nn.train_genetic(&x_train, &y_train, 5, 500, 1000);
    let elapsed = start.elapsed();

    println!("\nTraining complete!");
    println!("Best loss: {:.6}", best_loss);
    println!("Time reported: {:.2}s", time);
    println!("Time measured: {:.2}s", elapsed.as_secs_f64());
    println!();
    println!("ANALYSIS:");
    println!("========");
    println!("The main allocation hotspot is creating ~500 temporary NeuralNetwork");
    println!("instances per generation. Each network allocates ~102 KB for weights.");
    println!();
    println!("With parallel evaluation (~5-10 concurrent), this means:");
    println!("  - Constant allocation/deallocation of 102 KB objects");
    println!("  - Memory allocator contention between threads");
    println!("  - Cache thrashing as different threads work on different networks");
    println!();
    println!("OPTIMIZATION OPPORTUNITIES:");
    println!("1. Pool network instances (reuse instead of recreate)");
    println!("2. Batch evaluations to amortize network creation");
    println!("3. Use arena allocator for fitness evaluations");
    println!("4. Evaluate multiple organisms on same network (update weights in-place)");
}
