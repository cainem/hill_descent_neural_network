use ndarray::Array2;
use neural_network_scratch::neural_network::NeuralNetwork;

#[cfg(feature = "dhat")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

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
    let n_images = read_u32(&mut img_file) as usize;
    let n_rows = read_u32(&mut img_file) as usize;
    let n_cols = read_u32(&mut img_file) as usize;

    // Load training labels
    let mut lbl_file = File::open("dataset/train-labels.idx1-ubyte").unwrap();
    let _magic = read_u32(&mut lbl_file);
    let n_labels = read_u32(&mut lbl_file) as usize;

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
    #[cfg(feature = "dhat")]
    let _profiler = dhat::Profiler::new_heap();

    println!("=== Memory Profiling: Genetic Training ===\n");
    println!("Loading MNIST subset...");
    let (x_train, y_train) = load_mnist_subset();
    println!("Loaded {} examples\n", x_train.nrows());

    println!("Starting genetic training with memory profiling...");
    println!("Watch for allocation hotspots in dhat output\n");

    let mut nn = NeuralNetwork::new(784, 16, 10);
    let (best_loss, _time) = nn.train_genetic(
        &x_train, &y_train, 10, // Just 10 generations for profiling
        500, 1000, // subset size
    );

    println!("\nTraining complete!");
    println!("Final loss: {:.6}", best_loss);

    #[cfg(feature = "dhat")]
    println!("\nMemory profile saved to dhat-heap.json");
    #[cfg(feature = "dhat")]
    println!("View with: dh_view.html dhat-heap.json");
}
