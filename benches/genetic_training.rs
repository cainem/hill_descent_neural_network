use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use neural_network_scratch::NeuralNetwork;

/// Creates a small MNIST-like dataset for benchmarking
fn create_benchmark_data() -> (Array2<f64>, Array2<f64>) {
    // Create 2000 training examples (enough to be representative)
    let n_examples = 2000;
    let x_train = Array2::from_shape_fn((n_examples, 784), |(i, j)| {
        // Simple pattern based on indices
        ((i + j) % 256) as f64 / 255.0
    });

    let y_train = Array2::from_shape_fn((n_examples, 10), |(i, j)| {
        // One-hot encoding
        if j == i % 10 {
            1.0
        } else {
            0.0
        }
    });

    (x_train, y_train)
}

fn benchmark_genetic_training_1000_gen(c: &mut Criterion) {
    let (x_train, y_train) = create_benchmark_data();

    c.bench_function("genetic_training_784_16_10_40gen", |b| {
        b.iter(|| {
            let mut nn = NeuralNetwork::new(784, 16, 10);
            let (final_loss, _training_time) = nn.train_genetic(
                black_box(&x_train),
                black_box(&y_train),
                40,   // generations
                500,  // population size
                1000, // subset size
            );
            black_box(final_loss)
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10); // Only 10 samples since this is slow
    targets = benchmark_genetic_training_1000_gen
}

criterion_main!(benches);
