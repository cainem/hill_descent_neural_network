// Instrumented profiling for genetic training
// Run with: cargo run --example profile_genetic_instrumented --release
//
// This adds detailed timing instrumentation to identify bottlenecks

use ndarray::Array2;
use neural_network_scratch::NeuralNetwork;
use std::time::Instant;

fn main() {
    println!("=== Instrumented Genetic Training Profile ===\n");
    println!("Network: 784-16-10 (12,730 parameters)");
    println!("Generations: 40");
    println!("Population: 500");
    println!("Subset size: 1000\n");

    // Create training data
    let data_start = Instant::now();
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
    println!(
        "Data creation: {:.3}s\n",
        data_start.elapsed().as_secs_f64()
    );

    // Network initialization
    let init_start = Instant::now();
    let mut nn = NeuralNetwork::new(784, 16, 10);
    println!("Network init: {:.3}s", init_start.elapsed().as_secs_f64());

    // Parameter flattening test
    let flatten_start = Instant::now();
    let params = nn.flatten_parameters();
    let flatten_time = flatten_start.elapsed().as_secs_f64();
    println!(
        "Parameter flattening: {:.6}s ({} params)",
        flatten_time,
        params.len()
    );

    // Parameter unflattening test
    let unflatten_start = Instant::now();
    nn.unflatten_parameters(&params);
    let unflatten_time = unflatten_start.elapsed().as_secs_f64();
    println!("Parameter unflattening: {:.6}s", unflatten_time);

    // Single fitness evaluation test
    let fitness_start = Instant::now();
    let _loss = nn.evaluate_loss(
        &x_train.slice(ndarray::s![0..1000, ..]).to_owned(),
        &y_train.slice(ndarray::s![0..1000, ..]).to_owned(),
    );
    let fitness_time = fitness_start.elapsed().as_secs_f64();
    println!("Single fitness eval (1000 examples): {:.6}s", fitness_time);
    println!(
        "Estimated fitness evals per second: {:.0}\n",
        1.0 / fitness_time
    );

    // Full genetic training
    println!("Starting full genetic training...\n");
    let train_start = Instant::now();
    let (final_loss, training_time) = nn.train_genetic(
        &x_train, &y_train, 40,   // generations
        500,  // population size
        1000, // subset size
    );
    let total_time = train_start.elapsed().as_secs_f64();

    println!("\n=== Performance Analysis ===");
    println!("Total training time: {:.2}s", total_time);
    println!("Final loss: {:.6}", final_loss);
    println!();

    // Calculate derived metrics
    let generations = 40;
    let population = 500;
    let total_evaluations = generations * population;

    println!("Breakdown:");
    println!("  Generations: {}", generations);
    println!("  Population per generation: {}", population);
    println!("  Total fitness evaluations: {}", total_evaluations);
    println!(
        "  Time per generation: {:.3}s",
        training_time / generations as f64
    );
    println!(
        "  Time per fitness eval: {:.6}s",
        training_time / total_evaluations as f64
    );
    println!(
        "  Fitness evals per second: {:.0}",
        total_evaluations as f64 / training_time
    );
    println!();

    // Estimate component costs
    let estimated_fitness_cost = fitness_time * total_evaluations as f64;
    let estimated_flatten_cost = flatten_time * total_evaluations as f64;
    let estimated_unflatten_cost = unflatten_time * total_evaluations as f64;
    let overhead = training_time - estimated_fitness_cost;

    println!("Estimated time breakdown:");
    println!(
        "  Fitness evaluations: {:.2}s ({:.1}%)",
        estimated_fitness_cost,
        estimated_fitness_cost / training_time * 100.0
    );
    println!(
        "  Parameter operations: {:.2}s ({:.1}%)",
        estimated_flatten_cost + estimated_unflatten_cost,
        (estimated_flatten_cost + estimated_unflatten_cost) / training_time * 100.0
    );
    println!(
        "  GA overhead (selection, mutation, etc.): {:.2}s ({:.1}%)",
        overhead.max(0.0),
        (overhead.max(0.0) / training_time * 100.0)
    );
    println!();

    println!("Key findings:");
    if overhead / training_time > 0.5 {
        println!("  ⚠ GA overhead is >50% of total time");
        println!("    - Most time spent in hill_descent_lib operations");
        println!("    - Consider optimizing: selection, crossover, mutation");
    } else if estimated_fitness_cost / training_time > 0.7 {
        println!("  ⚠ Fitness evaluation is >70% of total time");
        println!("    - Most time in forward propagation");
        println!("    - Consider: smaller subset, caching, vectorization");
    }

    if (estimated_flatten_cost + estimated_unflatten_cost) / training_time > 0.1 {
        println!("  ⚠ Parameter flatten/unflatten is >10% of time");
        println!("    - Consider optimizing parameter representation");
    }
}
