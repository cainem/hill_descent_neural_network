use super::NeuralNetwork;
use ndarray::Array2;

impl NeuralNetwork {
    /// Trains the neural network on a dataset for multiple epochs.
    ///
    /// Training is the process of adjusting the network's weights and biases to minimize
    /// prediction errors. This function implements stochastic gradient descent: it processes
    /// one example at a time, computing the error and updating weights after each example.
    ///
    /// The training process:
    /// 1. For each epoch (complete pass through training data):
    ///    - For each training example:
    ///      a. Perform forward propagation to get predictions
    ///      b. Calculate loss (how wrong the prediction is)
    ///      c. Perform backpropagation to compute gradients
    ///      d. Update weights and biases using gradient descent
    ///    - Print average loss for the epoch
    ///
    /// # Arguments
    /// * `x_train` - Training images as 2D array (rows=examples, cols=784 pixels)
    /// * `y_train` - Training labels as 2D array (rows=examples, cols=10 one-hot encoded)
    /// * `epochs` - Number of complete passes through the training dataset
    /// * `learning_rate` - Step size for weight updates (typically 0.01)
    ///
    /// # Example
    /// ```
    /// nn.train(&x_train, &y_train, 10, 0.01);  // Train for 10 epochs
    /// ```
    ///
    /// # Performance Notes
    /// - Training is slow in debug mode; use `cargo run --release` for ~10x speedup
    /// - More epochs generally improve accuracy but take longer and may overfit
    /// - Loss should generally decrease over epochs (though may fluctuate)
    pub fn train(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &Array2<f64>,
        epochs: usize,
        learning_rate: f64,
    ) {
        // Iterate through the complete dataset multiple times
        // Each epoch gives the network another chance to learn from all examples
        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            // Process each training example individually (stochastic gradient descent)
            // Alternative approaches: batch gradient descent (all at once) or mini-batch
            for i in 0..x_train.nrows() {
                // Extract the i-th training example and its label
                let x = x_train.row(i).to_owned();
                let y = y_train.row(i).to_owned();

                // === FORWARD PASS ===
                // Compute the network's prediction and all intermediate values
                let (z1, a1, z2, a2) = match self.feed_forward(x.clone()) {
                    Ok(data) => data,
                    Err(e) => {
                        // If feed_forward fails (shouldn't happen with valid data),
                        // print error and skip this example rather than crashing
                        eprintln!("Feed forward error: {}", e);
                        continue;
                    }
                };

                // Calculate how wrong the prediction was and accumulate for epoch average
                total_loss += self.loss_function(&y, &a2);

                // === BACKWARD PASS ===
                // Compute gradients and update weights to reduce the error
                self.back_propagation(&x, &y, &z1, &a1, &z2, &a2, learning_rate);
            }

            // Calculate and display the average loss for this epoch
            // This helps monitor training progress - loss should generally decrease
            total_loss /= x_train.nrows() as f64;
            println!("Epoch {}: Loss = {}", epoch + 1, total_loss);
        }
    }
}
