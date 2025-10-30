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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn given_zero_epochs_when_train_then_no_changes() {
        let mut nn = NeuralNetwork::new(3, 2, 2);

        let original_w1 = nn.w1().clone();
        let original_w2 = nn.w2().clone();

        let x_train = arr2(&[[0.5, 0.3, 0.8], [0.2, 0.7, 0.4]]);
        let y_train = arr2(&[[1.0, 0.0], [0.0, 1.0]]);

        nn.train(&x_train, &y_train, 0, 0.01);

        // With zero epochs, nothing should change
        assert_eq!(nn.w1(), &original_w1);
        assert_eq!(nn.w2(), &original_w2);
    }

    #[test]
    fn given_training_data_when_train_one_epoch_then_weights_change() {
        let mut nn = NeuralNetwork::new(3, 2, 2);

        let original_w1 = nn.w1().clone();
        let original_w2 = nn.w2().clone();

        let x_train = arr2(&[[0.5, 0.3, 0.8], [0.2, 0.7, 0.4]]);
        let y_train = arr2(&[[1.0, 0.0], [0.0, 1.0]]);

        nn.train(&x_train, &y_train, 1, 0.01);

        // After one epoch, weights should have changed
        let mut w1_changed = false;
        let mut w2_changed = false;

        for i in 0..nn.w1().len() {
            if (nn.w1()[[i / 2, i % 2]] - original_w1[[i / 2, i % 2]]).abs() > 1e-10 {
                w1_changed = true;
                break;
            }
        }

        for i in 0..nn.w2().len() {
            if (nn.w2()[[i / 2, i % 2]] - original_w2[[i / 2, i % 2]]).abs() > 1e-10 {
                w2_changed = true;
                break;
            }
        }

        assert!(w1_changed, "W1 should change after training");
        assert!(w2_changed, "W2 should change after training");
    }

    #[test]
    fn given_single_example_when_train_then_completes_successfully() {
        let mut nn = NeuralNetwork::new(3, 2, 2);

        let x_train = arr2(&[[0.5, 0.3, 0.8]]);
        let y_train = arr2(&[[1.0, 0.0]]);

        // Should not panic with single example
        nn.train(&x_train, &y_train, 5, 0.01);
    }

    #[test]
    fn given_separable_data_when_train_many_epochs_then_loss_decreases() {
        let mut nn = NeuralNetwork::new(2, 4, 2);

        // Create linearly separable data
        let x_train = arr2(&[
            [0.0, 0.0],
            [0.0, 0.1],
            [0.1, 0.0],
            [0.9, 0.9],
            [1.0, 0.9],
            [0.9, 1.0],
        ]);
        let y_train = arr2(&[
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]);

        // Compute initial loss
        let mut total_loss = 0.0;
        for i in 0..x_train.nrows() {
            let x = x_train.row(i).to_owned();
            let y = y_train.row(i).to_owned();
            let (_, _, _, a2) = nn.feed_forward(x).unwrap();
            total_loss += nn.loss_function(&y, &a2);
        }
        let initial_loss = total_loss / x_train.nrows() as f64;

        // Train for many epochs
        nn.train(&x_train, &y_train, 100, 0.1);

        // Compute final loss
        let mut total_loss = 0.0;
        for i in 0..x_train.nrows() {
            let x = x_train.row(i).to_owned();
            let y = y_train.row(i).to_owned();
            let (_, _, _, a2) = nn.feed_forward(x).unwrap();
            total_loss += nn.loss_function(&y, &a2);
        }
        let final_loss = total_loss / x_train.nrows() as f64;

        // Loss should have decreased significantly
        assert!(
            final_loss < initial_loss * 0.7,
            "Final loss ({}) should be less than 70% of initial loss ({})",
            final_loss,
            initial_loss
        );
    }

    #[test]
    fn given_multiple_epochs_when_train_then_processes_all_epochs() {
        let mut nn = NeuralNetwork::new(3, 2, 2);

        let x_train = arr2(&[[0.5, 0.3, 0.8], [0.2, 0.7, 0.4]]);
        let y_train = arr2(&[[1.0, 0.0], [0.0, 1.0]]);

        // This test mainly ensures train completes without panicking
        // and processes multiple epochs (indirectly tested by not crashing)
        nn.train(&x_train, &y_train, 10, 0.01);
    }
}
