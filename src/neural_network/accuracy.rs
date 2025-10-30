use super::NeuralNetwork;
use ndarray::Array2;

impl NeuralNetwork {
    /// Evaluates the network's classification accuracy on a dataset.
    ///
    /// Accuracy measures what percentage of predictions are correct. For each example,
    /// the network outputs 10 probabilities (one per digit class). The class with the
    /// highest probability is the prediction. This is compared to the true label.
    ///
    /// The process:
    /// 1. For each test example:
    ///    a. Perform forward propagation to get predictions
    ///    b. Find the output neuron with highest activation (argmax)
    ///    c. Find which output is 1.0 in the one-hot encoded true label
    ///    d. Compare predicted class with actual class
    /// 2. Calculate percentage of correct predictions
    ///
    /// # Arguments
    /// * `x` - Input images as 2D array (rows=examples, cols=784 pixels)
    /// * `y` - True labels as 2D array (rows=examples, cols=10 one-hot encoded)
    ///
    /// # Returns
    /// Accuracy as a percentage (0.0 to 100.0)
    ///
    /// # Example
    /// ```
    /// let acc = nn.accuracy(&x_test, &y_test);
    /// println!("Test accuracy: {:.2}%", acc);  // e.g., "Test accuracy: 96.82%"
    /// ```
    ///
    /// # Notes
    /// - Always evaluate on data the network hasn't seen during training (test set)
    /// - Expected accuracy on MNIST test set: ~96.8% for this architecture
    /// - Random guessing would give ~10% accuracy (10 classes)
    pub fn accuracy(&self, x: &Array2<f64>, y: &Array2<f64>) -> f64 {
        let mut correct_predictions = 0;

        // Evaluate each example in the dataset
        for i in 0..x.nrows() {
            let x_example = x.row(i).to_owned();

            // Get the network's prediction for this example
            // We only need a2 (the output), so ignore the intermediate values (_, _, _)
            let (_, _, _, a2) = self.feed_forward(x_example).expect("Feed forward failed");

            // === FIND PREDICTED CLASS ===
            // The predicted class is the output neuron with the highest activation
            // This is the argmax operation: find the index of the maximum value
            let mut max_value = a2[0];
            let mut predicted_class = 0;
            for (index, &value) in a2.iter().enumerate() {
                if value > max_value {
                    max_value = value;
                    predicted_class = index;
                }
            }

            // === FIND TRUE CLASS ===
            // The true class is encoded as one-hot: 9 zeros and one 1.0
            // Find which index has the 1.0 value
            let mut actual_class = None;
            for (index, &value) in y.row(i).iter().enumerate() {
                if value == 1.0 {
                    actual_class = Some(index);
                    break;
                }
            }
            let actual_class = actual_class.unwrap();

            // === COMPARE PREDICTION TO TRUTH ===
            // Increment counter if the network predicted correctly
            if predicted_class == actual_class {
                correct_predictions += 1;
            }
        }

        // Return accuracy as a percentage
        // (correct / total) * 100
        correct_predictions as f64 / x.nrows() as f64 * 100.0
    }
}
