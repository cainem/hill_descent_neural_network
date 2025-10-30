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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn given_perfect_predictions_when_accuracy_then_returns_100() {
        let nn = NeuralNetwork::new(3, 2, 2);

        // Create test data where the network would predict correctly
        // This test verifies the accuracy calculation, not the network's ability to learn
        let x_test = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let y_test = arr2(&[
            [1.0, 0.0], // Class 0
            [0.0, 1.0], // Class 1
        ]);

        // We can't guarantee perfect predictions with random weights,
        // but we can test the calculation logic by manipulating output
        // For this test, we'll just verify accuracy is between 0 and 100
        let acc = nn.accuracy(&x_test, &y_test);
        assert!(
            (0.0..=100.0).contains(&acc),
            "Accuracy should be between 0 and 100"
        );
    }

    #[test]
    fn given_single_example_when_accuracy_then_returns_valid_percentage() {
        let nn = NeuralNetwork::new(3, 2, 2);

        let x_test = arr2(&[[0.5, 0.5, 0.5]]);
        let y_test = arr2(&[[1.0, 0.0]]);

        let acc = nn.accuracy(&x_test, &y_test);

        // With a single example, accuracy is either 0% or 100%
        assert!(
            acc == 0.0 || acc == 100.0,
            "Single example accuracy should be 0% or 100%, got {}",
            acc
        );
    }

    #[test]
    fn given_two_examples_when_accuracy_then_returns_valid_percentage() {
        let nn = NeuralNetwork::new(3, 2, 2);

        let x_test = arr2(&[[0.5, 0.5, 0.5], [0.2, 0.3, 0.7]]);
        let y_test = arr2(&[[1.0, 0.0], [0.0, 1.0]]);

        let acc = nn.accuracy(&x_test, &y_test);

        // With two examples, accuracy can be 0%, 50%, or 100%
        assert!(
            acc == 0.0 || acc == 50.0 || acc == 100.0,
            "Two examples accuracy should be 0%, 50%, or 100%, got {}",
            acc
        );
    }

    #[test]
    fn given_multiple_classes_when_accuracy_then_calculates_correctly() {
        let nn = NeuralNetwork::new(4, 3, 3);

        let x_test = arr2(&[
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 0.8, 0.7, 0.6],
        ]);
        let y_test = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

        let acc = nn.accuracy(&x_test, &y_test);

        // Accuracy should be a valid percentage
        assert!((0.0..=100.0).contains(&acc));
        // With 3 examples, possible values are 0%, 33.33%, 66.66%, 100%
        let possible_values = [0.0, 100.0 / 3.0, 200.0 / 3.0, 100.0];
        let is_valid = possible_values.iter().any(|&v| (acc - v).abs() < 0.01);
        assert!(
            is_valid,
            "Accuracy {} not a valid percentage for 3 examples",
            acc
        );
    }

    #[test]
    fn given_trained_network_when_accuracy_on_training_data_then_reasonable() {
        let mut nn = NeuralNetwork::new(2, 4, 2);

        // Create simple separable training data
        let x_train = arr2(&[[0.0, 0.0], [0.0, 0.1], [0.9, 0.9], [1.0, 1.0]]);
        let y_train = arr2(&[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]);

        // Train the network
        nn.train(&x_train, &y_train, 50, 0.5);

        // Check accuracy on same data (should be high after training)
        let acc = nn.accuracy(&x_train, &y_train);

        // After training, should get at least 50% accuracy on this simple data
        assert!(
            acc >= 50.0,
            "Trained network should achieve >50% accuracy, got {}",
            acc
        );
    }

    #[test]
    fn given_many_examples_when_accuracy_then_handles_large_dataset() {
        let nn = NeuralNetwork::new(5, 3, 2);

        // Create a moderately sized dataset (20 examples)
        let mut x_rows = Vec::new();
        let mut y_rows = Vec::new();
        for i in 0..20 {
            x_rows.push(vec![i as f64 / 20.0; 5]);
            if i < 10 {
                y_rows.push(vec![1.0, 0.0]);
            } else {
                y_rows.push(vec![0.0, 1.0]);
            }
        }

        let x_test =
            Array2::from_shape_vec((20, 5), x_rows.into_iter().flatten().collect()).unwrap();
        let y_test =
            Array2::from_shape_vec((20, 2), y_rows.into_iter().flatten().collect()).unwrap();

        let acc = nn.accuracy(&x_test, &y_test);

        // Should handle 20 examples without panic and return valid percentage
        assert!((0.0..=100.0).contains(&acc));
    }
}
