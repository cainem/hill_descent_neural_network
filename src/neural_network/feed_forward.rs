use super::NeuralNetwork;
use crate::sigmoid::sigmoid;
use ndarray::Array1;

/// Return type for feed_forward containing all intermediate layer values
pub type FeedForwardResult = Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), String>;

impl NeuralNetwork {
    /// Performs forward propagation through the network.
    ///
    /// Forward propagation computes the network's output by passing the input
    /// through each layer, applying weights, biases, and activation functions.
    /// This is the "prediction" phase of the neural network.
    ///
    /// The computation flow:
    /// 1. Input layer (X) → Hidden layer: Z1 = X·W1 + b1, then A1 = sigmoid(Z1)
    /// 2. Hidden layer (A1) → Output layer: Z2 = A1·W2 + b2, then A2 = sigmoid(Z2)
    ///
    /// # Arguments
    /// * `x` - Input vector (e.g., 784 values for a flattened 28×28 image)
    ///
    /// # Returns
    /// * `Ok((z1, a1, z2, a2))` - All intermediate values needed for backpropagation:
    ///   - z1: Hidden layer pre-activation (weighted sum before sigmoid)
    ///   - a1: Hidden layer activation (after sigmoid)
    ///   - z2: Output layer pre-activation
    ///   - a2: Output layer activation (final predictions)
    /// * `Err(String)` - If input size doesn't match the network's expected input size
    ///
    /// # Example
    /// ```
    /// use neural_network_scratch::NeuralNetwork;
    /// use ndarray::Array1;
    ///
    /// let nn = NeuralNetwork::new(784, 64, 10);
    /// let input_vector = Array1::zeros(784);
    /// let (z1, a1, z2, a2) = nn.feed_forward(input_vector)?;
    /// // a2 contains the network's predictions
    /// # Ok::<(), String>(())
    /// ```
    pub fn feed_forward(&self, x: Array1<f64>) -> FeedForwardResult {
        // Validate input dimensions match what the network expects
        if x.len() != self.input_size {
            return Err(format!(
                "Input size mismatch: expected {}, got {}",
                self.input_size,
                x.len()
            ));
        }

        // === HIDDEN LAYER COMPUTATION ===
        // z1 = x·W1 + b1
        // Matrix multiplication: (1×input_size) · (input_size×hidden_size) + (hidden_size) = (hidden_size)
        // This computes the weighted sum of inputs for each hidden neuron
        let z1 = x.dot(&self.W1) + &self.b1;

        // Apply sigmoid activation function to add non-linearity
        // Without activation, multiple layers would collapse to a single linear transformation
        let a1 = sigmoid(&z1);

        // === OUTPUT LAYER COMPUTATION ===
        // z2 = a1·W2 + b2
        // Matrix multiplication: (1×hidden_size) · (hidden_size×output_size) + (output_size) = (output_size)
        // This computes the weighted sum of hidden activations for each output neuron
        let z2 = a1.dot(&self.W2) + &self.b2;

        // Apply sigmoid activation to get final predictions in range [0, 1]
        // For classification, each output represents the probability of that class
        let a2 = sigmoid(&z2);

        // Return all intermediate values - needed for backpropagation gradient calculations
        Ok((z1, a1, z2, a2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn given_valid_input_when_feed_forward_then_returns_ok() {
        let nn = NeuralNetwork::new(5, 3, 2);
        let input = arr1(&[0.5, 0.3, 0.8, 0.1, 0.6]);

        let result = nn.feed_forward(input);
        assert!(result.is_ok());
    }

    #[test]
    fn given_valid_input_when_feed_forward_then_output_in_valid_range() {
        let nn = NeuralNetwork::new(5, 3, 2);
        let input = arr1(&[0.5, 0.3, 0.8, 0.1, 0.6]);

        let (_, _, _, a2) = nn.feed_forward(input).unwrap();

        // All outputs should be between 0 and 1 due to sigmoid
        for &val in a2.iter() {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn given_wrong_input_size_when_feed_forward_then_returns_err() {
        let nn = NeuralNetwork::new(5, 3, 2);
        let input = arr1(&[0.5, 0.3, 0.8]); // Wrong size: 3 instead of 5

        let result = nn.feed_forward(input);
        assert!(result.is_err());
    }

    #[test]
    fn given_zero_input_when_feed_forward_then_produces_output() {
        let nn = NeuralNetwork::new(3, 2, 2);
        let input = arr1(&[0.0, 0.0, 0.0]);

        let (_, _, _, a2) = nn.feed_forward(input).unwrap();

        // Even with zero input, biases should produce non-zero output
        assert_eq!(a2.len(), 2);
    }

    #[test]
    fn given_large_input_when_feed_forward_then_output_bounded() {
        let nn = NeuralNetwork::new(3, 2, 2);
        let input = arr1(&[100.0, 100.0, 100.0]);

        let (_, _, _, a2) = nn.feed_forward(input).unwrap();

        // Sigmoid should keep outputs in [0, 1] even with large inputs
        for &val in a2.iter() {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn given_negative_input_when_feed_forward_then_output_bounded() {
        let nn = NeuralNetwork::new(3, 2, 2);
        let input = arr1(&[-100.0, -100.0, -100.0]);

        let (_, _, _, a2) = nn.feed_forward(input).unwrap();

        // Sigmoid should keep outputs in [0, 1] even with large negative inputs
        for &val in a2.iter() {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn given_valid_input_when_feed_forward_then_hidden_layer_correct_size() {
        let nn = NeuralNetwork::new(5, 3, 2);
        let input = arr1(&[0.5, 0.3, 0.8, 0.1, 0.6]);

        let (z1, a1, _, _) = nn.feed_forward(input).unwrap();

        assert_eq!(z1.len(), 3);
        assert_eq!(a1.len(), 3);
    }

    #[test]
    fn given_valid_input_when_feed_forward_then_output_layer_correct_size() {
        let nn = NeuralNetwork::new(5, 3, 2);
        let input = arr1(&[0.5, 0.3, 0.8, 0.1, 0.6]);

        let (_, _, z2, a2) = nn.feed_forward(input).unwrap();

        assert_eq!(z2.len(), 2);
        assert_eq!(a2.len(), 2);
    }

    #[test]
    fn given_same_input_when_feed_forward_twice_then_same_output() {
        let nn = NeuralNetwork::new(3, 2, 2);
        let input = arr1(&[0.5, 0.3, 0.8]);

        let (_, _, _, a2_first) = nn.feed_forward(input.clone()).unwrap();
        let (_, _, _, a2_second) = nn.feed_forward(input).unwrap();

        // Should be deterministic
        for i in 0..a2_first.len() {
            assert_eq!(a2_first[i], a2_second[i]);
        }
    }

    #[test]
    fn given_different_inputs_when_feed_forward_then_different_outputs() {
        let nn = NeuralNetwork::new(3, 2, 2);
        let input1 = arr1(&[0.1, 0.2, 0.3]);
        let input2 = arr1(&[0.7, 0.8, 0.9]);

        let (_, _, _, a2_first) = nn.feed_forward(input1).unwrap();
        let (_, _, _, a2_second) = nn.feed_forward(input2).unwrap();

        // Different inputs should generally produce different outputs
        let mut has_difference = false;
        for i in 0..a2_first.len() {
            if (a2_first[i] - a2_second[i]).abs() > 1e-6 {
                has_difference = true;
                break;
            }
        }
        assert!(
            has_difference,
            "Different inputs should produce different outputs"
        );
    }
}
