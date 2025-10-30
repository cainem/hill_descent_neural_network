use super::NeuralNetwork;
use ndarray::Array;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

impl NeuralNetwork {
    /// Creates a new neural network with random weights and biases.
    ///
    /// Weights and biases are initialized with uniform random values in the range [-1.0, 1.0].
    /// This random initialization is important to break symmetry - if all weights started
    /// at the same value, all neurons would learn the same features.
    ///
    /// # Arguments
    /// * `input_size` - Number of input neurons (e.g., 784 for 28x28 images)
    /// * `hidden_size` - Number of neurons in the hidden layer
    /// * `output_size` - Number of output neurons (e.g., 10 for digit classification)
    ///
    /// # Returns
    /// A new `NeuralNetwork` with randomly initialized parameters
    ///
    /// # Example
    /// ```
    /// let nn = NeuralNetwork::new(784, 64, 10);
    /// ```
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            // W1 shape: (input_size × hidden_size) - transforms input to hidden layer
            W1: Array::random((input_size, hidden_size), Uniform::new(-1.0, 1.0)),
            // W2 shape: (hidden_size × output_size) - transforms hidden to output layer
            W2: Array::random((hidden_size, output_size), Uniform::new(-1.0, 1.0)),
            // b1 shape: (hidden_size) - biases for hidden layer neurons
            b1: Array::random(hidden_size, Uniform::new(-1.0, 1.0)),
            // b2 shape: (output_size) - biases for output layer neurons
            b2: Array::random(output_size, Uniform::new(-1.0, 1.0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn given_valid_dimensions_when_new_then_creates_network() {
        let nn = NeuralNetwork::new(784, 64, 10);

        assert_eq!(nn.input_size(), 784);
        assert_eq!(nn.hidden_size(), 64);
        assert_eq!(nn.output_size(), 10);
    }

    #[test]
    fn given_new_network_when_checking_weights_then_correct_shapes() {
        let nn = NeuralNetwork::new(784, 64, 10);

        // W1 should be (input_size × hidden_size)
        assert_eq!(nn.w1().shape(), &[784, 64]);

        // W2 should be (hidden_size × output_size)
        assert_eq!(nn.w2().shape(), &[64, 10]);

        // b1 should be (hidden_size)
        assert_eq!(nn.b1().len(), 64);

        // b2 should be (output_size)
        assert_eq!(nn.b2().len(), 10);
    }

    #[test]
    fn given_new_network_when_checking_weights_then_values_in_range() {
        let nn = NeuralNetwork::new(10, 5, 2);

        // All weights should be between -1.0 and 1.0
        for &val in nn.w1().iter() {
            assert!((-1.0..=1.0).contains(&val));
        }

        for &val in nn.w2().iter() {
            assert!((-1.0..=1.0).contains(&val));
        }

        for &val in nn.b1().iter() {
            assert!((-1.0..=1.0).contains(&val));
        }

        for &val in nn.b2().iter() {
            assert!((-1.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn given_two_networks_when_created_then_different_weights() {
        let nn1 = NeuralNetwork::new(10, 5, 2);
        let nn2 = NeuralNetwork::new(10, 5, 2);

        // Random initialization should give different values
        // Check if at least some weights are different
        let mut has_difference = false;
        for i in 0..nn1.w1().len() {
            if (nn1.w1()[[i / 5, i % 5]] - nn2.w1()[[i / 5, i % 5]]).abs() > 1e-10 {
                has_difference = true;
                break;
            }
        }

        assert!(
            has_difference,
            "Two networks should have different random weights"
        );
    }

    #[test]
    fn given_small_dimensions_when_new_then_creates_network() {
        let nn = NeuralNetwork::new(2, 3, 1);

        assert_eq!(nn.input_size(), 2);
        assert_eq!(nn.hidden_size(), 3);
        assert_eq!(nn.output_size(), 1);
    }

    #[test]
    fn given_large_dimensions_when_new_then_creates_network() {
        let nn = NeuralNetwork::new(1000, 500, 100);

        assert_eq!(nn.input_size(), 1000);
        assert_eq!(nn.hidden_size(), 500);
        assert_eq!(nn.output_size(), 100);
    }
}
