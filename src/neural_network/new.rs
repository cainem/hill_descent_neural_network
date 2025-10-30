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
