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
    /// let (z1, a1, z2, a2) = nn.feed_forward(input_vector)?;
    /// // a2 contains the network's predictions
    /// ```
    pub fn feed_forward(
        &self,
        x: Array1<f64>,
    ) -> FeedForwardResult {
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
