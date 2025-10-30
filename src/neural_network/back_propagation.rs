use super::NeuralNetwork;
use crate::sigmoid::sigmoid_derivative;
use ndarray::{Array1, Axis};

impl NeuralNetwork {
    /// Performs backpropagation to compute gradients and update network weights.
    ///
    /// Backpropagation is the core learning algorithm. It uses the chain rule to compute
    /// how much each weight and bias contributed to the error, then adjusts them to reduce
    /// that error. The algorithm works backwards from the output to the input.
    ///
    /// Key steps:
    /// 1. Calculate output layer error (how wrong were the predictions)
    /// 2. Compute gradients for output layer weights and biases
    /// 3. Propagate error backward to hidden layer
    /// 4. Compute gradients for hidden layer weights and biases
    /// 5. Update all weights and biases using gradient descent
    ///
    /// # Arguments
    /// * `x` - Input vector for this training example
    /// * `y` - True label (one-hot encoded) for this training example
    /// * `z1` - Hidden layer pre-activation (from feed_forward)
    /// * `a1` - Hidden layer activation (from feed_forward)
    /// * `z2` - Output layer pre-activation (from feed_forward)
    /// * `a2` - Output layer activation / prediction (from feed_forward)
    /// * `learning_rate` - Step size for gradient descent (typically 0.01)
    ///
    /// # Implementation Notes
    /// - Uses element-wise operations for efficiency with ndarray
    /// - Gradient descent update rule: weight = weight - learning_rate * gradient
    /// - Smaller learning rates = slower but more stable learning
    /// - Larger learning rates = faster but may overshoot optimal values
    #[allow(clippy::too_many_arguments)] // All parameters are needed for backpropagation algorithm
    pub fn back_propagation(
        &mut self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        z1: &Array1<f64>,
        a1: &Array1<f64>,
        z2: &Array1<f64>,
        a2: &Array1<f64>,
        learning_rate: f64,
    ) {
        // === OUTPUT LAYER GRADIENTS ===

        // Calculate the error at the output layer: how far predictions are from truth
        // d_a2 = a2 - y (derivative of binary cross-entropy with respect to a2)
        let d_a2 = a2 - y;

        // Apply chain rule with sigmoid derivative to get gradient at z2
        // d_z2 = d_a2 * σ'(z2) where σ' is the sigmoid derivative
        // This tells us how the error changes with respect to the pre-activation values
        let d_z2 = d_a2.clone() * sigmoid_derivative(z2);

        // Calculate weight gradient for W2: how much each weight contributed to error
        // d_w2 = a1^T · d_z2
        // Shape transformation: (hidden_size, 1) · (1, output_size) = (hidden_size, output_size)
        let d_w2 = a1
            .clone()
            .insert_axis(Axis(1))
            .dot(&d_z2.clone().insert_axis(Axis(0)));

        // Bias gradient is simply the error gradient (derivative is 1)
        let d_b2 = d_z2.clone();

        // === HIDDEN LAYER GRADIENTS ===

        // Propagate the error backward to the hidden layer
        // d_z1 = (d_z2 · W2^T) * σ'(z1)
        // The W2^T multiplication distributes the output error to hidden neurons
        // Then we apply sigmoid derivative for the hidden layer activation
        let d_z1 = d_z2.dot(&self.W2.t()) * sigmoid_derivative(z1);

        // Calculate weight gradient for W1
        // d_w1 = x^T · d_z1
        // Shape transformation: (input_size, 1) · (1, hidden_size) = (input_size, hidden_size)
        let d_w1 = x
            .clone()
            .insert_axis(Axis(1))
            .dot(&d_z1.clone().insert_axis(Axis(0)));

        // Bias gradient for hidden layer
        let d_b1 = d_z1.clone();

        // === GRADIENT DESCENT UPDATE ===
        // Move each parameter in the opposite direction of its gradient
        // Multiplied by learning_rate to control the step size

        self.W1 -= &(learning_rate * &d_w1); // Update input-to-hidden weights
        self.W2 -= &(learning_rate * &d_w2); // Update hidden-to-output weights
        self.b1 -= &(learning_rate * &d_b1); // Update hidden layer biases
        self.b2 -= &(learning_rate * &d_b2); // Update output layer biases
    }
}
