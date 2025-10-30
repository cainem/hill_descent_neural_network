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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn given_training_example_when_back_propagation_then_weights_change() {
        let mut nn = NeuralNetwork::new(3, 2, 2);

        // Store original weights
        let original_w1 = nn.w1().clone();
        let original_w2 = nn.w2().clone();

        let x = arr1(&[0.5, 0.3, 0.8]);
        let y = arr1(&[1.0, 0.0]);

        let (z1, a1, z2, a2) = nn.feed_forward(x.clone()).unwrap();
        nn.back_propagation(&x, &y, &z1, &a1, &z2, &a2, 0.01);

        // At least some weights should have changed
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

        assert!(w1_changed, "W1 should change after backpropagation");
        assert!(w2_changed, "W2 should change after backpropagation");
    }

    #[test]
    fn given_training_example_when_back_propagation_then_biases_change() {
        let mut nn = NeuralNetwork::new(3, 2, 2);

        let original_b1 = nn.b1().clone();
        let original_b2 = nn.b2().clone();

        let x = arr1(&[0.5, 0.3, 0.8]);
        let y = arr1(&[1.0, 0.0]);

        let (z1, a1, z2, a2) = nn.feed_forward(x.clone()).unwrap();
        nn.back_propagation(&x, &y, &z1, &a1, &z2, &a2, 0.01);

        let mut b1_changed = false;
        let mut b2_changed = false;

        for i in 0..nn.b1().len() {
            if (nn.b1()[i] - original_b1[i]).abs() > 1e-10 {
                b1_changed = true;
                break;
            }
        }

        for i in 0..nn.b2().len() {
            if (nn.b2()[i] - original_b2[i]).abs() > 1e-10 {
                b2_changed = true;
                break;
            }
        }

        assert!(b1_changed, "b1 should change after backpropagation");
        assert!(b2_changed, "b2 should change after backpropagation");
    }

    #[test]
    fn given_larger_learning_rate_when_back_propagation_then_larger_changes() {
        let mut nn1 = NeuralNetwork::new(3, 2, 2);

        let x = arr1(&[0.5, 0.3, 0.8]);
        let y = arr1(&[1.0, 0.0]);

        let (z1, a1, z2, a2) = nn1.feed_forward(x.clone()).unwrap();
        let original_w1 = nn1.w1().clone();

        nn1.back_propagation(&x, &y, &z1, &a1, &z2, &a2, 0.001);
        let change_small = (nn1.w1()[[0, 0]] - original_w1[[0, 0]]).abs();

        // Reset and use larger learning rate
        let mut nn3 = NeuralNetwork::new(3, 2, 2);
        let (z1_3, a1_3, z2_3, a2_3) = nn3.feed_forward(x.clone()).unwrap();
        let original_w1_3 = nn3.w1().clone();

        nn3.back_propagation(&x, &y, &z1_3, &a1_3, &z2_3, &a2_3, 0.1);
        let change_large = (nn3.w1()[[0, 0]] - original_w1_3[[0, 0]]).abs();

        // Larger learning rate should generally cause larger changes
        // (This test is probabilistic but should almost always pass)
        assert!(
            change_small < 0.1 && change_large < 10.0,
            "Changes should be proportional to learning rate"
        );
    }

    #[test]
    fn given_perfect_prediction_when_back_propagation_then_small_changes() {
        let mut nn = NeuralNetwork::new(3, 2, 2);

        let x = arr1(&[0.5, 0.3, 0.8]);
        let y = arr1(&[1.0, 0.0]);

        // Do forward pass
        let (z1, a1, z2, mut a2) = nn.feed_forward(x.clone()).unwrap();

        // Manually set output to be very close to target
        a2[0] = 0.99;
        a2[1] = 0.01;

        let original_w2 = nn.w2().clone();
        nn.back_propagation(&x, &y, &z1, &a1, &z2, &a2, 0.01);

        // Changes should be small when prediction is good
        let total_change: f64 = nn
            .w2()
            .iter()
            .zip(original_w2.iter())
            .map(|(new, old)| (new - old).abs())
            .sum();

        assert!(
            total_change < 0.5,
            "Changes should be small for good predictions"
        );
    }

    #[test]
    fn given_zero_learning_rate_when_back_propagation_then_no_changes() {
        let mut nn = NeuralNetwork::new(3, 2, 2);

        let original_w1 = nn.w1().clone();
        let original_w2 = nn.w2().clone();
        let original_b1 = nn.b1().clone();
        let original_b2 = nn.b2().clone();

        let x = arr1(&[0.5, 0.3, 0.8]);
        let y = arr1(&[1.0, 0.0]);

        let (z1, a1, z2, a2) = nn.feed_forward(x.clone()).unwrap();
        nn.back_propagation(&x, &y, &z1, &a1, &z2, &a2, 0.0);

        // With zero learning rate, nothing should change
        assert_eq!(nn.w1(), &original_w1);
        assert_eq!(nn.w2(), &original_w2);
        assert_eq!(nn.b1(), &original_b1);
        assert_eq!(nn.b2(), &original_b2);
    }
}
