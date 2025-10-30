use super::NeuralNetwork;
use ndarray::Array1;

impl NeuralNetwork {
    /// Calculates the binary cross-entropy loss between true and predicted values.
    ///
    /// Binary cross-entropy measures how far the network's predictions are from
    /// the true labels. Lower values indicate better predictions. The formula is:
    ///
    /// L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
    ///
    /// where y is the true label and ŷ is the predicted probability.
    ///
    /// # Arguments
    /// * `y_true` - True labels (one-hot encoded, values are 0.0 or 1.0)
    /// * `y_pred` - Predicted probabilities (output from sigmoid, range 0.0 to 1.0)
    ///
    /// # Returns
    /// The mean cross-entropy loss across all output neurons
    ///
    /// # Implementation Notes
    /// - Uses epsilon clipping (1e-15) to prevent log(0) which would be undefined
    /// - Clamps predictions to range [epsilon, 1-epsilon] for numerical stability
    /// - Returns the negative mean because we want to minimize loss
    pub fn loss_function(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        // Small epsilon value to prevent taking log of 0 or 1
        // log(0) = -∞ and log(1-1) = -∞, which would cause numerical issues
        let epsilon = 1e-15;

        // Clip predictions to valid range [epsilon, 1-epsilon]
        // This ensures numerical stability while having minimal impact on valid predictions
        let y_pred_clipped = y_pred.mapv(|p| p.max(epsilon).min(1.0 - epsilon));

        // Calculate cross-entropy for each output neuron:
        // For true label = 1: -log(ŷ) penalizes predictions far from 1
        // For true label = 0: -log(1-ŷ) penalizes predictions far from 0
        let loss = y_true * y_pred_clipped.mapv(|p| p.ln())
            + (1.0 - y_true) * (1.0 - y_pred_clipped).mapv(|p| p.ln());

        // Return the negative mean loss across all output neurons
        -loss.mean().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn given_perfect_prediction_when_loss_function_then_returns_near_zero() {
        let nn = NeuralNetwork::new(2, 3, 2);
        let y_true = arr1(&[1.0, 0.0]);
        let y_pred = arr1(&[0.9999, 0.0001]);

        let loss = nn.loss_function(&y_true, &y_pred);
        assert!(
            loss < 0.01,
            "Loss should be very small for accurate predictions"
        );
    }

    #[test]
    fn given_wrong_prediction_when_loss_function_then_returns_high_loss() {
        let nn = NeuralNetwork::new(2, 3, 2);
        let y_true = arr1(&[1.0, 0.0]);
        let y_pred = arr1(&[0.1, 0.9]);

        let loss = nn.loss_function(&y_true, &y_pred);
        assert!(loss > 1.0, "Loss should be high for wrong predictions");
    }

    #[test]
    fn given_medium_prediction_when_loss_function_then_returns_medium_loss() {
        let nn = NeuralNetwork::new(2, 3, 2);
        let y_true = arr1(&[1.0, 0.0]);
        let y_pred = arr1(&[0.5, 0.5]);

        let loss = nn.loss_function(&y_true, &y_pred);
        assert!(
            loss > 0.5 && loss < 1.5,
            "Loss should be moderate for uncertain predictions"
        );
    }

    #[test]
    fn given_all_zeros_true_when_loss_function_then_handles_correctly() {
        let nn = NeuralNetwork::new(2, 3, 3);
        let y_true = arr1(&[0.0, 0.0, 1.0]);
        let y_pred = arr1(&[0.1, 0.1, 0.8]);

        let loss = nn.loss_function(&y_true, &y_pred);
        assert!(loss.is_finite(), "Loss should be finite");
        assert!(loss >= 0.0, "Loss should be non-negative");
    }

    #[test]
    fn given_extreme_predictions_when_loss_function_then_clips_correctly() {
        let nn = NeuralNetwork::new(2, 3, 2);
        let y_true = arr1(&[1.0, 0.0]);
        // Extreme values that would cause log(0) without clipping
        let y_pred = arr1(&[1.0, 0.0]);

        let loss = nn.loss_function(&y_true, &y_pred);
        assert!(loss.is_finite(), "Loss should handle extreme values");
        assert!(loss < 0.1, "Perfect prediction should have very low loss");
    }

    #[test]
    fn given_multiple_outputs_when_loss_function_then_averages_correctly() {
        let nn = NeuralNetwork::new(2, 3, 5);
        let y_true = arr1(&[1.0, 0.0, 1.0, 0.0, 1.0]);
        let y_pred = arr1(&[0.9, 0.1, 0.9, 0.1, 0.9]);

        let loss = nn.loss_function(&y_true, &y_pred);
        assert!(loss.is_finite(), "Loss should be finite");
        assert!(
            loss < 0.2,
            "Good predictions across multiple outputs should have low loss"
        );
    }

    #[test]
    fn given_better_prediction_when_loss_function_then_lower_loss() {
        let nn = NeuralNetwork::new(2, 3, 2);
        let y_true = arr1(&[1.0, 0.0]);

        let y_pred_good = arr1(&[0.9, 0.1]);
        let y_pred_bad = arr1(&[0.6, 0.4]);

        let loss_good = nn.loss_function(&y_true, &y_pred_good);
        let loss_bad = nn.loss_function(&y_true, &y_pred_bad);

        assert!(
            loss_good < loss_bad,
            "Better predictions should have lower loss"
        );
    }
}
