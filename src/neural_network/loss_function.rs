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
