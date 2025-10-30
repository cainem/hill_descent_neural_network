use ndarray::Array1;

/// Applies the sigmoid activation function to an array of values.
///
/// The sigmoid function maps any real-valued number to the range (0, 1),
/// making it useful for neural network activations. The formula is:
/// σ(z) = 1 / (1 + e^(-z))
///
/// # Arguments
/// * `z` - Input array of pre-activation values
///
/// # Returns
/// Array with sigmoid applied element-wise
pub fn sigmoid(z: &Array1<f64>) -> Array1<f64> {
    1.0 / (1.0 + (-z).mapv(f64::exp))
}

/// Computes the derivative of the sigmoid function.
///
/// The derivative is used during backpropagation to calculate gradients.
/// The formula is: σ'(z) = σ(z) * (1 - σ(z))
///
/// This derivative tells us how much the sigmoid output changes with respect
/// to changes in the input, which is essential for the chain rule in backprop.
///
/// # Arguments
/// * `z` - Input array of pre-activation values
///
/// # Returns
/// Array with sigmoid derivative applied element-wise
pub fn sigmoid_derivative(z: &Array1<f64>) -> Array1<f64> {
    let sig = sigmoid(z);
    sig.clone() * (1.0 - sig)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn given_zero_input_when_sigmoid_then_returns_half() {
        let input = arr1(&[0.0]);
        let result = sigmoid(&input);
        assert!((result[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn given_large_positive_input_when_sigmoid_then_approaches_one() {
        let input = arr1(&[10.0]);
        let result = sigmoid(&input);
        assert!(result[0] > 0.9999);
    }

    #[test]
    fn given_large_negative_input_when_sigmoid_then_approaches_zero() {
        let input = arr1(&[-10.0]);
        let result = sigmoid(&input);
        assert!(result[0] < 0.0001);
    }

    #[test]
    fn given_multiple_values_when_sigmoid_then_applies_elementwise() {
        let input = arr1(&[-2.0, 0.0, 2.0]);
        let result = sigmoid(&input);

        assert!(result[0] < 0.2); // Negative input -> small output
        assert!((result[1] - 0.5).abs() < 1e-10); // Zero input -> 0.5
        assert!(result[2] > 0.8); // Positive input -> large output
    }

    #[test]
    fn given_sigmoid_output_when_sigmoid_derivative_then_returns_correct_value() {
        // At z=0, sigmoid(0)=0.5, and derivative should be 0.5*(1-0.5)=0.25
        let input = arr1(&[0.0]);
        let result = sigmoid_derivative(&input);
        assert!((result[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn given_positive_value_when_sigmoid_derivative_then_less_than_quarter() {
        // Derivative is maximum at 0, decreases as we move away
        let input = arr1(&[2.0]);
        let result = sigmoid_derivative(&input);
        assert!(result[0] < 0.25);
        assert!(result[0] > 0.0);
    }

    #[test]
    fn given_negative_value_when_sigmoid_derivative_then_less_than_quarter() {
        // Derivative is symmetric around 0
        let input = arr1(&[-2.0]);
        let result = sigmoid_derivative(&input);
        assert!(result[0] < 0.25);
        assert!(result[0] > 0.0);
    }

    #[test]
    fn given_multiple_values_when_sigmoid_derivative_then_applies_elementwise() {
        let input = arr1(&[-1.0, 0.0, 1.0]);
        let result = sigmoid_derivative(&input);

        // All values should be positive and less than 0.25
        assert!(result[0] > 0.0 && result[0] < 0.25);
        assert!((result[1] - 0.25).abs() < 1e-10); // Max at 0
        assert!(result[2] > 0.0 && result[2] < 0.25);

        // Should be symmetric
        assert!((result[0] - result[2]).abs() < 1e-10);
    }
}
