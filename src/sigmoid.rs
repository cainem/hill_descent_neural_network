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
