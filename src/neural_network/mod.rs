use ndarray::{Array1, Array2};
use serde_derive::{Deserialize, Serialize};

// Import all method implementations
mod accuracy;
mod back_propagation;
mod feed_forward;
mod load;
mod loss_function;
mod new;
mod save;
mod train;

/// A simple 3-layer feedforward neural network for classification.
///
/// Architecture:
/// - Input layer: `input_size` neurons (e.g., 784 for MNIST 28x28 images)
/// - Hidden layer: `hidden_size` neurons with sigmoid activation
/// - Output layer: `output_size` neurons with sigmoid activation (e.g., 10 for digits 0-9)
///
/// Training uses stochastic gradient descent with backpropagation.
/// Loss function is binary cross-entropy.
#[derive(Serialize, Deserialize, Debug)]
#[allow(non_snake_case)] // W1, W2 are standard notation in neural network literature
pub struct NeuralNetwork {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    W1: Array2<f64>, // Weights from input to hidden layer
    W2: Array2<f64>, // Weights from hidden to output layer
    b1: Array1<f64>, // Biases for hidden layer
    b2: Array1<f64>, // Biases for output layer
}

impl NeuralNetwork {
    // Getters for private fields

    /// Returns the input layer size
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Returns the hidden layer size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Returns the output layer size
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Returns a reference to the input-to-hidden weights matrix
    pub fn w1(&self) -> &Array2<f64> {
        &self.W1
    }

    /// Returns a reference to the hidden-to-output weights matrix
    pub fn w2(&self) -> &Array2<f64> {
        &self.W2
    }

    /// Returns a reference to the hidden layer biases
    pub fn b1(&self) -> &Array1<f64> {
        &self.b1
    }

    /// Returns a reference to the output layer biases
    pub fn b2(&self) -> &Array1<f64> {
        &self.b2
    }
}
