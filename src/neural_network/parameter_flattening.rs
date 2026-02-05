use super::NeuralNetwork;

impl NeuralNetwork {
    /// Flattens all network parameters (W1, W2, b1, b2) into a single vector.
    ///
    /// This conversion is necessary for genetic algorithm optimization, which operates
    /// on flat parameter vectors rather than structured weight matrices and bias vectors.
    ///
    /// The flattening order is:
    /// 1. W1 elements (row-major order)
    /// 2. b1 elements
    /// 3. W2 elements (row-major order)
    /// 4. b2 elements
    ///
    /// # Returns
    /// A 1D vector containing all network parameters in order
    ///
    /// # Example
    /// For a network with:
    /// - W1: (784, 64) = 50,176 parameters
    /// - b1: 64 parameters
    /// - W2: (64, 10) = 640 parameters
    /// - b2: 10 parameters
    ///   Total: 50,890 parameters
    pub fn flatten_parameters(&self) -> Vec<f64> {
        let mut params = Vec::with_capacity(self.parameter_count());

        // Flatten W1 (input to hidden weights)
        if let Some(slice) = self.W1.as_slice() {
            params.extend_from_slice(slice);
        } else {
            params.extend(self.W1.iter().copied());
        }

        // Flatten b1 (hidden layer biases)
        if let Some(slice) = self.b1.as_slice() {
            params.extend_from_slice(slice);
        } else {
            params.extend(self.b1.iter().copied());
        }

        // Flatten W2 (hidden to output weights)
        if let Some(slice) = self.W2.as_slice() {
            params.extend_from_slice(slice);
        } else {
            params.extend(self.W2.iter().copied());
        }

        // Flatten b2 (output layer biases)
        if let Some(slice) = self.b2.as_slice() {
            params.extend_from_slice(slice);
        } else {
            params.extend(self.b2.iter().copied());
        }

        params
    }

    /// Reconstructs network parameters from a flat vector.
    ///
    /// This is the inverse operation of `flatten_parameters`. It takes a flat vector
    /// produced by the genetic algorithm and updates the network's weights and biases.
    ///
    /// The vector must contain exactly the right number of parameters for the network's
    /// architecture, matching the flattening order described in `flatten_parameters`.
    ///
    /// # Arguments
    /// * `params` - Flat vector of parameters in the expected order
    ///
    /// # Panics
    /// Panics if the parameter vector doesn't contain the expected number of elements
    ///
    /// # Implementation Notes
    /// - Updates existing arrays in-place to avoid allocations
    /// - Parameters are extracted in the same order they were flattened
    pub fn unflatten_parameters(&mut self, params: &[f64]) {
        assert_eq!(params.len(), self.parameter_count(), "Parameter count mismatch");
        let mut offset = 0;

        // Reconstruct W1
        let w1_size = self.input_size * self.hidden_size;
        let w1_slice = &params[offset..offset + w1_size];
        if let Some(slice) = self.W1.as_slice_mut() {
            slice.copy_from_slice(w1_slice);
        } else {
            for (i, &val) in self.W1.iter_mut().zip(w1_slice) {
                *i = val;
            }
        }
        offset += w1_size;

        // Reconstruct b1
        let b1_size = self.hidden_size;
        let b1_slice = &params[offset..offset + b1_size];
        if let Some(slice) = self.b1.as_slice_mut() {
            slice.copy_from_slice(b1_slice);
        } else {
            for (i, &val) in self.b1.iter_mut().zip(b1_slice) {
                *i = val;
            }
        }
        offset += b1_size;

        // Reconstruct W2
        let w2_size = self.hidden_size * self.output_size;
        let w2_slice = &params[offset..offset + w2_size];
        if let Some(slice) = self.W2.as_slice_mut() {
            slice.copy_from_slice(w2_slice);
        } else {
            for (i, &val) in self.W2.iter_mut().zip(w2_slice) {
                *i = val;
            }
        }
        offset += w2_size;

        // Reconstruct b2
        let b2_size = self.output_size;
        let b2_slice = &params[offset..offset + b2_size];
        if let Some(slice) = self.b2.as_slice_mut() {
            slice.copy_from_slice(b2_slice);
        } else {
            for (i, &val) in self.b2.iter_mut().zip(b2_slice) {
                *i = val;
            }
        }
    }

    /// Returns the total number of parameters in the network.
    ///
    /// Useful for validation and memory allocation calculations.
    ///
    /// # Returns
    /// Total count of all weights and biases
    pub fn parameter_count(&self) -> usize {
        let w1_count = self.input_size * self.hidden_size;
        let b1_count = self.hidden_size;
        let w2_count = self.hidden_size * self.output_size;
        let b2_count = self.output_size;

        w1_count + b1_count + w2_count + b2_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn given_network_when_flatten_then_correct_size() {
        let nn = NeuralNetwork::new(784, 64, 10);
        let params = nn.flatten_parameters();

        // Expected: 784*64 + 64 + 64*10 + 10 = 50,176 + 64 + 640 + 10 = 50,890
        assert_eq!(params.len(), 50_890);
    }

    #[test]
    fn given_flattened_params_when_unflatten_then_reconstructs_network() {
        let original = NeuralNetwork::new(784, 64, 10);
        let params = original.flatten_parameters();

        let mut reconstructed = NeuralNetwork::new(784, 64, 10);
        reconstructed.unflatten_parameters(&params);

        // Verify all parameters match
        assert_eq!(reconstructed.w1(), original.w1());
        assert_eq!(reconstructed.b1(), original.b1());
        assert_eq!(reconstructed.w2(), original.w2());
        assert_eq!(reconstructed.b2(), original.b2());
    }

    #[test]
    fn given_small_network_when_flatten_unflatten_then_round_trip_works() {
        let mut nn = NeuralNetwork::new(3, 2, 2);

        // Get original values
        let original_w1 = nn.w1().clone();
        let original_b1 = nn.b1().clone();
        let original_w2 = nn.w2().clone();
        let original_b2 = nn.b2().clone();

        // Flatten and unflatten
        let params = nn.flatten_parameters();
        nn.unflatten_parameters(&params);

        // Verify nothing changed
        assert_eq!(nn.w1(), &original_w1);
        assert_eq!(nn.b1(), &original_b1);
        assert_eq!(nn.w2(), &original_w2);
        assert_eq!(nn.b2(), &original_b2);
    }

    #[test]
    fn given_network_when_parameter_count_then_matches_flatten_length() {
        let nn = NeuralNetwork::new(784, 64, 10);
        let params = nn.flatten_parameters();

        assert_eq!(nn.parameter_count(), params.len());
    }

    #[test]
    fn given_different_architectures_when_parameter_count_then_correct() {
        let nn1 = NeuralNetwork::new(784, 64, 10);
        assert_eq!(nn1.parameter_count(), 50_890);

        let nn2 = NeuralNetwork::new(784, 16, 10);
        assert_eq!(nn2.parameter_count(), 784 * 16 + 16 + 16 * 10 + 10);

        let nn3 = NeuralNetwork::new(4, 3, 2);
        assert_eq!(nn3.parameter_count(), 4 * 3 + 3 + 3 * 2 + 2);
    }

    #[test]
    fn given_modified_network_when_flatten_then_captures_changes() {
        let mut nn = NeuralNetwork::new(3, 2, 2);

        // Flatten original
        let original_params = nn.flatten_parameters();

        // Modify parameters
        nn.unflatten_parameters(&vec![1.0; nn.parameter_count()]);

        // Flatten modified
        let modified_params = nn.flatten_parameters();

        // Verify they're different
        assert_ne!(original_params, modified_params);
        assert!(modified_params.iter().all(|&x| x == 1.0));
    }
}
