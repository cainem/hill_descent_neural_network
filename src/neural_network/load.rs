use super::NeuralNetwork;
use std::fs::File;
use std::io::{self, Read};
use std::path::PathBuf;

impl NeuralNetwork {
    /// Loads a trained neural network from a binary file.
    ///
    /// The network is deserialized from a binary format using bincode.
    /// This function enforces that loaded models must have the correct dimensions
    /// for MNIST (784 inputs, 10 outputs) to prevent loading incompatible models.
    ///
    /// # Arguments
    /// * `path` - Path to the saved model file
    ///
    /// # Returns
    /// * `Ok(NeuralNetwork)` - Successfully loaded network
    /// * `Err(io::Error)` - If file cannot be opened, read, deserialized, or has wrong dimensions
    ///
    /// # Errors
    /// - File I/O errors (file not found, permission denied, etc.)
    /// - Deserialization errors (corrupted file, incompatible format)
    /// - Dimension validation errors (not 784 inputs or 10 outputs)
    ///
    /// # Example
    /// ```
    /// let nn = NeuralNetwork::load(&PathBuf::from("model.bin"))?;
    /// ```
    pub fn load(path: &PathBuf) -> io::Result<NeuralNetwork> {
        // Open and read the entire file into memory
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        // Deserialize the binary data into a NeuralNetwork struct
        let network: NeuralNetwork = bincode::deserialize(&buffer)
            .map_err(|e| io::Error::other(format!("Deserialization error: {:?}", e)))?;

        // Validate that the network has the expected dimensions for MNIST
        // This prevents accidentally loading models trained for different tasks
        if network.input_size != 784 || network.output_size != 10 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Invalid network dimensions: expected 784 inputs and 10 outputs, got {} inputs and {} outputs",
                    network.input_size, network.output_size
                ),
            ));
        }

        Ok(network)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn given_saved_network_when_load_then_succeeds() {
        let nn = NeuralNetwork::new(784, 64, 10);
        let path = PathBuf::from("test_load.bin");

        nn.save(&path).unwrap();
        let loaded = NeuralNetwork::load(&path);

        assert!(loaded.is_ok());

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn given_saved_network_when_load_then_preserves_dimensions() {
        let nn = NeuralNetwork::new(784, 64, 10);
        let path = PathBuf::from("test_load_dims.bin");

        nn.save(&path).unwrap();
        let loaded = NeuralNetwork::load(&path).unwrap();

        assert_eq!(loaded.input_size(), 784);
        assert_eq!(loaded.hidden_size(), 64);
        assert_eq!(loaded.output_size(), 10);

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn given_saved_network_when_load_then_preserves_weights() {
        let nn = NeuralNetwork::new(784, 64, 10);
        let path = PathBuf::from("test_load_weights.bin");

        nn.save(&path).unwrap();
        let loaded = NeuralNetwork::load(&path).unwrap();

        // Check shapes match
        assert_eq!(loaded.w1().shape(), nn.w1().shape());
        assert_eq!(loaded.w2().shape(), nn.w2().shape());
        assert_eq!(loaded.b1().len(), nn.b1().len());
        assert_eq!(loaded.b2().len(), nn.b2().len());

        // Check at least some values match (sampling a few to avoid huge comparison)
        assert_eq!(loaded.w1()[[0, 0]], nn.w1()[[0, 0]]);
        assert_eq!(loaded.w2()[[0, 0]], nn.w2()[[0, 0]]);
        assert_eq!(loaded.b1()[0], nn.b1()[0]);
        assert_eq!(loaded.b2()[0], nn.b2()[0]);

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn given_nonexistent_file_when_load_then_returns_err() {
        let path = PathBuf::from("nonexistent_file.bin");
        let result = NeuralNetwork::load(&path);

        assert!(result.is_err());
    }

    #[test]
    fn given_wrong_input_size_when_load_then_returns_err() {
        // Create a network with wrong input size
        let nn = NeuralNetwork::new(100, 64, 10); // Should be 784
        let path = PathBuf::from("test_wrong_input.bin");

        nn.save(&path).unwrap();
        let result = NeuralNetwork::load(&path);

        assert!(result.is_err());

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn given_wrong_output_size_when_load_then_returns_err() {
        // Create a network with wrong output size
        let nn = NeuralNetwork::new(784, 64, 5); // Should be 10
        let path = PathBuf::from("test_wrong_output.bin");

        nn.save(&path).unwrap();
        let result = NeuralNetwork::load(&path);

        assert!(result.is_err());

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn given_corrupted_file_when_load_then_returns_err() {
        let path = PathBuf::from("test_corrupted.bin");

        // Write some garbage data
        let garbage = vec![1, 2, 3, 4, 5];
        fs::write(&path, garbage).unwrap();

        let result = NeuralNetwork::load(&path);
        assert!(result.is_err());

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn given_saved_network_when_load_and_use_then_produces_same_output() {
        let nn = NeuralNetwork::new(784, 64, 10);
        let path = PathBuf::from("test_load_output.bin");

        // Create test input
        let input = ndarray::Array1::from_vec(vec![0.5; 784]);

        // Get output from original
        let (_, _, _, original_output) = nn.feed_forward(input.clone()).unwrap();

        // Save and load
        nn.save(&path).unwrap();
        let loaded = NeuralNetwork::load(&path).unwrap();

        // Get output from loaded
        let (_, _, _, loaded_output) = loaded.feed_forward(input).unwrap();

        // Outputs should match
        for i in 0..original_output.len() {
            assert_eq!(original_output[i], loaded_output[i]);
        }

        // Cleanup
        let _ = fs::remove_file(path);
    }
}
