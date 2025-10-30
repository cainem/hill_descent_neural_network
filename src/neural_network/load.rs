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
        let network: NeuralNetwork = bincode::deserialize(&buffer).map_err(|e| {
            io::Error::other(format!("Deserialization error: {:?}", e))
        })?;

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
