use super::NeuralNetwork;
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;

impl NeuralNetwork {
    /// Saves the trained neural network to a binary file.
    ///
    /// The network is serialized to a compact binary format using bincode,
    /// which includes all weights, biases, and network dimensions. This allows
    /// the trained model to be saved and loaded later without retraining.
    ///
    /// # Arguments
    /// * `path` - Path where the model should be saved
    ///
    /// # Returns
    /// * `Ok(())` - Successfully saved
    /// * `Err(io::Error)` - If serialization or file writing fails
    ///
    /// # Errors
    /// - Serialization errors (should be rare with correct NeuralNetwork structure)
    /// - File I/O errors (disk full, permission denied, invalid path, etc.)
    ///
    /// # Example
    /// ```
    /// nn.save(&PathBuf::from("model.bin"))?;
    /// ```
    pub fn save(&self, path: &PathBuf) -> io::Result<()> {
        // Serialize the entire NeuralNetwork struct to binary format
        let data = bincode::serialize(&self).map_err(|e| {
            io::Error::other(format!("Serialization error: {:?}", e))
        })?;

        // Create the file and write the serialized data
        let mut file = File::create(path)?;
        file.write_all(&data)?;

        Ok(())
    }
}
