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
        let data = bincode::serialize(&self)
            .map_err(|e| io::Error::other(format!("Serialization error: {:?}", e)))?;

        // Create the file and write the serialized data
        let mut file = File::create(path)?;
        file.write_all(&data)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn given_network_when_save_then_creates_file() {
        let nn = NeuralNetwork::new(10, 5, 2);
        let path = PathBuf::from("test_save.bin");

        let result = nn.save(&path);
        assert!(result.is_ok());

        // Verify file exists
        assert!(path.exists());

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn given_network_when_save_then_file_not_empty() {
        let nn = NeuralNetwork::new(10, 5, 2);
        let path = PathBuf::from("test_save_size.bin");

        nn.save(&path).unwrap();

        let metadata = fs::metadata(&path).unwrap();
        assert!(metadata.len() > 0);

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn given_invalid_path_when_save_then_returns_err() {
        let nn = NeuralNetwork::new(10, 5, 2);
        // Invalid path with illegal characters
        let path = PathBuf::from("invalid:\0path.bin");

        let result = nn.save(&path);
        assert!(result.is_err());
    }

    #[test]
    fn given_different_networks_when_save_then_different_file_sizes() {
        let nn1 = NeuralNetwork::new(10, 5, 2);
        let nn2 = NeuralNetwork::new(100, 50, 20);

        let path1 = PathBuf::from("test_small.bin");
        let path2 = PathBuf::from("test_large.bin");

        nn1.save(&path1).unwrap();
        nn2.save(&path2).unwrap();

        let size1 = fs::metadata(&path1).unwrap().len();
        let size2 = fs::metadata(&path2).unwrap().len();

        assert!(size2 > size1, "Larger network should create larger file");

        // Cleanup
        let _ = fs::remove_file(path1);
        let _ = fs::remove_file(path2);
    }
}
