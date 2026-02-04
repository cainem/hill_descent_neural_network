// External crate declarations
extern crate bincode;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
extern crate serde_derive;

// Module declarations
pub mod data_loader;
pub mod neural_network;
pub mod sigmoid;

// Re-export the main struct for convenience
pub use data_loader::load_mnist_data;
pub use neural_network::NeuralNetwork;
