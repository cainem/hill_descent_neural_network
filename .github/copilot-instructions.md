# Neural Network Scratch - GitHub Copilot Instructions

> **Read `AGENTS.md` for comprehensive project guidance.**

## Project Architecture
Rust neural network implementation from scratch for MNIST digit classification:
- `src/neural_network.rs` - Core neural network implementation (3-layer feedforward)
- `src/main.rs` - Interactive CLI for training, testing, and model persistence
- `dataset/` - MNIST dataset files (60k training, 10k test images)
- Developed on a Windows platform

## Essential Initialization Pattern
```rust
use neural_network::NeuralNetwork;
use ndarray::Array2;

// Create new network
let nn = NeuralNetwork::new(784, 64, 10);  // input, hidden, output sizes

// Load existing model
let nn = NeuralNetwork::load(&PathBuf::from("model.bin"))?;

// Train network
nn.train(&x_train, &y_train, epochs, 0.01);

// Evaluate accuracy
let accuracy = nn.accuracy(&x_test, &y_test);
```

## Critical Code Organization
- **File naming**: `src/module/struct_name.rs` (filename = struct name)
- **Fields**: Private only - use getters/setters, never public fields
- **Function size**: Split at >40 lines into separate files
- **Tests**: `given_xxx_when_yyy_then_zzz` naming convention
- **Network flow**: `Input -> Hidden (sigmoid) -> Output (sigmoid)`

## Development Workflow
```powershell
cargo fmt && cargo test && cargo clippy   # Pre-commit checks
cargo run --release                       # Run CLI (release mode for performance)
cargo test                                # Run all tests
cargo clippy --tests                      # Lint tests
```

## Key Domain Types
- `NeuralNetwork` - Main struct with weights (W1, W2), biases (b1, b2)
- `Array2<f64>` - 2D arrays for batch data (ndarray crate)
- `Array1<f64>` - 1D arrays for single examples and activations
- **Input**: 784 features (28×28 flattened grayscale images, normalized 0.0-1.0)
- **Labels**: One-hot encoded vectors (10 classes for digits 0-9)

## Integration Points
- **MNIST Format**: IDX file format with big-endian 32-bit integers
- **Serialization**: Uses `bincode` for model persistence (save/load)
- **Data Loading**: `load_mnist_data()` reads IDX files and normalizes
- **Training Loop**: Iterates through all examples each epoch, prints loss

## Testing Requirements  
- Full branch/condition coverage per function (not transitive)
- Test boundary conditions, especially with floating point operations
- Minimal mocking (only PRNG via `rand` and file I/O where necessary)
- Test mathematical operations (forward pass, backpropagation, loss)
- Verify activation functions (sigmoid, derivatives)

## Key Behaviors
- **Always read `AGENTS.md`** for comprehensive context at conversation start
- **Ask before changing** - clarify requirements vs assumptions
- **Ask before changing** - clarify any apparent contradictions in requests and/or the combination of request and the AGENTS.md file
- **Check README.md** when behavior changes to ensure documentation accuracy
- **Use existing patterns** - examine similar functions before creating new approaches
- **Performance matters** - Always recommend `--release` mode for training (debug is very slow)

## Neural Network Architecture
- **Layer 1**: Input (784) → Hidden (64) via W1, b1, sigmoid activation
- **Layer 2**: Hidden (64) → Output (10) via W2, b2, sigmoid activation
- **Loss**: Binary cross-entropy with epsilon clipping (1e-15)
- **Training**: Stochastic gradient descent with backpropagation
- **Learning rate**: 0.01 (hardcoded)
- **Expected accuracy**: ~96.8% on MNIST test set

## Important Constraints
- **Dataset integrity**: Never modify files in `dataset/` directory
- **Model compatibility**: Input=784, output=10 enforced in `load()`
- **Batch processing**: Training data as Array2 (rows=examples, cols=features)
- **Normalization**: Pixel values must be 0.0-1.0 (divide by 255.0)
