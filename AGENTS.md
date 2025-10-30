# Neural Network Scratch Project - Agent Instructions

## Project Context
This is a Rust-based neural network implementation from scratch for Windows, designed to work with the MNIST dataset of handwritten digits. The neural network achieves approximately 96.8% accuracy on the test set using a simple 3-layer architecture (784-64-10) with sigmoid activation functions and gradient descent training.

The project includes:
- Core neural network implementation in `src/neural_network.rs`
- Interactive CLI application in `src/main.rs`
- MNIST dataset in `dataset/` directory
- Pre-trained model file (`model.bin` or `test-model`)

## Development Standards

### Code Organization
- **Simple solutions first** - Always prefer straightforward implementations
- **No code duplication** - Check existing codebase before adding new functionality
- **Clean structure** - Keep codebase organized and maintainable
- **Avoid unnecessary refactoring** - Stick to requested tasks only

### File Structure Rules
- **Structs in own files** - `src/module/struct_name.rs` where filename matches struct name
- **Size limits** - Functions >40 lines should be split; files >40-100 lines should be refactored
- **Private fields only** - Use getters/setters instead of public struct fields
- **No one-off scripts** - Avoid temporary or single-use files

### Testing Requirements
- **Full unit test coverage** - Every function needs comprehensive tests that test each branch and condition
- **Test naming** - Use `given_xxx_when_yyy_then_zzz` pattern
- **Minimal mocking** - Only mock PRNG and file I/O where necessary
- **Performance** - Tests must run efficiently while maintaining coverage

### Change Management
- **Conservative changes** - Only implement requested features
- **Existing patterns first** - Exhaust current implementation before new patterns
- **Remove old code** - If introducing new patterns, clean up duplicates
- **Check comments for accuracy** - Check that changes haven't left inaccurate comments
- **Add new comments** - For functions added or amended make sure the function is accompanied by appropriate comments
- **Check the README.md** - Check that changes have not invalidated the README.md and update as necessary
- **Check tests** - Make sure the tests covering changed code still cover all conditions and branches

### Environment Notes
- **Windows development** - Code targets Windows environment
- **Never overwrite .env** - Always ask before modifying environment files
- **MNIST dataset** - Dataset files in `dataset/` should not be modified or committed if large

## Build & Test Commands
- Format: `cargo fmt`
- Build: `cargo build`
- Build release: `cargo build --release`
- Test: `cargo test`
- Run: `cargo run` (debug) or `cargo run --release` (optimized)
- Lint: `cargo clippy`
- Lint tests: `cargo clippy --tests`

## Before Committing Code to the Repository
- Ensure `cargo fmt` has been run
- Ensure all tests pass: `cargo test`
- Ensure linting passes with zero warnings: `cargo clippy` and `cargo clippy --tests`. **All clippy warnings must be fixed** - either by correcting the code or by adding explicit `#[allow(...)]` annotations with justification comments for intentional exceptions.
- Ensure code adheres to all outlined standards and guidelines
- Ensure commit messages are clear and descriptive
- Ensure all changed functions and structs have appropriate tests with full coverage
- Ensure that all changed functions and structs have appropriate documentation comments

## Key Files and Directories
- `src/main.rs` - CLI application for loading data, training, and testing
- `src/neural_network.rs` - Core neural network implementation (NeuralNetwork struct)
- `dataset/` - MNIST dataset files (training and test images/labels)
  - `train-images.idx3-ubyte` - 60,000 training images
  - `train-labels.idx1-ubyte` - 60,000 training labels
  - `t10k-images.idx3-ubyte` - 10,000 test images
  - `t10k-labels.idx1-ubyte` - 10,000 test labels
- `target/` - Build artifacts (not committed to repository)
- `model.bin` / `test-model` - Serialized trained models

## Neural Network Architecture Notes
- **Input layer**: 784 neurons (28×28 pixel images flattened)
- **Hidden layer**: 64 neurons with sigmoid activation
- **Output layer**: 10 neurons (digits 0-9) with sigmoid activation
- **Training**: Gradient descent with backpropagation
- **Loss function**: Binary cross-entropy
- **Learning rate**: 0.01 (hardcoded)
- **Expected accuracy**: ~96.8% on test set

## Performance Considerations
- **Debug mode is slow** - Always use `cargo build --release` for training
- **Training epochs** - More epochs improve accuracy but take longer
- **Test set separation** - Model is trained on 60k images, tested on separate 10k images
