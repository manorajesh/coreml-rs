## Core-ML Rust Bindings (Work in Progress)

`coreml-rs` is an experimental Rust library aimed at providing Rust bindings for Apple's Core ML framework.
Core ML is Apple's machine learning framework designed to integrate machine learning models into iOS, macOS, watchOS, and tvOS applications.

### NOTES:

`libswift_Concurrency.dylib` is required for the builds to work, put the dylib next to the built binary to run the process.

## Status

This project is currently work in progress.
The primary goal is to enable Rust developers to utilize Core ML models within their applications, leveraging Rust's performance and safety features for the rest of the ML infrastructure.

## Roadmap

- Cleanup & fix bugs with the types and allow more input formats.
- Build zerocopy types for more efficiently passing inputs and outputs.
- Provide more configuration options for models.

## Features

- **Model Loading**: Load Core ML models into Rust applications.
- **Inference**: Perform inference using loaded models.
- **Data Handling**: Manage input and output data for model inference.

## Installation

To include `coreml-rs` in your project, add the following to your `Cargo.toml` dependencies:
```toml
[dependencies]
coreml-rs = { version = "0.4", git = "https://github.com/swarnimarun/coreml-rs" }
```

## Usage

### Simple Inference

Load a Core ML model from a `.mlmodel` file and perform a single inference:

```rust
use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};
use ndarray::Array4;

pub fn main() {
    let file = std::fs::read("./demo/model_3.mlmodel").unwrap();

    let mut model_options = CoreMLModelOptions::default();
    model_options.compute_platform = ComputePlatform::CpuAndANE;

    let mut model = CoreMLModelWithState::from_buf(file, model_options);

    let mut input = Array4::<f32>::zeros((1, 3, 512, 512));
    input.fill(1.0f32);

    let Ok(_) = model.add_input("image", input.into_dyn()) else {
        panic!("failed to add input feature, `image` to the model");
    };

    let output = model.predict();

    let v = output.unwrap().bytesFrom("output_1".to_string());
    let output: Array4<f32> = Array4::from_shape_vec([1, 3, 2048, 2048], v).unwrap();

    // Use output as needed
}
```

### Batch Inference

Perform batch inference by adding multiple inputs:

```rust
use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};
use ndarray::Array4;

pub fn main() {
    let file = std::fs::read("./demo/model_3.mlmodel").unwrap();

    let mut model_options = CoreMLModelOptions::default();
    model_options.compute_platform = ComputePlatform::CpuAndANE;

    let mut model = CoreMLModelWithState::from_buf(file, model_options);

    let mut input = Array4::<f32>::zeros((1, 3, 512, 512));
    input.fill(1.0f32);

    // Add multiple inputs for batch processing
    for i in 0..10 {
        let _ = model.add_input("image", input.clone().into_dyn(), i);
    }

    let output = model.predict().unwrap();

    // Process batch outputs
    for i in 0..10 {
        let v = output.bytesFrom(&format!("output_1_{}", i));
        let batch_output: Array4<f32> = Array4::from_shape_vec([1, 3, 2048, 2048], v).unwrap();
        // Use batch_output as needed
    }
}
```

### Loading from Zip Archive

Load a model from a zip archive containing an `.mlpackage`:

```rust
use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};
use ndarray::Array4;
use std::path::PathBuf;

fn unzip_to_path_from_hash(buf: &[u8]) -> Option<PathBuf> {
    fn get_cache_filename(model_buffer: &[u8]) -> String {
        use sha2::Digest;
        let mut hasher = sha2::Sha256::new();
        hasher.update(model_buffer);
        let hash = hasher.finalize();
        format!("{:x}.mlpackage", hash)
    }
    let name = get_cache_filename(buf);

    let path = PathBuf::from("/tmp/coreml-aftershoot/");
    let path = path.join(name);
    _ = std::fs::remove_dir_all(&path);
    _ = std::fs::remove_file(&path);

    let mut res = zip::ZipArchive::new(std::io::Cursor::new(buf)).ok()?;
    res.extract(&path).ok()?;

    let m = path.join("model.mlpackage");
    if m.exists() {
        Some(m)
    } else {
        None
    }
}

pub fn main() {
    let buf = std::fs::read("./demo/model_2.zip").unwrap();

    let model_path = unzip_to_path_from_hash(&buf).unwrap();

    let mut model_options = CoreMLModelOptions::default();
    model_options.compute_platform = ComputePlatform::CpuAndANE;

    let mut model = CoreMLModelWithState::new(model_path, model_options).load().unwrap();

    let mut input = Array4::<f32>::zeros((1, 3, 512, 512));
    input.fill(1.0f32);

    let Ok(_) = model.add_input("image", input.into_dyn()) else {
        panic!("failed to add input feature, `image` to the model");
    };

    let output = model.predict();

    let v = output.unwrap().bytesFrom("output_1".to_string());
    let output: Array4<f32> = Array4::from_shape_vec([1, 3, 2048, 2048], v).unwrap();

    // Use output as needed
}
```

### Memory Management

Unload and reload models to manage memory usage:

```rust
use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};
use ndarray::Array4;

pub fn main() {
    let file = std::fs::read("./demo/model_3.mlmodel").unwrap();

    let mut model_options = CoreMLModelOptions::default();
    model_options.compute_platform = ComputePlatform::CpuAndANE;

    let mut model = CoreMLModelWithState::from_buf(file, model_options);

    let mut input = Array4::<f32>::zeros((1, 3, 512, 512));
    input.fill(1.0f32);

    let Ok(_) = model.add_input("image", input.clone().into_dyn()) else {
        panic!("failed to add input feature, `image` to the model");
    };

    let output = model.predict();

    let v = output.unwrap().bytesFrom("output_1".to_string());
    let output: Array4<f32> = Array4::from_shape_vec([1, 3, 2048, 2048], v).unwrap();

    // Unload model to free memory
    let unloaded_model = model.unload().unwrap();

    // Later, reload the model
    let mut model = unloaded_model.load().unwrap();

    // Add input again and predict
    let Ok(_) = model.add_input("image", input.into_dyn()) else {
        panic!("failed to add input feature, `image` to the model");
    };

    let output = model.predict();

    let v = output.unwrap().bytesFrom("output_1".to_string());
    let output: Array4<f32> = Array4::from_shape_vec([1, 3, 2048, 2048], v).unwrap();

    // Use output as needed
}
```

**Note**: These examples assume specific model inputs/outputs. Adjust based on your model's specifications.

## Contributing

Contributions are welcome!
If you have experience with Core ML and Rust, consider helping to advance this project.
