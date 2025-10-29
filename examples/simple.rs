use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};
use ndarray::{Array, Array4};

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

    let v = output.unwrap();
    for (_out, v) in v.outputs {
        let _output: Array<f32, _> = v.extract_to_tensor();
    }

    // Use output as needed
}
