use coreml_rs::{mlbatchmodel::CoreMLBatchModelWithState, ComputePlatform, CoreMLModelOptions};
use ndarray::{Array, Array4};

pub fn main() {
    let file = std::fs::read("./demo/model_3.mlmodel").unwrap();

    let mut model_options = CoreMLModelOptions::default();
    model_options.compute_platform = ComputePlatform::CpuAndANE;

    let mut model = CoreMLBatchModelWithState::from_buf(file, model_options);

    let mut input = Array4::<f32>::zeros((1, 3, 512, 512));
    input.fill(1.0f32);

    // Add multiple inputs for batch processing
    for i in 0..10 {
        let _ = model.add_input("image", input.clone().into_dyn(), i);
    }

    let output = model.predict().unwrap();
    for output in output.outputs {
        for (_out, v) in output {
            let _output: Array<f32, _> = v.extract_to_tensor();
        }
    }
}
