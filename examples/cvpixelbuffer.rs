use coreml_rs::{ CoreMLModelOptions, CoreMLModelWithState };

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "path/to/your/model.mlpackage";
    let mut model_options = CoreMLModelOptions::default();

    // Load the model
    let mut model = CoreMLModelWithState::new(model_path, model_options).load()?;

    let width = 640;
    let height = 640;
    let bgra_data = vec![0u8; width * height * 4]; // 4 bytes per pixel (BGRA)

    model.add_input_cvpixelbuffer("input", width, height, bgra_data)?;

    let outputs = model.predict()?;

    println!("Prediction complete!");
    println!("Available outputs: {:?}", outputs.outputs.keys().collect::<Vec<_>>());

    Ok(())
}
