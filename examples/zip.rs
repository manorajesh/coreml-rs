use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};
use ndarray::{Array, Array4};
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

    let path = PathBuf::from("/tmp/coreml-cache/");
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

    let mut model = CoreMLModelWithState::new(model_path, model_options)
        .load()
        .unwrap();

    let mut input = Array4::<f32>::zeros((1, 3, 512, 512));
    input.fill(1.0f32);

    let Ok(_) = model.add_input("image", input.into_dyn()) else {
        panic!("failed to add input feature, `image` to the model");
    };

    let output = model.predict().unwrap();

    for (_out, v) in output.outputs {
        let _output: Array<f32, _> = v.extract_to_tensor();
    }
}
