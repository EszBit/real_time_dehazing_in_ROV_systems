// Script used for enabling FastAPI onnx model deployment through Python

use std::fs;
use std::path::Path;
use reqwest::blocking::{Client, multipart};
use std::io::Read;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Process images
    let input_dir = "../data/test/full/";
    let output_dir = "../data/output_rust/";
    fs::create_dir_all(output_dir)?;

    // Start timer
    let start_time = std::time::Instant::now();

    for entry in fs::read_dir(input_dir)? {
        let path = entry?.path();
        if let Some(ext) = path.extension() {
            if matches!(ext.to_str(), Some("jpg") | Some("png") | Some("jpeg")) {
                process_image_http(&path, output_dir)?;
            }
        }
    }

    // Timing summary
    let duration = start_time.elapsed();
    println!("Processing 20 images in: {:?}", duration);
    println!("FPS: {:.2}", 20.0 / duration.as_secs_f32());

    Ok(())
}

fn process_image_http(
    input_path: &Path,
    output_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Read image into memory
    let mut buf = Vec::new();
    fs::File::open(input_path)?.read_to_end(&mut buf)?;
    let part = multipart::Part::bytes(buf).file_name("image.png");
    let form = multipart::Form::new().part("file", part);

    // Send to FastAPI server
    let client = Client::new();
    let resp = client.post("http://127.0.0.1:8000/predict/?width=960&height=600")
        .multipart(form)
        .send()?;

    if !resp.status().is_success() {
        return Err(format!("HTTP Error: {}", resp.status()).into());
    }

    // Save received image
    let result_bytes = resp.bytes()?;
    let output_path = format!(
        "{}/{}",
        output_dir,
        input_path.file_name().unwrap().to_str().unwrap()
    );
    fs::write(output_path, &result_bytes)?;

    println!("Processed via FastAPI: {:?}", input_path);
    Ok(())
}
