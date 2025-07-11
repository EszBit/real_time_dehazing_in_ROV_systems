// Original script with added padding

use image::{ImageBuffer, Rgb};
use ndarray::{self, Array4, CowArray};
use ort::{ExecutionProvider, Session, SessionBuilder, Value};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize ONNX Runtime
    let environment = Arc::new(ort::Environment::builder().with_name("funiegan").build()?);

    // Load the model
    let session = SessionBuilder::new(&environment)?
        .with_execution_providers([ExecutionProvider::CPU(Default::default())])?
        .with_model_from_file("../models/funiegan_model.onnx")?;

    // Process images
    let input_dir = "../data/test/full/";
    let output_dir = "../data/output_rust/";
    std::fs::create_dir_all(output_dir)?;

    // Start time
    let start_time = std::time::Instant::now();

    for entry in std::fs::read_dir(input_dir)? {
        let path = entry?.path();
        if let Some(ext) = path.extension() {
            if matches!(ext.to_str(), Some("jpg") | Some("png") | Some("jpeg")) {
                process_image_resize(&session, &path, output_dir)?;
            }
        }
    }

    // Print duration
    let duration = start_time.elapsed();
    println!("Processing 20 images in: {:?}", duration);
    println!("FPS: {:.2}", 20.0 / duration.as_secs_f32());

    Ok(())
}

fn process_image(
    session: &Session,
    input_path: &std::path::Path,
    output_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load and preprocess image
    let img_dyn = image::open(input_path)?;
    let img_rgb = img_dyn.to_rgb8();
    // Padding
    let (padded_img, orig_w, orig_h) = pad_to_multiple(&img_rgb, 32);

    // Convert to normalized tensor format
    let input_tensor = preprocess_image(&padded_img);

    // Convert to CowArray with dynamic dimensions
    let input_array: CowArray<f32, _> = input_tensor.into();
    let input_array = input_array.into_dyn(); // Convert to dynamic dimensions

    // Create ONNX input
    let input_value = Value::from_array(session.allocator(), &input_array)?;

    // Run inference
    let outputs = session.run(vec![input_value])?;

    // Post-process and save
    let output_tensor = outputs[0].try_extract::<f32>()?;
    let output_view = output_tensor
        .view()
        .clone()
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|e| format!("Failed to convert to 4D array: {}", e))?;
    let output_img = postprocess_image(output_view);
    let cropped_img = image::imageops::crop_imm(&output_img, 0, 0, orig_w, orig_h).to_image();

    let output_path = format!(
        "{}/{}",
        output_dir,
        input_path.file_name().unwrap().to_str().unwrap()
    );
    //output_img.save(output_path)?;
    cropped_img.save(output_path)?;

    println!("Processed: {:?}", input_path);
    Ok(())
}

fn process_image_resize(
    session: &Session,
    input_path: &std::path::Path,
    output_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use image::imageops::FilterType;

    // Load and resize image
    let img_dyn = image::open(input_path)?;
    let img_rgb = image::imageops::resize(
        &img_dyn.to_rgb8(),
        640,
        640,
        FilterType::Triangle,
    );

    // Padding
    let (padded_img, orig_w, orig_h) = pad_to_multiple(&img_rgb, 32);

    // Convert to normalized tensor format
    let input_tensor = preprocess_image(&padded_img);

    // Convert to CowArray with dynamic dimensions
    let input_array: CowArray<f32, _> = input_tensor.into();
    let input_array = input_array.into_dyn();

    // Run inference
    let input_value = Value::from_array(session.allocator(), &input_array)?;
    let outputs = session.run(vec![input_value])?;

    let output_tensor = outputs[0].try_extract::<f32>()?;
    let output_view = output_tensor
        .view()
        .clone()
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|e| format!("Failed to convert to 4D array: {}", e))?;

    let output_img = postprocess_image(output_view);
    let cropped_img = image::imageops::crop_imm(&output_img, 0, 0, orig_w, orig_h).to_image();

    let output_path = format!(
        "{}/{}",
        output_dir,
        input_path.file_name().unwrap().to_str().unwrap()
    );
    cropped_img.save(output_path)?;

    println!("Processed: {:?}", input_path);
    Ok(())
}


fn preprocess_image(img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Array4<f32> {
    let (width, height) = img.dimensions(); // dynamic size
    let mut tensor = Array4::<f32>::zeros((1, 3, height as usize, width as usize));
    println!("Creating tensor of shape: [1, 3, {}, {}]", height, width);

    for (y, row) in img.rows().enumerate() {
        for (x, pixel) in row.enumerate() {
            tensor[[0, 0, y, x]] = (pixel[0] as f32 / 255.0 - 0.5) / 0.5;
            tensor[[0, 1, y, x]] = (pixel[1] as f32 / 255.0 - 0.5) / 0.5;
            tensor[[0, 2, y, x]] = (pixel[2] as f32 / 255.0 - 0.5) / 0.5;
        }
    }
    
    tensor
}


fn postprocess_image(tensor: ndarray::ArrayView4<f32>) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (_, _, height, width) = tensor.dim();

    let mut img = ImageBuffer::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let r = ((tensor[[0, 0, y, x]] * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            let g = ((tensor[[0, 1, y, x]] * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            let b = ((tensor[[0, 2, y, x]] * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    img
}


fn pad_to_multiple(
    img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    multiple: u32,
) -> (ImageBuffer<Rgb<u8>, Vec<u8>>, u32, u32) {
    let (width, height) = img.dimensions();
    let pad_w = (multiple - width % multiple) % multiple;
    let pad_h = (multiple - height % multiple) % multiple;

    let new_w = width + pad_w;
    let new_h = height + pad_h;

    let mut padded = ImageBuffer::new(new_w, new_h);

    for y in 0..new_h {
        for x in 0..new_w {
            let src_x = if x < width {
                x
            } else {
                2 * width - x - 1
            };
            let src_y = if y < height {
                y
            } else {
                2 * height - y - 1
            };

            let pixel = img.get_pixel(src_x, src_y);
            padded.put_pixel(x, y, *pixel);
        }
    }

    (padded, width, height)
}

