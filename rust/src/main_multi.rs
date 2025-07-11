// Script used for multi-threading when dehazing images

use image::{ImageBuffer, Rgb};
use ndarray::{Array4, CowArray};
use ort::{
    execution_providers::CoreMLExecutionProviderOptions, ExecutionProvider, Session,
    SessionBuilder, Value,
};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    thread,
    time::{Duration, Instant},
};
use crossbeam_channel::unbounded;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let coreml_options = CoreMLExecutionProviderOptions {
        use_cpu_only: false,
        enable_on_subgraph: false,
        only_enable_device_with_ane: false,
    };

    let total_start_time = Instant::now();

    let environment = Arc::new(ort::Environment::builder().with_name("funiegan").build()?);
    let session = Arc::new(
        SessionBuilder::new(&environment)?
            .with_execution_providers([ExecutionProvider::CoreML(coreml_options)])?
            .with_model_from_file("../models/funiegan_model_dynamic.onnx")?,
    );

    let output_dir = "../data/output_threaded/";
    std::fs::create_dir_all(output_dir)?;

    let (tx1, rx1) = unbounded(); // Frame acquisition → Inference
    let (tx2, rx2) = unbounded(); // Inference → Output

    // Frame acquisition thread
    let acquisition_thread = {
        let tx1 = tx1.clone();
        thread::spawn(move || {
            let input_dir = PathBuf::from("../data/test/full/");
            if let Ok(entries) = std::fs::read_dir(&input_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(ext) = path.extension() {
                        if matches!(ext.to_str(), Some("jpg") | Some("jpeg") | Some("png")) {
                            if tx1.send(path).is_err() {
                                break;
                            }
                            thread::sleep(Duration::from_millis(125));
                        }
                    }
                }
            }
        })
    };

    // Start multiple inference workers
    let num_workers = 4;
    let mut worker_handles = vec![];
    for _ in 0..num_workers {
        let session = Arc::clone(&session);
        let rx1 = rx1.clone();
        let tx2 = tx2.clone();
        let output_dir = output_dir.to_string();

        let handle = thread::spawn(move || {
            for frame_path in rx1.iter() {
                let start = Instant::now();
                match process_image_resize(&session, &frame_path, &output_dir) {
                    Ok(output_path) => {
                        let elapsed = start.elapsed();
                        println!(
                            "[Worker] Inference on {:?} took {:.2?}",
                            frame_path.file_name().unwrap(),
                            elapsed
                        );
                        if tx2.send(output_path).is_err() {
                            break;
                        }
                    }
                    Err(e) => eprintln!("Failed to process {:?}: {}", frame_path, e),
                }
            }
        });
        worker_handles.push(handle);
    }

    // Drop tx1 after acquisition is done
    acquisition_thread.join().unwrap();
    drop(tx1); // Ensures workers exit when no more frames

    // Join all worker threads
    for handle in worker_handles {
        handle.join().unwrap();
    }
    drop(tx2); // Ensures output thread exits

    // Output thread
    let output_thread = thread::spawn(move || {
        let mut count = 0;
        for processed_path in rx2.iter() {
            println!("Saved enhanced image to: {:?}", processed_path);
            count += 1;
        }
        println!("✅ Total images processed: {}", count);
    });

    output_thread.join().unwrap();

    let total_duration = total_start_time.elapsed();
    println!("⏱️ Total elapsed time: {:.2?}", total_duration);

    let input_dir = PathBuf::from("../data/test/full/");
    let num_images = std::fs::read_dir(&input_dir)?
        .filter_map(Result::ok)
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext_str| matches!(ext_str, "jpg" | "jpeg" | "png"))
                .unwrap_or(false)
        })
        .count();

    let fps = num_images as f64 / total_duration.as_secs_f64();
    println!("📸 Processed {} images", num_images);
    println!("🚀 Average FPS: {:.2}", fps);

    Ok(())
}

fn process_image_resize(
    session: &Session,
    input_path: &Path,
    output_dir: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    use image::imageops::FilterType;

    let img_dyn = image::open(input_path)?;
    let img_rgb = image::imageops::resize(&img_dyn.to_rgb8(), 1280, 800, FilterType::Triangle);

    let (padded_img, orig_w, orig_h) = pad_to_multiple(&img_rgb, 32);
    let input_tensor = preprocess_image(&padded_img);
    let input_array: CowArray<f32, _> = input_tensor.into();
    let input_array = input_array.into_dyn();

    let input_value = Value::from_array(session.allocator(), &input_array)?;
    let outputs = session.run(vec![input_value])?;

    let output_tensor = outputs[0].try_extract::<f32>()?;
    let output_array = output_tensor
        .view()
        .to_owned()
        .into_dimensionality::<ndarray::Ix4>()?;

    let output_img = postprocess_image(output_array.view());
    let cropped_img = image::imageops::crop_imm(&output_img, 0, 0, orig_w, orig_h).to_image();

    let output_path = format!(
        "{}/{}",
        output_dir,
        input_path.file_name().unwrap().to_str().unwrap()
    );
    cropped_img.save(&output_path)?;
    Ok(output_path)
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
            let src_x = if x < width { x } else { 2 * width - x - 1 };
            let src_y = if y < height { y } else { 2 * height - y - 1 };
            let pixel = img.get_pixel(src_x, src_y);
            padded.put_pixel(x, y, *pixel);
        }
    }

    (padded, width, height)
}

fn preprocess_image(img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Array4<f32> {
    let (width, height) = img.dimensions();
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
