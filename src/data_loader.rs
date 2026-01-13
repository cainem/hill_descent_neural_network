use ndarray::Array2;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

fn read_u32_from_file(file: &mut File) -> Result<u32, io::Error> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

pub fn load_mnist_data<P: AsRef<Path>>(
    images_path: P,
    labels_path: P,
) -> Result<(Array2<f64>, Array2<f64>), io::Error> {
    let mut image_file = File::open(images_path).expect("Failed to open file");
    let mut label_file = File::open(labels_path).expect("Failed to open file");

    // Read header information
    let _magic_images =
        read_u32_from_file(&mut image_file).expect("Failed to read header information");
    let num_images =
        read_u32_from_file(&mut image_file).expect("Failed to read header information");
    let num_rows = read_u32_from_file(&mut image_file).expect("Failed to read header information");
    let num_cols = read_u32_from_file(&mut image_file).expect("Failed to read header information");

    let _magic_labels =
        read_u32_from_file(&mut label_file).expect("Failed to read header information");
    let num_labels =
        read_u32_from_file(&mut label_file).expect("Failed to read header information");

    assert_eq!(
        num_images, num_labels,
        "Number of images and labels do not match"
    );

    let mut image_data = vec![0u8; (num_images * num_rows * num_cols) as usize];
    image_file.read_exact(&mut image_data)?;

    let images = Array2::from_shape_vec(
        (num_images as usize, (num_rows * num_cols) as usize),
        image_data.into_iter().map(|x| x as f64 / 255.0).collect(),
    )
    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    let mut label_data = vec![0u8; num_labels as usize];
    label_file.read_exact(&mut label_data)?;

    let mut labels = Array2::zeros((num_labels as usize, 10));
    for (i, &label) in label_data.iter().enumerate() {
        labels[[i, label as usize]] = 1.0;
    }

    Ok((images, labels))
}
