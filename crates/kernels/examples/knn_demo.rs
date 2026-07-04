use kernels::{knn_predict_l2_sq_u32, nearest_k_l2_sq_f32};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let query = [1.0, 1.0];
    let samples: [&[f32]; 4] = [&[1.0, 1.0], &[1.2, 1.1], &[8.0, 8.0], &[7.5, 8.0]];
    let labels = [10, 10, 20, 20];

    let neighbors = nearest_k_l2_sq_f32(&query, &samples, 3)?;
    let prediction = knn_predict_l2_sq_u32(&query, &samples, &labels, 3)?;

    println!("neighbors={neighbors:?}");
    println!("prediction={prediction}");

    Ok(())
}
