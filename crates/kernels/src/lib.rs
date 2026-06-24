#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    // Preconditions: dot product only makes sense for equal-length vectors.
    assert_eq!(a.len(), b.len(), "Vectors must have equal length");

    let mut sum = 0.0f32;

    // Simple baseline loop / need optimization later (SIMD/ASM).
    for (&x, &y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }

    sum
}

/// Computes squared Euclidean distance between two equal-length vectors.
///
/// l2_sq(a, b) = sum_i (a[i] - b[i])^2
///
/// We intentionally do not take sqrt here.
/// For nearest-neighbor search and k-means assignment, comparing squared
/// distances is enough because sqrt preserves ordering.
#[inline]
pub fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have equal length");

    let mut sum = 0.0f32;

    for (&x, &y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        sum += diff * diff;
    }

    sum
}

/// Computes squared L2 norm of a vector.
///
/// l2_norm_sq(a) = sum_i a[i]^2
///
/// This is the squared length of the vector.
/// The actual L2 norm would be sqrt(l2_norm_sq(a)).
#[inline]
pub fn l2_norm_sq_f32(a: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    for &x in a {
        sum += x * x;
    }

    sum
}

/// Computes cosine similarity between two equal-length vectors.
///
/// cosine(a, b) = dot(a, b) / (||a|| * ||b||)
///
/// Returns:
/// - Some(value) if both vectors have non-zero norm
/// - None if at least one vector is zero-length in the geometric sense
#[inline]
pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> Option<f32> {
    assert_eq!(a.len(), b.len(), "Vectors must have equal length");

    let norm_a_sq = l2_norm_sq_f32(a);
    let norm_b_sq = l2_norm_sq_f32(b);

    if norm_a_sq == 0.0 || norm_b_sq == 0.0 {
        return None;
    }

    let denominator = norm_a_sq.sqrt() * norm_b_sq.sqrt();
    let similarity = dot_f32(a, b) / denominator;

    Some(similarity)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn assert_some_approx_eq(got: Option<f32>, expected: f32, eps: f32) {
        let value = got.expect("expected Some(value), got None");
        assert!(
            approx_eq(value, expected, eps),
            "expected {expected}, got {value}"
        );
    }

    #[test]
    fn dot_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 32
        let got = dot_f32(&a, &b);
        assert!(approx_eq(got, 32.0, 1e-6));
    }

    #[test]
    fn dot_works_with_vec() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = vec![4.0, 5.0, 6.0];
        let got = dot_f32(&a, &b);
        assert!(approx_eq(got, 32.0, 1e-6));
    }

    #[test]
    #[should_panic]
    fn dot_panics_on_different_lengths() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0, 3.0];
        let _ = dot_f32(&a, &b);
    }

    #[test]
    fn dot_same_vector_is_sum_of_squares() {
        let a = [2.0, -3.0, 4.0];
        let got = dot_f32(&a, &a);
        // 2^2 + (-3)^2 + 4^2 = 4 + 9 + 16 = 29
        assert!(approx_eq(got, 29.0, 1e-6));
    }

    #[test]
    fn l2_sq_basic_2d() {
        let a = [1.0, 2.0];
        let b = [4.0, 6.0];

        // (1 - 4)^2 + (2 - 6)^2 = 9 + 16 = 25
        let got = l2_sq_f32(&a, &b);

        assert!(approx_eq(got, 25.0, 1e-6));
    }

    #[test]
    fn l2_sq_same_vector_is_zero() {
        let a = [2.0, -3.0, 4.0];

        let got = l2_sq_f32(&a, &a);

        assert!(approx_eq(got, 0.0, 1e-6));
    }

    #[test]
    fn l2_sq_works_with_vec() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = vec![2.0, 4.0, 6.0];

        // (1 - 2)^2 + (2 - 4)^2 + (3 - 6)^2
        // = 1 + 4 + 9 = 14
        let got = l2_sq_f32(&a, &b);

        assert!(approx_eq(got, 14.0, 1e-6));
    }

    #[test]
    fn l2_sq_is_symmetric() {
        let a = [1.0, -2.0, 3.5];
        let b = [4.0, 0.0, -1.5];

        let ab = l2_sq_f32(&a, &b);
        let ba = l2_sq_f32(&b, &a);

        assert!(approx_eq(ab, ba, 1e-6));
    }

    #[test]
    #[should_panic]
    fn l2_sq_panics_on_different_lengths() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0, 3.0];

        let _ = l2_sq_f32(&a, &b);
    }

    #[test]
    fn l2_norm_sq_basic() {
        let a = [2.0, -3.0, 4.0];

        // 2^2 + (-3)^2 + 4^2 = 4 + 9 + 16 = 29
        let got = l2_norm_sq_f32(&a);

        assert!(approx_eq(got, 29.0, 1e-6));
    }

    #[test]
    fn l2_norm_sq_zero_vector() {
        let a = [0.0, 0.0, 0.0];

        let got = l2_norm_sq_f32(&a);

        assert!(approx_eq(got, 0.0, 1e-6));
    }

    #[test]
    fn cosine_same_direction_is_one() {
        let a = [1.0, 0.0];
        let b = [2.0, 0.0];

        let got = cosine_similarity_f32(&a, &b);

        assert_some_approx_eq(got, 1.0, 1e-6);
    }

    #[test]
    fn cosine_opposite_direction_is_minus_one() {
        let a = [1.0, 0.0];
        let b = [-2.0, 0.0];

        let got = cosine_similarity_f32(&a, &b);

        assert_some_approx_eq(got, -1.0, 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors_is_zero() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];

        let got = cosine_similarity_f32(&a, &b);

        assert_some_approx_eq(got, 0.0, 1e-6);
    }

    #[test]
    fn cosine_basic_3d() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];

        // dot = 32
        // ||a|| = sqrt(14)
        // ||b|| = sqrt(77)
        // cosine = 32 / (sqrt(14) * sqrt(77)) ~= 0.9746318
        let got = cosine_similarity_f32(&a, &b);

        assert_some_approx_eq(got, 0.9746318, 1e-6);
    }

    #[test]
    fn cosine_zero_vector_returns_none() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 2.0, 3.0];

        let got = cosine_similarity_f32(&a, &b);

        assert_eq!(got, None);
    }

    #[test]
    #[should_panic]
    fn cosine_panics_on_different_lengths() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0, 3.0];

        let _ = cosine_similarity_f32(&a, &b);
    }
}
