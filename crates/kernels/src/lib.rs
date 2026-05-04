#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    // Preconditions: dot product only makes sense for equal-length vectors.
    assert_eq!(a.len(), b.len(), "Vectors must have equal length");

    let mut sum = 0.0f32;

    // Simple baseline loop / need optimization later (SIMD/ASM).
    for i in 0..a.len() {
        sum += a[i] * b[i];
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

    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
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
}
