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
}
