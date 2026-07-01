use std::error::Error;
use std::fmt;

pub type KernelResult<T> = Result<T, KernelError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    LengthMismatch { left: usize, right: usize },
    ZeroVector,
    EmptyCandidates,
    InvalidK { k: usize, len: usize },
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { left, right } => {
                write!(f, "vectors must have equal length: {left} vs {right}")
            }
            Self::ZeroVector => write!(f, "operation is undefined for zero vectors"),
            Self::EmptyCandidates => write!(f, "candidate set must not be empty"),
            Self::InvalidK { k, len } => write!(f, "k must be in 1..={len}, got {k}"),
        }
    }
}

impl Error for KernelError {}

#[inline]
fn validate_equal_len(a: &[f32], b: &[f32]) -> KernelResult<()> {
    if a.len() != b.len() {
        return Err(KernelError::LengthMismatch {
            left: a.len(),
            right: b.len(),
        });
    }

    Ok(())
}

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

#[inline]
pub fn try_dot_f32(a: &[f32], b: &[f32]) -> KernelResult<f32> {
    validate_equal_len(a, b)?;
    Ok(dot_f32(a, b))
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

#[inline]
pub fn try_l2_sq_f32(a: &[f32], b: &[f32]) -> KernelResult<f32> {
    validate_equal_len(a, b)?;
    Ok(l2_sq_f32(a, b))
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

#[inline]
pub fn l2_norm_f32(a: &[f32]) -> f32 {
    l2_norm_sq_f32(a).sqrt()
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

#[inline]
pub fn try_cosine_similarity_f32(a: &[f32], b: &[f32]) -> KernelResult<f32> {
    validate_equal_len(a, b)?;
    cosine_similarity_f32(a, b).ok_or(KernelError::ZeroVector)
}

pub fn normalize_l2_inplace_f32(values: &mut [f32]) -> KernelResult<()> {
    let norm = l2_norm_f32(values);
    if norm == 0.0 {
        return Err(KernelError::ZeroVector);
    }

    for value in values {
        *value /= norm;
    }

    Ok(())
}

pub fn normalized_l2_f32(values: &[f32]) -> KernelResult<Vec<f32>> {
    let mut normalized = values.to_vec();
    normalize_l2_inplace_f32(&mut normalized)?;
    Ok(normalized)
}

pub fn nearest_l2_sq_f32(query: &[f32], candidates: &[&[f32]]) -> KernelResult<(usize, f32)> {
    let mut best: Option<(usize, f32)> = None;

    for (index, candidate) in candidates.iter().enumerate() {
        let distance = try_l2_sq_f32(query, candidate)?;
        match best {
            Some((_, best_distance)) if best_distance <= distance => {}
            _ => best = Some((index, distance)),
        }
    }

    best.ok_or(KernelError::EmptyCandidates)
}

pub fn l2_sq_all_f32(query: &[f32], candidates: &[&[f32]]) -> KernelResult<Vec<f32>> {
    if candidates.is_empty() {
        return Err(KernelError::EmptyCandidates);
    }

    candidates
        .iter()
        .map(|candidate| try_l2_sq_f32(query, candidate))
        .collect()
}

pub fn nearest_k_l2_sq_f32(
    query: &[f32],
    candidates: &[&[f32]],
    k: usize,
) -> KernelResult<Vec<(usize, f32)>> {
    if candidates.is_empty() {
        return Err(KernelError::EmptyCandidates);
    }
    if k == 0 || k > candidates.len() {
        return Err(KernelError::InvalidK {
            k,
            len: candidates.len(),
        });
    }

    let mut scored = l2_sq_all_f32(query, candidates)?
        .into_iter()
        .enumerate()
        .collect::<Vec<_>>();

    scored.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    scored.truncate(k);

    Ok(scored)
}

pub fn cosine_similarity_all_f32(query: &[f32], candidates: &[&[f32]]) -> KernelResult<Vec<f32>> {
    if candidates.is_empty() {
        return Err(KernelError::EmptyCandidates);
    }

    candidates
        .iter()
        .map(|candidate| try_cosine_similarity_f32(query, candidate))
        .collect()
}

pub fn nearest_cosine_similarity_f32(
    query: &[f32],
    candidates: &[&[f32]],
) -> KernelResult<(usize, f32)> {
    let mut best: Option<(usize, f32)> = None;

    for (index, candidate) in candidates.iter().enumerate() {
        let similarity = try_cosine_similarity_f32(query, candidate)?;
        match best {
            Some((_, best_similarity)) if best_similarity >= similarity => {}
            _ => best = Some((index, similarity)),
        }
    }

    best.ok_or(KernelError::EmptyCandidates)
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

    fn assert_approx_eq(got: f32, expected: f32, eps: f32) {
        assert!(
            approx_eq(got, expected, eps),
            "expected {expected}, got {got}"
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
    fn try_dot_reports_different_lengths() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0, 3.0];

        let got = try_dot_f32(&a, &b);

        assert_eq!(got, Err(KernelError::LengthMismatch { left: 2, right: 3 }));
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
    fn l2_norm_returns_square_root_of_norm_sq() {
        let a = [3.0, 4.0];

        let got = l2_norm_f32(&a);

        assert_approx_eq(got, 5.0, 1e-6);
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
    fn try_cosine_zero_vector_returns_error() {
        let a = [0.0, 0.0];
        let b = [1.0, 2.0];

        let got = try_cosine_similarity_f32(&a, &b);

        assert_eq!(got, Err(KernelError::ZeroVector));
    }

    #[test]
    #[should_panic]
    fn cosine_panics_on_different_lengths() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0, 3.0];

        let _ = cosine_similarity_f32(&a, &b);
    }

    #[test]
    fn normalize_l2_inplace_scales_vector_to_unit_norm() {
        let mut values = [3.0, 4.0];

        normalize_l2_inplace_f32(&mut values).expect("normalization should succeed");

        assert_approx_eq(values[0], 0.6, 1e-6);
        assert_approx_eq(values[1], 0.8, 1e-6);
        assert_approx_eq(l2_norm_f32(&values), 1.0, 1e-6);
    }

    #[test]
    fn normalized_l2_returns_new_vector() {
        let values = [0.0, 5.0];

        let got = normalized_l2_f32(&values).expect("normalization should succeed");

        assert_eq!(values, [0.0, 5.0]);
        assert_approx_eq(got[0], 0.0, 1e-6);
        assert_approx_eq(got[1], 1.0, 1e-6);
    }

    #[test]
    fn normalize_l2_rejects_zero_vector() {
        let mut values = [0.0, 0.0];

        let got = normalize_l2_inplace_f32(&mut values);

        assert_eq!(got, Err(KernelError::ZeroVector));
    }

    #[test]
    fn nearest_l2_sq_returns_index_and_distance() {
        let query = [1.0, 1.0];
        let candidates: [&[f32]; 3] = [&[5.0, 5.0], &[2.0, 1.0], &[0.0, 0.0]];

        let got = nearest_l2_sq_f32(&query, &candidates);

        assert_eq!(got, Ok((1, 1.0)));
    }

    #[test]
    fn nearest_l2_sq_rejects_empty_candidates() {
        let query = [1.0, 1.0];
        let candidates: [&[f32]; 0] = [];

        let got = nearest_l2_sq_f32(&query, &candidates);

        assert_eq!(got, Err(KernelError::EmptyCandidates));
    }

    #[test]
    fn l2_sq_all_returns_distances_for_each_candidate() {
        let query = [1.0, 1.0];
        let candidates: [&[f32]; 3] = [&[1.0, 1.0], &[2.0, 1.0], &[3.0, 3.0]];

        let got = l2_sq_all_f32(&query, &candidates);

        assert_eq!(got, Ok(vec![0.0, 1.0, 8.0]));
    }

    #[test]
    fn l2_sq_all_rejects_empty_candidates() {
        let query = [1.0, 1.0];
        let candidates: [&[f32]; 0] = [];

        let got = l2_sq_all_f32(&query, &candidates);

        assert_eq!(got, Err(KernelError::EmptyCandidates));
    }

    #[test]
    fn nearest_k_l2_sq_returns_sorted_neighbors() {
        let query = [1.0, 1.0];
        let candidates: [&[f32]; 4] = [&[5.0, 5.0], &[2.0, 1.0], &[1.0, 1.0], &[0.0, 0.0]];

        let got = nearest_k_l2_sq_f32(&query, &candidates, 3);

        assert_eq!(got, Ok(vec![(2, 0.0), (1, 1.0), (3, 2.0)]));
    }

    #[test]
    fn nearest_k_l2_sq_rejects_zero_k() {
        let query = [1.0, 1.0];
        let candidates: [&[f32]; 1] = [&[1.0, 1.0]];

        let got = nearest_k_l2_sq_f32(&query, &candidates, 0);

        assert_eq!(got, Err(KernelError::InvalidK { k: 0, len: 1 }));
    }

    #[test]
    fn cosine_similarity_all_returns_similarity_per_candidate() {
        let query = [1.0, 0.0];
        let candidates: [&[f32]; 3] = [&[1.0, 0.0], &[0.0, 1.0], &[-1.0, 0.0]];

        let got =
            cosine_similarity_all_f32(&query, &candidates).expect("cosine scoring should succeed");

        assert_approx_eq(got[0], 1.0, 1e-6);
        assert_approx_eq(got[1], 0.0, 1e-6);
        assert_approx_eq(got[2], -1.0, 1e-6);
    }

    #[test]
    fn nearest_cosine_similarity_returns_highest_similarity() {
        let query = [1.0, 0.0];
        let candidates: [&[f32]; 3] = [&[0.0, 1.0], &[0.5, 0.0], &[-1.0, 0.0]];

        let got = nearest_cosine_similarity_f32(&query, &candidates);

        assert_eq!(got, Ok((1, 1.0)));
    }

    #[test]
    fn nearest_cosine_similarity_rejects_zero_candidate() {
        let query = [1.0, 0.0];
        let candidates: [&[f32]; 1] = [&[0.0, 0.0]];

        let got = nearest_cosine_similarity_f32(&query, &candidates);

        assert_eq!(got, Err(KernelError::ZeroVector));
    }
}
