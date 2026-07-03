use kernels::{
    cosine_similarity_all_f32, cosine_similarity_f32, dot_f32, knn_predict_l2_sq_u32, l2_norm_f32,
    l2_sq_all_f32, l2_sq_f32, nearest_cosine_similarity_f32, nearest_k_l2_sq_f32,
    nearest_l2_sq_f32, normalized_l2_f32,
};

fn main() {
    match run(std::env::args().skip(1).collect()) {
        Ok(output) => println!("{output}"),
        Err(error) => {
            eprintln!("{error}");
            eprintln!("{}", usage());
            std::process::exit(2);
        }
    }
}

fn run(args: Vec<String>) -> Result<String, String> {
    if args.is_empty() {
        return Err("expected a command".to_string());
    }

    let command = &args[0];
    match command.as_str() {
        "dot" => run_binary_kernel(&args, dot_f32),
        "l2-sq" => run_binary_kernel(&args, l2_sq_f32),
        "cosine" => run_cosine(&args),
        "norm" => run_norm(&args),
        "normalize" => run_normalize(&args),
        "l2-sq-all" => run_l2_sq_all(&args),
        "nearest-l2-sq" => run_nearest_l2_sq(&args),
        "nearest-k-l2-sq" => run_nearest_k_l2_sq(&args),
        "cosine-all" => run_cosine_all(&args),
        "nearest-cosine" => run_nearest_cosine(&args),
        "knn-l2-sq" => run_knn_l2_sq(&args),
        _ => Err(format!("unknown command: {command}")),
    }
}

fn run_binary_kernel(args: &[String], kernel: fn(&[f32], &[f32]) -> f32) -> Result<String, String> {
    let (a, b) = parse_binary_vectors(args)?;
    Ok(format!("{}", kernel(&a, &b)))
}

fn run_cosine(args: &[String]) -> Result<String, String> {
    let (a, b) = parse_binary_vectors(args)?;
    match cosine_similarity_f32(&a, &b) {
        Some(value) => Ok(format!("{value}")),
        None => Err("cosine similarity is undefined for zero vectors".to_string()),
    }
}

fn run_norm(args: &[String]) -> Result<String, String> {
    if args.len() != 2 {
        return Err("norm expects exactly 1 vector argument".to_string());
    }

    let values = parse_vector(&args[1])?;
    Ok(format!("{}", l2_norm_f32(&values)))
}

fn run_normalize(args: &[String]) -> Result<String, String> {
    if args.len() != 2 {
        return Err("normalize expects exactly 1 vector argument".to_string());
    }

    let values = parse_vector(&args[1])?;
    let normalized =
        normalized_l2_f32(&values).map_err(|_| "normalization is undefined for zero vectors")?;

    Ok(format_vector(&normalized))
}

fn run_nearest_l2_sq(args: &[String]) -> Result<String, String> {
    if args.len() != 3 {
        return Err("nearest-l2-sq expects a query vector and candidate set".to_string());
    }

    let query = parse_vector(&args[1])?;
    let candidates = parse_vector_set(&args[2])?;
    let candidate_refs: Vec<&[f32]> = candidates.iter().map(Vec::as_slice).collect();
    let (index, distance) =
        nearest_l2_sq_f32(&query, &candidate_refs).map_err(|error| error.to_string())?;

    Ok(format!("{index}:{distance}"))
}

fn run_l2_sq_all(args: &[String]) -> Result<String, String> {
    let (query, candidates) = parse_query_and_candidates(args, "l2-sq-all")?;
    let candidate_refs: Vec<&[f32]> = candidates.iter().map(Vec::as_slice).collect();
    let distances = l2_sq_all_f32(&query, &candidate_refs).map_err(|error| error.to_string())?;

    Ok(format_vector(&distances))
}

fn run_nearest_k_l2_sq(args: &[String]) -> Result<String, String> {
    if args.len() != 4 {
        return Err("nearest-k-l2-sq expects query, candidates, and k".to_string());
    }

    let query = parse_vector(&args[1])?;
    let candidates = parse_vector_set(&args[2])?;
    let k = parse_usize(&args[3], "k")?;
    let candidate_refs: Vec<&[f32]> = candidates.iter().map(Vec::as_slice).collect();
    let neighbors =
        nearest_k_l2_sq_f32(&query, &candidate_refs, k).map_err(|error| error.to_string())?;

    Ok(format_index_scores(&neighbors))
}

fn run_cosine_all(args: &[String]) -> Result<String, String> {
    let (query, candidates) = parse_query_and_candidates(args, "cosine-all")?;
    let candidate_refs: Vec<&[f32]> = candidates.iter().map(Vec::as_slice).collect();
    let similarities =
        cosine_similarity_all_f32(&query, &candidate_refs).map_err(|error| error.to_string())?;

    Ok(format_vector(&similarities))
}

fn run_nearest_cosine(args: &[String]) -> Result<String, String> {
    let (query, candidates) = parse_query_and_candidates(args, "nearest-cosine")?;
    let candidate_refs: Vec<&[f32]> = candidates.iter().map(Vec::as_slice).collect();
    let (index, similarity) = nearest_cosine_similarity_f32(&query, &candidate_refs)
        .map_err(|error| error.to_string())?;

    Ok(format!("{index}:{similarity}"))
}

fn run_knn_l2_sq(args: &[String]) -> Result<String, String> {
    if args.len() != 5 {
        return Err("knn-l2-sq expects query, samples, labels, and k".to_string());
    }

    let query = parse_vector(&args[1])?;
    let samples = parse_vector_set(&args[2])?;
    let labels = parse_labels(&args[3])?;
    let k = parse_usize(&args[4], "k")?;
    let sample_refs: Vec<&[f32]> = samples.iter().map(Vec::as_slice).collect();
    let label = knn_predict_l2_sq_u32(&query, &sample_refs, &labels, k)
        .map_err(|error| error.to_string())?;

    Ok(format!("{label}"))
}

fn parse_query_and_candidates(
    args: &[String],
    command: &str,
) -> Result<(Vec<f32>, Vec<Vec<f32>>), String> {
    if args.len() != 3 {
        return Err(format!(
            "{command} expects a query vector and candidate set"
        ));
    }

    Ok((parse_vector(&args[1])?, parse_vector_set(&args[2])?))
}

fn parse_binary_vectors(args: &[String]) -> Result<(Vec<f32>, Vec<f32>), String> {
    if args.len() != 3 {
        return Err("binary kernels expect exactly 2 vector arguments".to_string());
    }

    let a = parse_vector(&args[1])?;
    let b = parse_vector(&args[2])?;

    if a.len() != b.len() {
        return Err(format!(
            "vectors must have equal length: {} vs {}",
            a.len(),
            b.len()
        ));
    }

    Ok((a, b))
}

fn parse_vector(input: &str) -> Result<Vec<f32>, String> {
    if input.trim().is_empty() {
        return Err("vectors must not be empty".to_string());
    }

    input
        .split(',')
        .map(|part| {
            let value = part.trim();
            if value.is_empty() {
                return Err(format!("invalid vector element in '{input}'"));
            }
            value
                .parse::<f32>()
                .map_err(|_| format!("invalid f32 value: {value}"))
        })
        .collect()
}

fn parse_vector_set(input: &str) -> Result<Vec<Vec<f32>>, String> {
    if input.trim().is_empty() {
        return Err("candidate set must not be empty".to_string());
    }

    input.split(';').map(parse_vector).collect()
}

fn parse_labels(input: &str) -> Result<Vec<u32>, String> {
    if input.trim().is_empty() {
        return Err("labels must not be empty".to_string());
    }

    input
        .split(',')
        .map(|part| {
            let value = part.trim();
            if value.is_empty() {
                return Err(format!("invalid label in '{input}'"));
            }
            value
                .parse::<u32>()
                .map_err(|_| format!("invalid u32 label: {value}"))
        })
        .collect()
}

fn parse_usize(input: &str, name: &str) -> Result<usize, String> {
    input
        .parse::<usize>()
        .map_err(|_| format!("invalid {name}: {input}"))
}

fn format_vector(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{value}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_index_scores(values: &[(usize, f32)]) -> String {
    values
        .iter()
        .map(|(index, score)| format!("{index}:{score}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn usage() -> &'static str {
    "usage:
  cargo run -p cli -- dot <a,b,c> <x,y,z>
  cargo run -p cli -- l2-sq <a,b,c> <x,y,z>
  cargo run -p cli -- cosine <a,b,c> <x,y,z>
  cargo run -p cli -- norm <a,b,c>
  cargo run -p cli -- normalize <a,b,c>
  cargo run -p cli -- l2-sq-all <query> <candidate;candidate>
  cargo run -p cli -- nearest-l2-sq <query> <candidate;candidate>
  cargo run -p cli -- nearest-k-l2-sq <query> <candidate;candidate> <k>
  cargo run -p cli -- cosine-all <query> <candidate;candidate>
  cargo run -p cli -- nearest-cosine <query> <candidate;candidate>
  cargo run -p cli -- knn-l2-sq <query> <sample;sample> <label,label> <k>"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_command_returns_dot_product() {
        let got = run(vec![
            "dot".to_string(),
            "1,2,3".to_string(),
            "4,5,6".to_string(),
        ]);

        assert_eq!(got, Ok("32".to_string()));
    }

    #[test]
    fn l2_sq_command_returns_squared_distance() {
        let got = run(vec![
            "l2-sq".to_string(),
            "1,2,3".to_string(),
            "4,5,6".to_string(),
        ]);

        assert_eq!(got, Ok("27".to_string()));
    }

    #[test]
    fn cosine_command_handles_orthogonal_vectors() {
        let got = run(vec![
            "cosine".to_string(),
            "1,0".to_string(),
            "0,1".to_string(),
        ]);

        assert_eq!(got, Ok("0".to_string()));
    }

    #[test]
    fn rejects_mismatched_lengths() {
        let got = run(vec![
            "dot".to_string(),
            "1,2".to_string(),
            "1,2,3".to_string(),
        ]);

        assert_eq!(
            got,
            Err("vectors must have equal length: 2 vs 3".to_string())
        );
    }

    #[test]
    fn rejects_zero_vector_cosine() {
        let got = run(vec![
            "cosine".to_string(),
            "0,0".to_string(),
            "1,2".to_string(),
        ]);

        assert_eq!(
            got,
            Err("cosine similarity is undefined for zero vectors".to_string())
        );
    }

    #[test]
    fn norm_command_returns_l2_norm() {
        let got = run(vec!["norm".to_string(), "3,4".to_string()]);

        assert_eq!(got, Ok("5".to_string()));
    }

    #[test]
    fn normalize_command_returns_unit_vector() {
        let got = run(vec!["normalize".to_string(), "3,4".to_string()]);

        assert_eq!(got, Ok("0.6,0.8".to_string()));
    }

    #[test]
    fn nearest_l2_sq_command_returns_index_and_distance() {
        let got = run(vec![
            "nearest-l2-sq".to_string(),
            "1,1".to_string(),
            "5,5;2,1;0,0".to_string(),
        ]);

        assert_eq!(got, Ok("1:1".to_string()));
    }

    #[test]
    fn l2_sq_all_command_returns_distances() {
        let got = run(vec![
            "l2-sq-all".to_string(),
            "1,1".to_string(),
            "1,1;2,1;3,3".to_string(),
        ]);

        assert_eq!(got, Ok("0,1,8".to_string()));
    }

    #[test]
    fn nearest_k_l2_sq_command_returns_neighbors() {
        let got = run(vec![
            "nearest-k-l2-sq".to_string(),
            "1,1".to_string(),
            "5,5;2,1;1,1;0,0".to_string(),
            "3".to_string(),
        ]);

        assert_eq!(got, Ok("2:0,1:1,3:2".to_string()));
    }

    #[test]
    fn cosine_all_command_returns_similarities() {
        let got = run(vec![
            "cosine-all".to_string(),
            "1,0".to_string(),
            "1,0;0,1;-1,0".to_string(),
        ]);

        assert_eq!(got, Ok("1,0,-1".to_string()));
    }

    #[test]
    fn nearest_cosine_command_returns_best_similarity() {
        let got = run(vec![
            "nearest-cosine".to_string(),
            "1,0".to_string(),
            "0,1;1,0;-1,0".to_string(),
        ]);

        assert_eq!(got, Ok("1:1".to_string()));
    }

    #[test]
    fn knn_l2_sq_command_returns_predicted_label() {
        let got = run(vec![
            "knn-l2-sq".to_string(),
            "1,1".to_string(),
            "1,1;1.2,1.1;8,8;7.5,8".to_string(),
            "10,10,20,20".to_string(),
            "3".to_string(),
        ]);

        assert_eq!(got, Ok("10".to_string()));
    }
}
