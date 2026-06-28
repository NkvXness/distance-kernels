use kernels::{
    cosine_similarity_f32, dot_f32, l2_norm_f32, l2_sq_f32, nearest_l2_sq_f32, normalized_l2_f32,
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
        "nearest-l2-sq" => run_nearest_l2_sq(&args),
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

fn format_vector(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{value}"))
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
  cargo run -p cli -- nearest-l2-sq <query> <candidate;candidate>"
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
}
