use kernels::{cosine_similarity_f32, dot_f32, l2_sq_f32};

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
    if args.len() != 3 {
        return Err("expected exactly 3 arguments".to_string());
    }

    let command = &args[0];
    let a = parse_vector(&args[1])?;
    let b = parse_vector(&args[2])?;

    if a.len() != b.len() {
        return Err(format!(
            "vectors must have equal length: {} vs {}",
            a.len(),
            b.len()
        ));
    }

    match command.as_str() {
        "dot" => Ok(format!("{}", dot_f32(&a, &b))),
        "l2-sq" => Ok(format!("{}", l2_sq_f32(&a, &b))),
        "cosine" => match cosine_similarity_f32(&a, &b) {
            Some(value) => Ok(format!("{value}")),
            None => Err("cosine similarity is undefined for zero vectors".to_string()),
        },
        _ => Err(format!("unknown command: {command}")),
    }
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

fn usage() -> &'static str {
    "usage: cargo run -p cli -- <dot|l2-sq|cosine> <a,b,c> <x,y,z>"
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
}
