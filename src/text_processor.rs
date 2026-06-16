use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct TimestampItem {
    pub text: String,
    pub start_time: f64,
    pub end_time: f64,
}

/// Encode text with timestamp markers for forced alignment.
/// Returns (word_list, formatted_input_string).
pub fn encode_timestamp(text: &str, language: &str) -> (Vec<String>, String) {
    let words = split_words(text, language);
    let timestamp_template = "<timestamp><timestamp>";
    let mut formatted = String::new();
    formatted.push_str(timestamp_template);
    for word in &words {
        formatted.push_str(word);
        formatted.push_str(timestamp_template);
    }
    (words, formatted)
}

fn split_words(text: &str, language: &str) -> Vec<String> {
    match language {
        "Chinese" | "Japanese" | "Korean" => {
            let mut words = Vec::new();
            let mut latin_buf = String::new();

            let cjk_range = |c: char| -> bool {
                ('\u{4E00}'..='\u{9FFF}').contains(&c)
                    || ('\u{3400}'..='\u{4DBF}').contains(&c)
                    || ('\u{3040}'..='\u{30FF}').contains(&c)
                    || ('\u{AC00}'..='\u{D7AF}').contains(&c)
                    || ('\u{3000}'..='\u{303F}').contains(&c)
                    || ('\u{FF00}'..='\u{FFEF}').contains(&c)
            };

            for c in text.chars() {
                if cjk_range(c) {
                    if !latin_buf.is_empty() {
                        words.push(latin_buf.clone());
                        latin_buf.clear();
                    }
                    words.push(c.to_string());
                } else if c.is_alphanumeric() || c == '\'' || c == '-' {
                    latin_buf.push(c);
                } else if c.is_whitespace() || c.is_ascii_punctuation() {
                    if !latin_buf.is_empty() {
                        words.push(latin_buf.clone());
                        latin_buf.clear();
                    }
                } else {
                    if !latin_buf.is_empty() {
                        words.push(latin_buf.clone());
                        latin_buf.clear();
                    }
                    words.push(c.to_string());
                }
            }
            if !latin_buf.is_empty() {
                words.push(latin_buf);
            }
            words
        }
        _ => text
            .split_whitespace()
            .map(|w| w.to_string())
            .collect(),
    }
}

/// Parse raw timestamp values into word-level TimestampItems.
/// Each word gets 2 timestamps: start (even index) and end (odd index).
pub fn parse_timestamp(words: &[String], timestamps: &[f64]) -> Vec<TimestampItem> {
    words
        .iter()
        .enumerate()
        .map(|(i, word)| {
            let start = timestamps.get(i * 2).copied().unwrap_or(0.0);
            let end = timestamps.get(i * 2 + 1).copied().unwrap_or(0.0);
            TimestampItem {
                text: word.clone(),
                start_time: (start / 1000.0 * 1000.0).round() / 1000.0,
                end_time: (end / 1000.0 * 1000.0).round() / 1000.0,
            }
        })
        .collect()
}

/// LIS-based monotonicity fix for timestamp predictions.
/// Ensures timestamps are non-decreasing.
pub fn fix_timestamp(data: &[f64]) -> Vec<f64> {
    if data.len() <= 1 {
        return data.to_vec();
    }

    let lis = longest_increasing_subsequence(data);

    let mut valid = vec![false; data.len()];
    for &idx in &lis {
        valid[idx] = true;
    }

    let mut result = vec![0.0; data.len()];

    for &idx in &lis {
        result[idx] = data[idx];
    }

    let mut i = 0;
    while i < data.len() {
        if valid[i] {
            i += 1;
            continue;
        }

        let mut j = i;
        while j < data.len() && !valid[j] {
            j += 1;
        }

        let anomaly_len = j - i;

        if anomaly_len <= 2 {
            for k in i..j {
                let left_val = if i > 0 {
                    result[i - 1]
                } else {
                    result[j]
                };
                let right_val = if j < data.len() {
                    result[j]
                } else {
                    result[i - 1]
                };
                let ratio = (k - i + 1) as f64 / (anomaly_len + 1) as f64;
                result[k] = left_val + (right_val - left_val) * ratio;
            }
        } else {
            let left_val = if i > 0 {
                result[i - 1]
            } else {
                data[i]
            };
            let right_val = if j < data.len() {
                result[j]
            } else {
                data[j - 1]
            };
            for k in i..j {
                let ratio = (k - i + 1) as f64 / (anomaly_len + 1) as f64;
                result[k] = left_val + (right_val - left_val) * ratio;
            }
        }

        i = j;
    }

    result
}

fn longest_increasing_subsequence(data: &[f64]) -> Vec<usize> {
    if data.is_empty() {
        return vec![];
    }

    let n = data.len();
    let mut dp = vec![1usize; n];
    let mut prev = vec![None; n];

    for i in 0..n {
        for j in 0..i {
            if data[j] <= data[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
                prev[i] = Some(j);
            }
        }
    }

    let mut max_len = 0;
    let mut max_idx = 0;
    for i in 0..n {
        if dp[i] > max_len {
            max_len = dp[i];
            max_idx = i;
        }
    }

    let mut lis = Vec::new();
    let mut current = Some(max_idx);
    while let Some(idx) = current {
        lis.push(idx);
        current = prev[idx];
    }
    lis.reverse();
    lis
}
