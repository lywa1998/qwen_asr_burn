use serde::Serialize;

const FRAME_SIZE: usize = 256; // samples per frame at 16kHz = 16ms
const SAMPLE_RATE: u32 = 16_000;
const VOICE_THRESHOLD: f32 = 0.5;
const MERGE_GAP_SECS: f64 = 0.5; // merge segments closer than this
const MAX_SEGMENT_SECS: f64 = 30.0; // split segments longer than this
const MIN_SEGMENT_SECS: f64 = 0.5; // drop segments shorter than this
const OVERLAP_SECS: f64 = 1.0; // overlap when splitting long segments

#[derive(Debug, Clone, Serialize)]
pub struct VoiceSegment {
    /// Start time in seconds relative to original audio
    pub start_secs: f64,
    /// End time in seconds relative to original audio
    pub end_secs: f64,
}

/// Run VAD on f32 mono 16kHz audio, returning voice segments with absolute timestamps.
pub fn detect_segments(samples: &[f32]) -> Vec<VoiceSegment> {
    if samples.is_empty() {
        return vec![];
    }

    let mut detector = earshot::Detector::default();
    let num_full_frames = samples.len() / FRAME_SIZE;

    // Collect per-frame speech probabilities
    let mut scores: Vec<f32> = Vec::with_capacity(num_full_frames);
    for i in 0..num_full_frames {
        let start = i * FRAME_SIZE;
        let frame: &[f32; FRAME_SIZE] = samples[start..start + FRAME_SIZE]
            .try_into()
            .unwrap();
        let score = detector.predict_f32(frame);
        scores.push(score);
    }

    if scores.is_empty() {
        return vec![];
    }

    let frame_dur = FRAME_SIZE as f64 / SAMPLE_RATE as f64;

    // Step 1: merge consecutive speech frames into raw segments
    let mut raw_segments: Vec<(usize, usize)> = Vec::new();
    let mut in_speech = false;
    let mut seg_start = 0usize;

    for (i, &score) in scores.iter().enumerate() {
        let is_speech = score >= VOICE_THRESHOLD;
        if is_speech && !in_speech {
            seg_start = i;
            in_speech = true;
        } else if !is_speech && in_speech {
            raw_segments.push((seg_start, i));
            in_speech = false;
        }
    }
    if in_speech {
        raw_segments.push((seg_start, scores.len()));
    }

    if raw_segments.is_empty() {
        return vec![];
    }

    // Step 2: merge adjacent segments with gap < MERGE_GAP_SECS
    let gap_frames = (MERGE_GAP_SECS / frame_dur) as usize;
    let mut merged: Vec<(usize, usize)> = Vec::new();
    let mut current = raw_segments[0];

    for &seg in &raw_segments[1..] {
        if seg.0 - current.1 < gap_frames {
            // merge
            current.1 = seg.1;
        } else {
            merged.push(current);
            current = seg;
        }
    }
    merged.push(current);

    // Step 3: enforce max segment length, split at nearest silence
    let max_frames = (MAX_SEGMENT_SECS / frame_dur) as usize;
    let min_frames = (MIN_SEGMENT_SECS / frame_dur) as usize;
    let overlap_frames = (OVERLAP_SECS / frame_dur) as usize;

    let mut segments: Vec<VoiceSegment> = Vec::new();

    for (start, end) in merged {
        let len = end - start;
        if len <= max_frames {
            if len >= min_frames {
                segments.push(VoiceSegment {
                    start_secs: start as f64 * frame_dur,
                    end_secs: end as f64 * frame_dur,
                });
            }
            continue;
        }

        // Split long segment: find silence points within overlap region
        let mut chunk_start = start;
        while chunk_start < end {
            let chunk_end_candidate = (chunk_start + max_frames).min(end);
            if chunk_end_candidate >= end {
                // last chunk
                let seg_len = end - chunk_start;
                if seg_len >= min_frames {
                    segments.push(VoiceSegment {
                        start_secs: chunk_start as f64 * frame_dur,
                        end_secs: end as f64 * frame_dur,
                    });
                }
                break;
            }

            // Find a good split point: look for silence (score < threshold) near chunk_end
            let split_start = if chunk_end_candidate > overlap_frames {
                chunk_end_candidate - overlap_frames
            } else {
                chunk_start
            };
            let split_end = (chunk_end_candidate + overlap_frames).min(end);

            // Find the frame with minimum score (most silent) in the overlap window
            let mut best_split = chunk_end_candidate;
            let mut best_score = f32::MAX;
            for i in split_start..split_end {
                if i < scores.len() && scores[i] < best_score {
                    best_score = scores[i];
                    best_split = i;
                }
            }

            let seg_len = best_split - chunk_start;
            if seg_len >= min_frames {
                segments.push(VoiceSegment {
                    start_secs: chunk_start as f64 * frame_dur,
                    end_secs: best_split as f64 * frame_dur,
                });
            }
            chunk_start = best_split;
        }
    }

    log::info!(
        "VAD: {} frames scored, {} raw segments → {} final segments",
        scores.len(),
        raw_segments.len(),
        segments.len()
    );
    segments
}
