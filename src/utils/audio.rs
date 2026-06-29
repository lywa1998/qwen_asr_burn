use anyhow::anyhow;
use rustfft::{num_complex::Complex, FftPlanner};

pub struct MelSpectrogram {
    n_fft: usize,
    hop_length: usize,
    num_mel_bins: usize,
    n_freqs: usize,
    hann_window: Vec<f32>,
    mel_filters: Vec<f32>, // flat: [num_mel_bins × n_freqs]
}

impl MelSpectrogram {
    pub fn new(n_fft: usize, hop_length: usize, n_mels: usize, sample_rate: u32) -> Self {
        let n_freqs = n_fft / 2 + 1;
        let hann_window = hann_window(n_fft);

        // Try loading HF's Whisper mel filterbank (exact match to Python reference).
        // Falls back to the candle-compatible Slaney filterbank if file not found.
        let mel_filters = Self::try_load_hf_filterbank(n_mels, n_freqs)
            .unwrap_or_else(|| {
                log::info!("Using built-in Slaney mel filterbank (HF filterbank not found)");
                create_mel_filterbank(n_mels, n_fft, sample_rate, 0.0, sample_rate as f64 / 2.0)
            });

        Self { n_fft, hop_length, num_mel_bins: n_mels, n_freqs, hann_window, mel_filters }
    }

    /// Try to load HuggingFace's exact Whisper mel filterbank from a pre-saved file.
    /// The filterbank must be saved as row-major f32: [num_mels × n_freqs].
    fn try_load_hf_filterbank(num_mels: usize, n_freqs: usize) -> Option<Vec<f32>> {
        let path = "/tmp/hf_mel_fb.bin";
        let data = std::fs::read(path).ok()?;
        let expected = num_mels * n_freqs * 4;
        if data.len() != expected {
            log::warn!("HF filterbank size mismatch: got {} bytes, expected {expected}", data.len());
            return None;
        }
        let filters: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        log::info!("Loaded HF Whisper mel filterbank from {path} ({num_mels}×{n_freqs})");
        Some(filters)
    }

    /// Extract log-mel spectrogram.
    /// Returns Vec<Vec<f32>> where the outer vec has length `num_mel_bins` and each
    /// inner vec has length `n_frames`.
    pub fn compute(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        // Pad to next multiple of hop_length for consistent frame count
        let padded_len = samples.len().div_ceil(self.hop_length) * self.hop_length;
        let mut padded_samples = samples.to_vec();
        padded_samples.resize(padded_len, 0.0);

        // Reflection-pad for center padding (n_fft/2 on each side)
        // This matches torch.stft's default pad_mode='reflect' with center=True.
        let pad = self.n_fft / 2;
        let padded_signal = reflection_pad(&padded_samples, pad);

        // STFT power spectrum: [n_freqs × n_frames_with_last], column-major
        let (power, n_frames_with_last) =
            compute_power_stft(&padded_signal, self.n_fft, self.hop_length, &self.hann_window);

        // Remove last frame (matches torch.stft[..., :-1] in WhisperFeatureExtractor)
        let n_frames = if n_frames_with_last > 0 { n_frames_with_last - 1 } else { 0 };

        // Apply mel filterbank: [num_mel_bins × n_freqs] · [n_freqs × n_frames]
        // → [num_mel_bins × n_frames]
        let mut mel_spec = vec![0.0f32; self.num_mel_bins * n_frames];
        for m in 0..self.num_mel_bins {
            let filter_row = &self.mel_filters[m * self.n_freqs..(m + 1) * self.n_freqs];
            let out_row = &mut mel_spec[m * n_frames..(m + 1) * n_frames];
            for f in 0..self.n_freqs {
                let w = filter_row[f];
                if w == 0.0 {
                    continue;
                }
                let power_row = &power[f * n_frames_with_last..f * n_frames_with_last + n_frames];
                for (t, &p) in power_row.iter().enumerate() {
                    out_row[t] += w * p;
                }
            }
        }

        // Log normalization: log10 via ln/ln(10)
        let log10_factor = 1.0 / 10.0f32.ln();
        let mut max_val = f32::NEG_INFINITY;
        for v in mel_spec.iter_mut() {
            *v = v.max(1e-10f32).ln() * log10_factor;
            if *v > max_val {
                max_val = *v;
            }
        }

        // Clamp and normalize: max(log_mel, max_val - 8.0) then (x + 4) / 4
        let min_val = max_val - 8.0;
        for v in mel_spec.iter_mut() {
            *v = (v.max(min_val) + 4.0) / 4.0;
        }

        // Convert flat to Vec<Vec<f32>> format expected by callers
        let mut result = vec![vec![0.0f32; n_frames]; self.num_mel_bins];
        for m in 0..self.num_mel_bins {
            let start = m * n_frames;
            result[m].copy_from_slice(&mel_spec[start..start + n_frames]);
        }
        result
    }

}

/// Pad or truncate audio to exactly 30 seconds (480,000 samples @ 16kHz).
/// The model was trained with WhisperFeatureExtractor which always produces
/// exactly 3000 mel frames (n_samples=480000).
pub fn pad_to_30s(samples: &[f32]) -> Vec<f32> {
    const TARGET: usize = 480_000;
    if samples.len() < TARGET {
        let mut v = samples.to_vec();
        v.resize(TARGET, 0.0);
        v
    } else if samples.len() > TARGET {
        samples[..TARGET].to_vec()
    } else {
        samples.to_vec()
    }
}

/// Pad audio to the next multiple of 30 seconds (480,000 samples @ 16kHz).
/// Used by the forced aligner, which mirrors Python's
/// `padding=True, truncation=False` in WhisperFeatureExtractor — long inputs
/// must not be truncated, or tail-word timestamps cluster at the boundary.
pub fn pad_to_30s_multiple(samples: &[f32]) -> Vec<f32> {
    const CHUNK: usize = 480_000;
    let target = samples.len().div_ceil(CHUNK).max(1) * CHUNK;
    let mut v = samples.to_vec();
    v.resize(target, 0.0);
    v
}

// ─── Hann window ────────────────────────────────────────────────────────────

/// Standard Hann window: `0.5 * (1 - cos(2πi / (N-1)))`.
/// Matches `torch.hann_window(N)`.
fn hann_window(n: usize) -> Vec<f32> {
    let denom = (n - 1) as f32;
    (0..n)
        .map(|i| {
            let x = 2.0 * std::f32::consts::PI * i as f32 / denom;
            0.5 * (1.0 - x.cos())
        })
        .collect()
}

// ─── Reflection padding ─────────────────────────────────────────────────────

/// Mirror the signal at both ends by `pad` samples.
/// Matches `torch.stft` default `pad_mode='reflect'`.
fn reflection_pad(signal: &[f32], pad: usize) -> Vec<f32> {
    let n = signal.len();
    let mut padded = Vec::with_capacity(n + 2 * pad);
    for i in (1..=pad.min(n - 1)).rev() {
        padded.push(signal[i]);
    }
    padded.extend_from_slice(signal);
    let right_start = if n > pad { n - pad - 1 } else { 0 };
    for i in (right_start..n - 1).rev() {
        padded.push(signal[i]);
    }
    padded
}

// ─── STFT ───────────────────────────────────────────────────────────────────

/// Compute power spectrogram (magnitude squared) from a pre-padded signal.
/// Returns flat column-major array `[n_freqs × n_frames]` and the frame count.
fn compute_power_stft(
    signal: &[f32],
    n_fft: usize,
    hop_length: usize,
    window: &[f32],
) -> (Vec<f32>, usize) {
    let n_freqs = n_fft / 2 + 1;
    let n_frames = if signal.len() >= n_fft {
        (signal.len() - n_fft) / hop_length + 1
    } else {
        0
    };

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    // Column-major: power[f * n_frames + t] for frequency f, frame t
    let mut power = vec![0.0f32; n_freqs * n_frames];
    let mut frame_buf = vec![Complex::new(0.0f32, 0.0f32); n_fft];

    for t in 0..n_frames {
        let start = t * hop_length;
        for j in 0..n_fft {
            frame_buf[j] = Complex::new(signal[start + j] * window[j], 0.0);
        }
        fft.process(&mut frame_buf);
        for f in 0..n_freqs {
            let re = frame_buf[f].re;
            let im = frame_buf[f].im;
            power[f * n_frames + t] = re * re + im * im;
        }
    }

    (power, n_frames)
}

// ─── Mel filterbank ─────────────────────────────────────────────────────────

/// Create mel filterbank matrix, flat `[num_mels × n_freqs]`.
/// Uses the Slaney mel scale with a linear region below 1000 Hz.
/// Matches `librosa.filters.mel(sr, n_fft, n_mels, htk=True, norm="slaney")`
/// and the candle reference implementation.
fn create_mel_filterbank(
    num_mels: usize,
    n_fft: usize,
    sample_rate: u32,
    fmin: f64,
    fmax: f64,
) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let sr = sample_rate as f64;

    // Slaney mel scale (linear < 1000 Hz, logarithmic above)
    let f_sp = 200.0 / 3.0; // ~66.67 Hz/mel in the linear region
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;

    let hz_to_mel = |f: f64| -> f64 {
        if f < min_log_hz {
            f / f_sp
        } else {
            min_log_mel + (f / min_log_hz).ln() / logstep
        }
    };

    let mel_to_hz = |m: f64| -> f64 {
        if m < min_log_mel {
            f_sp * m
        } else {
            min_log_hz * (logstep * (m - min_log_mel)).exp()
        }
    };

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    let filter_freqs: Vec<f64> = (0..num_mels + 2)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f64 / (num_mels + 1) as f64;
            mel_to_hz(mel)
        })
        .collect();

    let all_freqs: Vec<f64> = (0..n_freqs)
        .map(|j| j as f64 * sr / n_fft as f64)
        .collect();

    let f_diff: Vec<f64> = filter_freqs.windows(2).map(|w| w[1] - w[0]).collect();

    let mut filters = vec![0.0f32; num_mels * n_freqs];

    for j in 0..n_freqs {
        for i in 0..num_mels {
            let down = (all_freqs[j] - filter_freqs[i]) / f_diff[i];
            let up = (filter_freqs[i + 2] - all_freqs[j]) / f_diff[i + 1];
            let val = down.min(up).max(0.0);
            filters[i * n_freqs + j] = val as f32;
        }
    }

    // Slaney normalization: area of each triangular filter = 1.0
    for i in 0..num_mels {
        let enorm = 2.0 / (filter_freqs[i + 2] - filter_freqs[i]);
        for j in 0..n_freqs {
            filters[i * n_freqs + j] *= enorm as f32;
        }
    }

    filters
}

// ─── WAV loading ────────────────────────────────────────────────────────────

/// Load a WAV file and resample to 16kHz mono f32.
pub fn load_wav_samples(wav_path: &str) -> anyhow::Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(wav_path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = 2.0_f32.powi(spec.bits_per_sample as i32 - 1);
            reader
                .samples::<i32>()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| anyhow!("WAV read error: {e}"))?
                .into_iter()
                .map(|s| s as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow!("WAV read error: {e}"))?,
    };

    let mono_samples: Vec<f32> = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
            .collect()
    } else {
        samples
    };

    resample_with_rubato(&mono_samples, spec.sample_rate, 16_000)
}

fn resample_with_rubato(samples: &[f32], src_rate: u32, dst_rate: u32) -> anyhow::Result<Vec<f32>> {
    if src_rate == dst_rate {
        return Ok(samples.to_vec());
    }

    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
        WindowFunction,
    };

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        dst_rate as f64 / src_rate as f64,
        2.0,
        params,
        samples.len(),
        1,
    )?;

    let output = resampler.process(&[samples.to_vec()], None)?;
    Ok(output.into_iter().next().unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window_edges() {
        let w = hann_window(400);
        assert_eq!(w.len(), 400);
        assert!(w[0].abs() < 1e-6);
        assert!(w[399].abs() < 1e-6);
    }

    #[test]
    fn test_reflection_pad_basic() {
        let signal = vec![1.0f32, 2.0, 3.0];
        let padded = reflection_pad(&signal, 1);
        assert_eq!(padded, vec![2.0f32, 1.0, 2.0, 3.0, 2.0]);
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let filters = create_mel_filterbank(128, 400, 16000, 0.0, 8000.0);
        assert_eq!(filters.len(), 128 * 201);
        assert!(filters.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_mel_spectrogram_shape() {
        let mel = MelSpectrogram::new(400, 160, 128, 16000);
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let spec = mel.compute(&samples);
        assert_eq!(spec.len(), 128);
        assert!(spec[0].len() > 0);
    }
}
