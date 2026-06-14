use rustfft::{FftPlanner, num_complex::Complex32};

pub struct MelSpectrogram {
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    sample_rate: u32,
    hann_window: Vec<f32>,
    mel_filters: Vec<Vec<f32>>,
    center: bool,
}

impl MelSpectrogram {
    pub fn new(n_fft: usize, hop_length: usize, n_mels: usize, sample_rate: u32) -> Self {
        // Periodic Hann window matching Python np.hanning(n_fft + 1)[:-1]
        let hann_window = (0..n_fft)
            .map(|i| {
                let two_pi_n = 2.0 * std::f32::consts::PI * i as f32 / (n_fft as f32);
                0.5 - 0.5 * two_pi_n.cos()
            })
            .collect();

        let mel_filters = Self::create_mel_filterbank(n_mels, n_fft / 2 + 1, sample_rate, n_fft);

        Self {
            n_fft,
            hop_length,
            n_mels,
            sample_rate,
            hann_window,
            mel_filters,
            center: true,
        }
    }

    fn create_mel_filterbank(
        n_mels: usize,
        n_freq_bins: usize,
        sample_rate: u32,
        _n_fft: usize,
    ) -> Vec<Vec<f32>> {
        // Match Python transformers.audio_utils.mel_filter_bank with norm="slaney", mel_scale="slaney"

        let f_min = 0.0;
        let f_max = sample_rate as f32 / 2.0;

        let mel_min = Self::hz_to_mel_slaney(f_min);
        let mel_max = Self::hz_to_mel_slaney(f_max);

        // mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
        let mel_freqs: Vec<f32> = (0..(n_mels + 2))
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();

        // filter_freqs = mel_to_hertz(mel_freqs, mel_scale="slaney")
        let filter_freqs: Vec<f32> = mel_freqs.iter().map(|&m| Self::mel_to_hz_slaney(m)).collect();

        // fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)
        let fft_freqs: Vec<f32> = (0..n_freq_bins)
            .map(|i| (sample_rate as f32 / 2.0) * i as f32 / (n_freq_bins - 1) as f32)
            .collect();

        // _create_triangular_filter_bank(fft_freqs, filter_freqs)
        let mut mel_filters = vec![vec![0.0f32; n_mels]; n_freq_bins]; // [freq_bins, n_mels]

        for i in 0..n_mels {
            let left = filter_freqs[i];
            let center = filter_freqs[i + 1];
            let right = filter_freqs[i + 2];

            for (j, &freq) in fft_freqs.iter().enumerate() {
                if freq >= left && freq <= center {
                    mel_filters[j][i] = (freq - left) / (center - left);
                } else if freq >= center && freq <= right {
                    mel_filters[j][i] = (right - freq) / (right - center);
                }
            }
        }

        // Slaney normalization: enorm = 2.0 / (filter_freqs[2:] - filter_freqs[:-2])
        for i in 0..n_mels {
            let enorm = 2.0 / (filter_freqs[i + 2] - filter_freqs[i]);
            for j in 0..n_freq_bins {
                mel_filters[j][i] *= enorm;
            }
        }

        // Transpose to [n_mels, n_freq_bins] for our computation
        let mut filters = vec![vec![0.0f32; n_freq_bins]; n_mels];
        for i in 0..n_mels {
            for j in 0..n_freq_bins {
                filters[i][j] = mel_filters[j][i];
            }
        }

        filters
    }

    fn hz_to_mel_slaney(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz_slaney(mel: f32) -> f32 {
        700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
    }

    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
    }

    pub fn compute(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let samples = self.resample_if_needed(samples);

        // Do NOT pad to 30 seconds. Python WhisperFeatureExtractor with padding=True
        // uses PaddingStrategy.LONGEST, which for a single audio means no padding.
        let target_len = samples.len();

        let n_frames = if self.center {
            // WhisperFeatureExtractor: pad n_fft/2 on each side, then
            // num_frames = 1 + floor((padded_len - n_fft) / hop_length)
            //            = 1 + floor(original_len / hop_length)
            1 + samples.len() / self.hop_length
        } else {
            (samples.len() - self.n_fft) / self.hop_length + 1
        };
        let n_frames = n_frames.max(1);

        let pad_len = if self.center {
            self.n_fft / 2
        } else {
            0
        };

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.n_fft);

        let n_freq_bins = self.n_fft / 2 + 1;
        let mut spectrogram = vec![vec![0.0f32; n_freq_bins]; n_frames];

        for frame_idx in 0..n_frames {
            let start = frame_idx as isize * self.hop_length as isize - pad_len as isize;
            let mut frame = vec![Complex32::new(0.0, 0.0); self.n_fft];

            for i in 0..self.n_fft {
                let sample_idx = start + i as isize;
                // torch.stft uses ZERO padding (center=True pads n_fft//2 zeros on each side)
                let val = if sample_idx >= 0 && (sample_idx as usize) < samples.len() {
                    samples[sample_idx as usize]
                } else {
                    0.0
                };
                frame[i] = Complex32::new(val * self.hann_window[i], 0.0);
            }

            fft.process(&mut frame);

            for j in 0..n_freq_bins {
                // Python uses power=2.0: real²+imag² (NOT sqrt!)
                let re = frame[j].re;
                let im = frame[j].im;
                spectrogram[frame_idx][j] = re * re + im * im;
            }
        }

        let mut mel_spec = vec![vec![0.0f32; n_frames]; self.n_mels];
        for i in 0..self.n_mels {
            for j in 0..n_frames {
                let mut sum = 0.0;
                for k in 0..n_freq_bins {
                    sum += spectrogram[j][k] * self.mel_filters[i][k];
                }
                mel_spec[i][j] = sum.max(1e-10).log10();
            }
        }

        // Python: log_spec = log_spec[:, :-1] — drop last column
        let n_frames = if n_frames > 1 { n_frames - 1 } else { n_frames };
        for i in 0..self.n_mels {
            mel_spec[i].truncate(n_frames);
        }

        // HuggingFace WhisperFeatureExtractor normalization:
        // log_mel = max(log_mel, log_mel.max() - 8.0)
        // log_mel = (log_mel + 4.0) / 4.0
        let mut global_max = f32::NEG_INFINITY;
        for i in 0..self.n_mels {
            for j in 0..n_frames {
                global_max = global_max.max(mel_spec[i][j]);
            }
        }
        let clamp_min = global_max - 8.0;
        for i in 0..self.n_mels {
            for j in 0..n_frames {
                mel_spec[i][j] = mel_spec[i][j].max(clamp_min);
                mel_spec[i][j] = (mel_spec[i][j] + 4.0) / 4.0;
            }
        }

        mel_spec
    }

    fn resample_if_needed(&self, samples: &[f32]) -> Vec<f32> {
        // Assume input is already 16kHz since hound reads WAV at its native rate.
        // For simplicity, we skip resampling. User should provide 16kHz audio.
        samples.to_vec()
    }

    pub fn compute_from_wav(&self, wav_path: &str) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut reader = hound::WavReader::open(wav_path)?;
        let spec = reader.spec();
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max_val = 2.0_f32.powi(spec.bits_per_sample as i32 - 1);
                reader
                    .samples::<i32>()
                    .map(|s| s.unwrap_or(0) as f32 / max_val)
                    .collect()
            }
            hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
        };

        // Convert to mono if stereo
        let mono_samples: Vec<f32> = if spec.channels == 2 {
            samples
                .chunks(2)
                .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
                .collect()
        } else {
            samples
        };

        // Simple resampling to 16kHz if needed
        let resampled = if spec.sample_rate != 16000 {
            Self::resample(&mono_samples, spec.sample_rate, 16000)
        } else {
            mono_samples
        };

        Ok(self.compute(&resampled))
    }

    fn resample(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
        if src_rate == dst_rate {
            return samples.to_vec();
        }
        let ratio = dst_rate as f64 / src_rate as f64;
        let out_len = (samples.len() as f64 * ratio).ceil() as usize;
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src_idx = i as f64 / ratio;
            let src_idx_floor = src_idx.floor() as usize;
            let src_idx_ceil = (src_idx_floor + 1).min(samples.len() - 1);
            let frac = src_idx - src_idx_floor as f64;
            let val = samples[src_idx_floor] as f64 * (1.0 - frac) + samples[src_idx_ceil] as f64 * frac;
            out.push(val as f32);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_spectrogram_shape() {
        let mel = MelSpectrogram::new(400, 160, 128, 16000);
        // Generate 1 second of 440Hz sine wave at 16kHz
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let spec = mel.compute(&samples);
        assert_eq!(spec.len(), 128); // n_mels
        assert!(spec[0].len() > 0); // some frames
    }
}
