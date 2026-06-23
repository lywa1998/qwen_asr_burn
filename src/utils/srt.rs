use super::vad::VoiceSegment;

/// Write SRT subtitle file from aligned segment timestamps and texts.
pub fn write_srt(segments: &[VoiceSegment], texts: &[String], output_path: &str) -> anyhow::Result<()> {
    let n = segments.len().min(texts.len());
    if n == 0 {
        return Ok(());
    }

    let mut out = String::new();
    for i in 0..n {
        let seg = &segments[i];
        let text = texts[i].trim();
        if text.is_empty() {
            continue;
        }
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!(
            "{} --> {}\n",
            format_srt_time(seg.start_secs),
            format_srt_time(seg.end_secs),
        ));
        out.push_str(text);
        out.push_str("\n\n");
    }

    std::fs::write(output_path, out)?;
    log::info!("Wrote SRT: {output_path} ({} entries)", n);
    Ok(())
}

fn format_srt_time(secs: f64) -> String {
    let h = (secs / 3600.0) as u32;
    let m = ((secs % 3600.0) / 60.0) as u32;
    let s = (secs % 60.0) as u32;
    let ms = ((secs - secs.floor()) * 1000.0).round() as u32;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, ms)
}
