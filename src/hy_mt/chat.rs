use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Format messages into the HunYuan chat template string.
///
/// Template (ported from chat_template.jinja):
/// - If system message: `<｜hy_begin▁of▁sentence｜>{system}<｜hy_place▁holder▁no▁3｜>`
/// - Else: `<｜hy_begin▁of▁sentence｜>`
/// - User: `<｜hy_User｜>{content}`
/// - Assistant: `<｜hy_Assistant｜>{content}<｜hy_place▁holder▁no▁2｜>`
/// - End: `<｜hy_Assistant｜>` (generation prompt)
pub fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    let (system_msg, loop_messages) =
        if messages.first().map(|m| m.role.as_str()) == Some("system") {
            prompt.push_str("<｜hy_begin▁of▁sentence｜>");
            prompt.push_str(&messages[0].content);
            prompt.push_str("<｜hy_place▁holder▁no▁3｜>");
            (&messages[..1], &messages[1..])
        } else {
            prompt.push_str("<｜hy_begin▁of▁sentence｜>");
            (&messages[..0], messages)
        };

    for msg in loop_messages {
        match msg.role.as_str() {
            "user" => {
                prompt.push_str("<｜hy_User｜>");
                prompt.push_str(&msg.content);
            }
            "assistant" => {
                prompt.push_str("<｜hy_Assistant｜>");
                prompt.push_str(&msg.content);
                prompt.push_str("<｜hy_place▁holder▁no▁2｜>");
            }
            _ => {}
        }
    }

    // Add generation prompt
    prompt.push_str("<｜hy_Assistant｜>");

    // Suppress unused warning
    let _ = system_msg;

    prompt
}
