//! 增量解析 Flint 的输出：`<think>…</think>` → reasoning_content，`<tool_call>…</tool_call>` → tool_calls，其余 → content。
//!
//! 与官方 FlintReasoningDetector 一致：不预设本轮处于推理中，只在模型自己写出 `<think>` 时进入推理。
//! 工具调用格式：`<tool_call>名<arg_key>k</arg_key><arg_value>v</arg_value>…</tool_call>`；
//! v 为字符串原文或 JSON（模板渲染时非字符串值用 tojson），按工具 schema 判断类型。

use serde_json::{Map, Value as Json};

#[derive(Debug, Clone)]
pub enum Event {
    Reasoning(String),
    Content(String),
    ToolCall { name: String, arguments: String },
}

#[derive(PartialEq)]
enum State {
    Start,
    Reasoning,
    Content,
    ToolCall,
}

pub struct OutputParser {
    state: State,
    buf: String,
    tool_types: Map<String, Json>, // 工具名 → { 参数名: 类型 }
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";

/// buf 末尾若是 tag 的真前缀，返回该前缀长度（这部分先不输出）
fn partial_suffix(buf: &str, tag: &str) -> usize {
    for n in (1..tag.len()).rev() {
        if buf.len() >= n && buf.is_char_boundary(buf.len() - n) && tag.starts_with(&buf[buf.len() - n..]) {
            return n;
        }
    }
    0
}

impl OutputParser {
    pub fn new(tools: Option<&Json>) -> OutputParser {
        let mut tool_types = Map::new();
        if let Some(Json::Array(ts)) = tools {
            for t in ts {
                let f = t.get("function").unwrap_or(t);
                let (Some(name), Some(props)) = (
                    f.get("name").and_then(|n| n.as_str()),
                    f.pointer("/parameters/properties").and_then(|p| p.as_object()),
                ) else { continue };
                let types: Map<String, Json> = props
                    .iter()
                    .map(|(k, v)| (k.clone(), v.get("type").cloned().unwrap_or(Json::Null)))
                    .collect();
                tool_types.insert(name.to_string(), Json::Object(types));
            }
        }
        OutputParser { state: State::Start, buf: String::new(), tool_types }
    }

    fn parse_tool(&self, inner: &str) -> Event {
        let (name, mut rest) = match inner.find("<arg_key>") {
            Some(i) => (inner[..i].trim().to_string(), &inner[i..]),
            None => (inner.trim().to_string(), ""),
        };
        let types = self.tool_types.get(&name).and_then(|t| t.as_object());
        let mut args = Map::new();
        while let Some(k0) = rest.find("<arg_key>") {
            let after = &rest[k0 + 9..];
            let Some(k1) = after.find("</arg_key>") else { break };
            let key = after[..k1].trim().to_string();
            let after = &after[k1 + 10..];
            let Some(v0) = after.find("<arg_value>") else { break };
            let after = &after[v0 + 11..];
            let Some(v1) = after.find("</arg_value>") else { break };
            let raw = &after[..v1];
            let is_string = types
                .and_then(|t| t.get(&key))
                .map(|ty| ty.as_str() == Some("string"))
                .unwrap_or(false);
            let val = if is_string {
                Json::String(raw.to_string())
            } else {
                serde_json::from_str(raw).unwrap_or_else(|_| Json::String(raw.to_string()))
            };
            args.insert(key, val);
            rest = &after[v1 + 12..];
        }
        Event::ToolCall { name, arguments: Json::Object(args).to_string() }
    }

    /// 喂入新生成的文本，返回可以立即发出的事件
    pub fn push(&mut self, text: &str) -> Vec<Event> {
        self.buf.push_str(text);
        self.drain(false)
    }

    /// 生成结束：把剩余内容全部输出
    pub fn finish(&mut self) -> Vec<Event> {
        self.drain(true)
    }

    fn drain(&mut self, eof: bool) -> Vec<Event> {
        let mut ev = Vec::new();
        loop {
            match self.state {
                State::Start => {
                    let b = self.buf.trim_start();
                    if b.starts_with(THINK_OPEN) {
                        let skip = self.buf.len() - b.len() + THINK_OPEN.len();
                        self.buf.drain(..skip);
                        self.state = State::Reasoning;
                    } else if !eof && (b.is_empty() || THINK_OPEN.starts_with(b)) {
                        return ev;  // 还可能是 <think> 的开头
                    } else {
                        self.state = State::Content;
                    }
                }
                State::Reasoning => {
                    if let Some(i) = self.buf.find(THINK_CLOSE) {
                        if i > 0 {
                            ev.push(Event::Reasoning(self.buf[..i].to_string()));
                        }
                        self.buf.drain(..i + THINK_CLOSE.len());
                        self.state = State::Content;
                        continue;
                    }
                    let keep = if eof { 0 } else { partial_suffix(&self.buf, THINK_CLOSE) };
                    let n = self.buf.len() - keep;
                    if n > 0 {
                        ev.push(Event::Reasoning(self.buf[..n].to_string()));
                        self.buf.drain(..n);
                    }
                    return ev;
                }
                State::Content => {
                    if let Some(i) = self.buf.find(TOOL_OPEN) {
                        if i > 0 {
                            ev.push(Event::Content(self.buf[..i].to_string()));
                        }
                        self.buf.drain(..i + TOOL_OPEN.len());
                        self.state = State::ToolCall;
                        continue;
                    }
                    let keep = if eof { 0 } else { partial_suffix(&self.buf, TOOL_OPEN) };
                    let n = self.buf.len() - keep;
                    if n > 0 {
                        ev.push(Event::Content(self.buf[..n].to_string()));
                        self.buf.drain(..n);
                    }
                    return ev;
                }
                State::ToolCall => {
                    if let Some(i) = self.buf.find(TOOL_CLOSE) {
                        let inner = self.buf[..i].to_string();
                        self.buf.drain(..i + TOOL_CLOSE.len());
                        ev.push(self.parse_tool(&inner));
                        self.state = State::Content;
                        continue;
                    }
                    if eof && !self.buf.trim().is_empty() {
                        // 未闭合的工具调用（被 max_tokens 截断）：尽量解析
                        let inner = std::mem::take(&mut self.buf);
                        ev.push(self.parse_tool(&inner));
                    }
                    return ev;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(chunks: &[&str], tools: Option<&Json>) -> Vec<String> {
        let mut p = OutputParser::new(tools);
        let mut ev = Vec::new();
        for c in chunks {
            ev.extend(p.push(c));
        }
        ev.extend(p.finish());
        // 合并相邻同类事件，便于比较
        let mut out: Vec<String> = Vec::new();
        for e in ev {
            let (tag, s) = match e {
                Event::Reasoning(s) => ("R", s),
                Event::Content(s) => ("C", s),
                Event::ToolCall { name, arguments } => ("T", format!("{name} {arguments}")),
            };
            match out.last_mut() {
                Some(last) if tag != "T" && last.starts_with(tag) => last.push_str(&s),
                _ => out.push(format!("{tag}:{s}")),
            }
        }
        out
    }

    #[test]
    fn reasoning_then_content() {
        let r = collect(&["<th", "ink>We need", " to think</th", "ink>Answer: 4"], None);
        assert_eq!(r, vec!["R:We need to think", "C:Answer: 4"]);
    }

    #[test]
    fn no_reasoning() {
        assert_eq!(collect(&["Hello", " world"], None), vec!["C:Hello world"]);
    }

    #[test]
    fn tool_call_typed_by_schema() {
        let tools = serde_json::json!([{"type": "function", "function": {"name": "read",
            "parameters": {"properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}}}}]);
        let r = collect(
            &["<think>plan</think>", "Let me read.<tool", "_call>read<arg_key>path</arg_key><arg_value>123</arg_value>",
              "<arg_key>limit</arg_key><arg_value>50</arg_value></tool_call>"],
            Some(&tools),
        );
        assert_eq!(r, vec!["R:plan", "C:Let me read.", r#"T:read {"path":"123","limit":50}"#]);
    }
}
