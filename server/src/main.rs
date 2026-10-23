//! flint-server：OpenAI 兼容的 Flint 推理服务（/v1/chat/completions、/v1/models）。
//!
//! - 引擎单会话：请求串行执行（Mutex），跨请求复用 KV（与上次 token 序列的公共前缀不重算）
//! - 输出按 `<think>` / `<tool_call>` 拆成 reasoning_content / content / tool_calls，支持 SSE 流式
//!
//! 用法：flint-server --model <gguf> [--port 8080] [--ctx 32768] [--pool auto|GB|0]

mod chat;
mod ffi;
mod http;
mod parse;
mod sample;

use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value as Json};
use tokenizers::Tokenizer;

use parse::{Event, OutputParser};

/// 结束生成的 token（GGUF 的 eog 列表：<|end_of_text|> <|user|> <|observation|> 等）
const EOG: [u32; 4] = [150001, 150003, 150005, 150018];
/// SWA 窗口 513：回退超过 511 个 token 时环形缓冲里的旧 key 已被覆盖，需要往前多算 512 个
const SWA_SAFE_ROLLBACK: usize = 511;

struct Config {
    model: String,
    port: u16,
    n_ctx: i32,
    pool_gb: f64,
    metallib_dir: String,
    tokenizer: String,
    template: String,
    model_id: String,
}

fn parse_args() -> Config {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let mut c = Config {
        model: String::new(),
        port: 8080,
        n_ctx: 32768,
        pool_gb: 0.0,
        metallib_dir: format!("{root}/build"),
        tokenizer: format!("{root}/ref/flint-hf/tokenizer.json"),
        template: format!("{root}/ref/flint-hf/chat_template.jinja"),
        model_id: "flint".into(),
    };
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let v = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--model" => c.model = v,
            "--port" => c.port = v.parse().expect("--port"),
            "--ctx" => c.n_ctx = v.parse().expect("--ctx"),
            "--pool" => c.pool_gb = if v == "auto" { -1.0 } else { v.parse().expect("--pool") },
            "--metallib-dir" => c.metallib_dir = v,
            "--tokenizer" => c.tokenizer = v,
            "--template" => c.template = v,
            "--model-id" => c.model_id = v,
            other => panic!("unknown argument {other}"),
        }
        i += 2;
    }
    if c.model.is_empty() {
        eprintln!("usage: flint-server --model <gguf> [--port 8080] [--ctx 32768] [--pool auto|GB|0]");
        std::process::exit(2);
    }
    c
}

struct Session {
    engine: ffi::Engine,
    cached: Vec<u32>, // 当前 KV 中的 token 序列
}

struct App {
    cfg: Config,
    tok: Tokenizer,
    tmpl: chat::ChatTemplate,
    session: Mutex<Session>,
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// 准备 KV：复用与上次的公共前缀，只 prefill 剩余部分。返回复用的 token 数
fn prepare(s: &mut Session, ids: &[u32]) -> Result<usize, String> {
    let mut p = common_prefix(&s.cached, ids).min(ids.len() - 1); // 至少重算最后一个 token 以得到 logits
    if s.cached.len() > p && s.cached.len() - p > SWA_SAFE_ROLLBACK {
        p = p.saturating_sub(512);
    }
    s.engine.truncate(p);
    s.cached.truncate(p);
    let t0 = Instant::now();
    s.engine.prefill(&ids[p..])?;
    s.cached.extend_from_slice(&ids[p..]);
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "[flint] prompt {} tokens, reused {}, prefill {} in {:.2}s ({:.0} tok/s)",
        ids.len(),
        p,
        ids.len() - p,
        dt,
        (ids.len() - p) as f64 / dt
    );
    Ok(p)
}

struct Gen {
    reasoning: String,
    content: String,
    tool_calls: Vec<(String, String)>,
    finish: &'static str,
    n_gen: usize,
}

/// 生成循环；每产生一批事件调用 on_event（返回 false 表示客户端断开，停止生成）
fn generate(
    app: &App,
    s: &mut Session,
    req: &Json,
    tools: Option<&Json>,
    mut on_event: impl FnMut(&Event) -> bool,
) -> Result<Gen, String> {
    let max_tokens = req
        .get("max_completion_tokens")
        .or_else(|| req.get("max_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(32768) as usize;
    let temp = req.get("temperature").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let top_p = req.get("top_p").and_then(|v| v.as_f64()).unwrap_or(0.95) as f32;
    let top_k = req.get("top_k").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
    let seed = req.get("seed").and_then(|v| v.as_u64()).unwrap_or_else(|| {
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)
    });
    let mut sampler = sample::Sampler::new(temp, top_k, top_p, seed);
    let mut parser = OutputParser::new(tools);
    let mut g = Gen { reasoning: String::new(), content: String::new(), tool_calls: vec![], finish: "length", n_gen: 0 };
    let mut gen_ids: Vec<u32> = Vec::new();
    let mut emitted = 0usize; // 已解码输出的字节数
    let t0 = Instant::now();
    let mut alive = true;

    let mut handle = |evs: Vec<Event>, g: &mut Gen| -> bool {
        let mut ok = true;
        for e in evs {
            match &e {
                Event::Reasoning(t) => g.reasoning.push_str(t),
                Event::Content(t) => g.content.push_str(t),
                Event::ToolCall { name, arguments } => g.tool_calls.push((name.clone(), arguments.clone())),
            }
            ok &= on_event(&e);
        }
        ok
    };

    while g.n_gen < max_tokens && alive {
        if s.engine.pos() + 1 >= s.engine.n_ctx() {
            g.finish = "length";
            break;
        }
        let t = sampler.sample(s.engine.logits());
        s.engine.decode(t)?; // EOG 也写进 KV：下一轮 prompt 里正好有它（<|user|> / <|observation|>）
        s.cached.push(t);
        g.n_gen += 1;
        if EOG.contains(&t) {
            g.finish = "stop";
            break;
        }
        gen_ids.push(t);
        let text = app.tok.decode(&gen_ids, false).map_err(|e| e.to_string())?;
        if !text.ends_with('\u{FFFD}') && text.len() > emitted && text.is_char_boundary(emitted) {
            let delta = text[emitted..].to_string();
            emitted = text.len();
            let evs = parser.push(&delta);
            alive = handle(evs, &mut g);
        }
    }
    let evs = parser.finish();
    handle(evs, &mut g);
    if !g.tool_calls.is_empty() && g.finish == "stop" {
        g.finish = "tool_calls";
    }
    let dt = t0.elapsed().as_secs_f64();
    eprintln!("[flint] generated {} tokens in {:.2}s ({:.1} tok/s), finish={}", g.n_gen, dt, g.n_gen as f64 / dt, g.finish);
    Ok(g)
}

fn tool_call_json(i: usize, id: &str, name: &str, args: &str) -> Json {
    json!({"index": i, "id": id, "type": "function", "function": {"name": name, "arguments": args}})
}

fn handle_chat(app: &App, stream: &mut TcpStream, body: &[u8]) -> std::io::Result<()> {
    let req: Json = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return http::respond_json(stream, 400, &json!({"error": {"message": e.to_string()}})),
    };
    let tools = match req.get("tool_choice").and_then(|v| v.as_str()) {
        Some("none") => None, // 同 vLLM --exclude-tools-when-tool-choice-none
        _ => req.get("tools").filter(|t| t.as_array().map(|a| !a.is_empty()).unwrap_or(false)),
    };
    let prompt = chat::normalize_messages(req.get("messages").unwrap_or(&Json::Null))
        .and_then(|m| app.tmpl.render(&m, tools));
    let prompt = match prompt {
        Ok(p) => p,
        Err(e) => return http::respond_json(stream, 400, &json!({"error": {"message": e}})),
    };
    let ids: Vec<u32> = match app.tok.encode(prompt.as_str(), false) {
        Ok(enc) => enc.get_ids().to_vec(),
        Err(e) => return http::respond_json(stream, 500, &json!({"error": {"message": e.to_string()}})),
    };
    if std::env::var("FLINT_DUMP_PROMPT").is_ok() {
        eprintln!("[flint] prompt text:\n{prompt}\n[flint] ids: {:?}", &ids[..ids.len().min(64)]);
    }

    let mut s = app.session.lock().unwrap();
    if ids.is_empty() || ids.len() + 1 >= s.engine.n_ctx() {
        return http::respond_json(stream, 400, &json!({"error": {"message": format!(
            "prompt has {} tokens; context is {}", ids.len(), s.engine.n_ctx())}}));
    }
    let reused = match prepare(&mut s, &ids) {
        Ok(p) => p,
        Err(e) => return http::respond_json(stream, 500, &json!({"error": {"message": e}})),
    };

    let id = format!("chatcmpl-{}", now_secs());
    let created = now_secs();
    let model = app.cfg.model_id.clone();
    let streaming = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let usage = |n_gen: usize| {
        json!({"prompt_tokens": ids.len(), "completion_tokens": n_gen, "total_tokens": ids.len() + n_gen,
               "prompt_tokens_details": {"cached_tokens": reused}})
    };

    if !streaming {
        let g = match generate(app, &mut s, &req, tools, |_| true) {
            Ok(g) => g,
            Err(e) => return http::respond_json(stream, 500, &json!({"error": {"message": e}})),
        };
        let mut msg = json!({"role": "assistant", "content": if g.content.is_empty() && !g.tool_calls.is_empty() { Json::Null } else { Json::String(g.content.clone()) }});
        if !g.reasoning.is_empty() {
            msg["reasoning_content"] = Json::String(g.reasoning.clone());
        }
        if !g.tool_calls.is_empty() {
            msg["tool_calls"] = Json::Array(
                g.tool_calls.iter().enumerate()
                    .map(|(i, (n, a))| { let mut t = tool_call_json(i, &format!("call_{created}_{i}"), n, a); t.as_object_mut().unwrap().remove("index"); t })
                    .collect(),
            );
        }
        let resp = json!({"id": id, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": g.finish}], "usage": usage(g.n_gen)});
        return http::respond_json(stream, 200, &resp);
    }

    let include_usage = req.pointer("/stream_options/include_usage").and_then(|v| v.as_bool()).unwrap_or(false);
    let mut sse = http::Sse::start(stream)?;
    let chunk = |delta: Json, finish: Option<&str>| {
        json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}).to_string()
    };
    if sse.send(&chunk(json!({"role": "assistant", "content": ""}), None)).is_err() {
        return Ok(());
    }
    let mut n_tools = 0usize;
    let result = generate(app, &mut s, &req, tools, |e| {
        let delta = match e {
            Event::Reasoning(t) => json!({"reasoning_content": t}),
            Event::Content(t) => json!({"content": t}),
            Event::ToolCall { name, arguments } => {
                let d = json!({"tool_calls": [tool_call_json(n_tools, &format!("call_{created}_{n_tools}"), name, arguments)]});
                n_tools += 1;
                d
            }
        };
        sse.send(&chunk(delta, None)).is_ok()
    });
    match result {
        Ok(g) => {
            let _ = sse.send(&chunk(json!({}), Some(g.finish)));
            if include_usage {
                let _ = sse.send(&json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                                          "choices": [], "usage": usage(g.n_gen)}).to_string());
            }
        }
        Err(e) => {
            let _ = sse.send(&json!({"error": {"message": e}}).to_string());
        }
    }
    let _ = sse.send("[DONE]");
    Ok(())
}

fn handle(app: &App, mut stream: TcpStream) {
    let req = match http::read_request(&stream) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[flint] bad request: {e}");
            return;
        }
    };
    let r = match (req.method.as_str(), req.path.split('?').next().unwrap_or("")) {
        ("GET", "/v1/models") => http::respond_json(
            &mut stream,
            200,
            &json!({"object": "list", "data": [{"id": app.cfg.model_id, "object": "model", "owned_by": "flint"}]}),
        ),
        ("POST", "/v1/chat/completions") => handle_chat(app, &mut stream, &req.body),
        ("GET", "/health") => http::respond(&mut stream, 200, "text/plain", b"ok"),
        _ => http::respond_json(&mut stream, 404, &json!({"error": {"message": "not found"}})),
    };
    if let Err(e) = r {
        eprintln!("[flint] connection error: {e}");
    }
}

fn main() {
    let cfg = parse_args();
    let tok = Tokenizer::from_file(&cfg.tokenizer).expect("load tokenizer");
    let tmpl = chat::ChatTemplate::new(&std::fs::read_to_string(&cfg.template).expect("read chat template"))
        .expect("parse chat template");
    eprintln!("[flint] loading {} (ctx {}, pool {})", cfg.model, cfg.n_ctx, cfg.pool_gb);
    let engine = ffi::Engine::open(&cfg.model, &cfg.metallib_dir, cfg.pool_gb, cfg.n_ctx).expect("open engine");
    let listener = TcpListener::bind(("127.0.0.1", cfg.port)).expect("bind");
    eprintln!("[flint] listening on http://127.0.0.1:{}/v1", cfg.port);
    let app = Arc::new(App { cfg, tok, tmpl, session: Mutex::new(Session { engine, cached: Vec::new() }) });
    for conn in listener.incoming().flatten() {
        let app = app.clone();
        std::thread::spawn(move || handle(&app, conn));
    }
}
