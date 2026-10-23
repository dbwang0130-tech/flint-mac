//! 最小 HTTP/1.1：只服务本机的 OpenAI 兼容客户端（pi 等）。每个连接一个线程，响应后关闭连接。

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}

pub fn read_request(stream: &TcpStream) -> std::io::Result<Request> {
    let mut r = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    r.read_line(&mut line)?;
    let mut it = line.split_whitespace();
    let method = it.next().unwrap_or("").to_string();
    let path = it.next().unwrap_or("").to_string();
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                len = v.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = vec![0; len];
    r.read_exact(&mut body)?;
    Ok(Request { method, path, body })
}

pub fn respond(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()
}

pub fn respond_json(stream: &mut TcpStream, status: u16, v: &serde_json::Value) -> std::io::Result<()> {
    respond(stream, status, "application/json", v.to_string().as_bytes())
}

/// Server-Sent Events：写响应头后逐条发送 `data: …`，以关闭连接结束
pub struct Sse<'a> {
    stream: &'a mut TcpStream,
}

impl<'a> Sse<'a> {
    pub fn start(stream: &'a mut TcpStream) -> std::io::Result<Sse<'a>> {
        stream.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
        )?;
        stream.flush()?;
        Ok(Sse { stream })
    }

    pub fn send(&mut self, data: &str) -> std::io::Result<()> {
        write!(self.stream, "data: {data}\n\n")?;
        self.stream.flush()
    }
}
