//! Shared raw-socket mock-upstream primitives for integration tests.
//!
//! Tests assert on the bytes the proxy puts on the wire, so the upstream is a
//! dumb TCP peer rather than a real HTTP server that would re-canonicalize the
//! request. The fiddly part — finding the header terminator and draining the
//! body (Content-Length or chunked) — lives here once.
// Each integration-test binary compiles this module separately and uses only a
// subset of these helpers, so unused-per-binary is expected.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Bind an ephemeral loopback listener, returning it and its port.
pub async fn bind() -> (TcpListener, u16) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    (listener, port)
}

/// Read one read's worth and discard it — enough to unblock a client that is
/// waiting for its small upload to be consumed before we respond.
pub async fn read_once(conn: &mut TcpStream) {
    let mut buf = [0u8; 4096];
    let _ = conn.read(&mut buf).await;
}

/// Read a complete HTTP/1.1 request and return its header block (everything
/// before the blank line). The body is drained but discarded; framing is taken
/// from `Content-Length`, else chunked transfer-encoding. Memory stays bounded
/// regardless of body size: the body is counted, never accumulated.
pub async fn read_request(conn: &mut TcpStream) -> std::io::Result<String> {
    let mut buf = vec![0u8; 64 * 1024];

    // Accumulate only until the header terminator (headers are small).
    let mut header = Vec::new();
    let (head_end, spill) = loop {
        let n = conn.read(&mut buf).await?;
        if n == 0 {
            return Ok(String::from_utf8_lossy(&header).into_owned());
        }
        header.extend_from_slice(&buf[..n]);
        if let Some(pos) = header.windows(4).position(|w| w == b"\r\n\r\n") {
            let end = pos + 4;
            break (end, header[end..].to_vec());
        }
    };
    let head = String::from_utf8_lossy(&header[..head_end - 4]).into_owned();

    let lower = head.to_ascii_lowercase();
    let content_len: Option<usize> = lower
        .lines()
        .find_map(|l| l.strip_prefix("content-length:"))
        .and_then(|v| v.trim().parse().ok());
    let chunked = lower
        .lines()
        .any(|l| l.starts_with("transfer-encoding:") && l.contains("chunked"));

    match content_len {
        Some(len) => {
            let mut got = spill.len();
            while got < len {
                let n = conn.read(&mut buf).await?;
                if n == 0 {
                    break;
                }
                got += n;
            }
        }
        None if chunked => {
            // Keep only the trailing bytes between reads so the terminator is
            // still spotted when it straddles a read boundary.
            let mut recent = spill;
            while !recent.windows(5).any(|w| w == b"0\r\n\r\n") {
                let keep = recent.len().min(4);
                recent.drain(..recent.len() - keep);
                let n = conn.read(&mut buf).await?;
                if n == 0 {
                    break;
                }
                recent.extend_from_slice(&buf[..n]);
            }
        }
        None => {}
    }
    Ok(head)
}

async fn send(conn: &mut TcpStream, status_line: &str, body: &str, close: bool) {
    let conn_hdr = if close { "Connection: close\r\n" } else { "" };
    let resp = format!(
        "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\n{conn_hdr}Content-Length: {}\r\n\r\n{body}",
        body.len(),
    );
    let _ = conn.write_all(resp.as_bytes()).await;
}

/// Write a `200 OK` JSON response.
pub async fn write_ok(conn: &mut TcpStream, body: &str) {
    send(conn, "200 OK", body, false).await;
}

/// Write a `200 OK` JSON response and signal `Connection: close`.
pub async fn write_ok_close(conn: &mut TcpStream, body: &str) {
    send(conn, "200 OK", body, true).await;
}

/// Write a JSON response with a custom status line (e.g. `"503 Service Unavailable"`).
pub async fn write_status(conn: &mut TcpStream, status_line: &str, body: &str) {
    send(conn, status_line, body, false).await;
}
