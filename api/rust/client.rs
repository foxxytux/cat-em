// CodeAgent-RWKV Rust client
// Build: rustc client.rs --edition 2021 (or cargo init && cargo add reqwest serde serde_json tokio)
// Better: Add to Cargo.toml: reqwest = { version = "0.12", features = ["json"] }, serde = { version = "1", features = ["derive"] }, serde_json = "1", tokio = { version = "1", features = ["full"] }

use serde::{Deserialize, Serialize};
use std::env;

const API_URL: &str = "http://127.0.0.1:8080/v1/chat";

#[derive(Serialize)]
struct ChatRequest {
    prompt: String,
    system: String,
    max_tokens: u32,
    temperature: f64,
    top_p: f64,
}

#[derive(Deserialize, Debug)]
struct Usage {
    tokens: u32,
    time: f64,
}

#[derive(Deserialize, Debug)]
struct ChatResponse {
    text: String,
    usage: Usage,
}

async fn chat(prompt: &str) -> Result<ChatResponse, reqwest::Error> {
    let client = reqwest::Client::new();
    let req = ChatRequest {
        prompt: prompt.to_string(),
        system: "You are a helpful coding assistant. Think step by step.".to_string(),
        max_tokens: 512,
        temperature: 0.7,
        top_p: 0.9,
    };

    client
        .post(API_URL)
        .json(&req)
        .send()
        .await?
        .json::<ChatResponse>()
        .await
}

#[tokio::main]
async fn main() {
    let prompt = env::args()
        .skip(1)
        .collect::<Vec<_>>()
        .join(" ");

    let prompt = if prompt.is_empty() {
        "Write a hello world in Python".to_string()
    } else {
        prompt
    };

    println!("CodeAgent > {}\n", prompt);

    match chat(&prompt).await {
        Ok(resp) => {
            println!("{}", resp.text);
            println!("\n[{} tokens, {:.1}s]", resp.usage.tokens, resp.usage.time);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}
