# CodeAgent-RWKV7

0.4B RWKV-7 model trained on code + agent + reasoning data with thinking format.

## Quick Start

```bash
# API server
python3 api/python/server.py
# → http://127.0.0.1:8080/docs

# CLI chatbot
python3 chatbot/cli.py
```

## API

```bash
curl -X POST http://127.0.0.1:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write a hello world in Python","max_tokens":200}'
```

## Install as systemd service

```bash
bash systemd/install.sh
systemctl --user start codeagent
```

## Clients

- `api/go/client.go` — Go
- `api/cpp/client.cpp` — C++ (needs libcurl)
- `api/rust/client.rs` — Rust
- `api/python/server.py` — FastAPI server
- `api/json/openapi.yaml` — OpenAPI spec

## Training

Base model: `SmerkyG/RWKV7-Goose-0.4B-Pile-HF`
Continued pre-training on 11 code/agent/reasoning datasets.
Checkpoint available on HuggingFace.

```
python3 train_phase.py --phase 0  # Real training
python3 train_phase.py --phase 1  # Context extension 4K
```
