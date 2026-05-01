# CAT-EM: Coding and Agentic Trained — Edge Model

> RWKV-7 0.4B continued pre-training on code, agentic reasoning, and tool-use data with thinking format.

**Status:** Training 74.5% complete (14910/20000 steps) · Checkpoint soon

## Model

| | |
|---|---|
| Base | `SmerkyG/RWKV7-Goose-0.4B-Pile-HF` |
| Architecture | RWKV-7 (24 layers, 1024 dim) |
| Parameters | 421M |
| Format | `User:` → `Thinking...` → `Answer:` |
| Datasets | 11 code + agent + reasoning datasets |
| Training | 20K steps, 327M tokens, ~8h on RTX 5070 Ti |
| Context | 4096 (extensible to 128K) |

## Quick Start

```bash
git clone https://github.com/foxxytux/cat-em
cd cat-em

# API server
pip install fastapi uvicorn pydantic
python3 api/python/server.py
# → http://127.0.0.1:8080/docs

# CLI chatbot
python3 chatbot/cli.py
```

## API

```bash
# Chat
curl -X POST http://127.0.0.1:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write a Python function that sorts a list"}'

# Health
curl http://127.0.0.1:8080/health
```

## Chatbot Commands

```
>>> Write a function to reverse a string
>>> /tokens 512    # set max response length
>>> /temp 0.8      # set temperature
>>> /clear         # reset history
>>> /quit
```

Output uses colored blocks: cyan = thinking, green = answer.

## Clients

| Language | File | Build |
|---|---|---|
| Python | `api/python/server.py` | `pip install fastapi uvicorn` |
| Go | `api/go/client.go` | `go run client.go` |
| C++ | `api/cpp/client.cpp` | `g++ -o codeagent client.cpp -lcurl` |
| Rust | `api/rust/client.rs` | `cargo init && cargo add reqwest serde serde_json tokio` |
| JSON | `api/json/openapi.yaml` | OpenAPI 3.0 spec |

## Install as Service

```bash
bash systemd/install.sh
systemctl --user start codeagent
journalctl --user -u codeagent -f
```

## Training

```bash
# Continued pre-training
python3 train_phase.py --phase 0

# Context extension (requires prior checkpoint)
python3 train_phase.py --phase 1  # 4K
python3 train_phase.py --phase 2  # 16K
python3 train_phase.py --phase 3  # 64K
python3 train_phase.py --phase 4  # 128K
```

## License

MIT
