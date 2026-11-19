# ekoi-llm-router

Sequential vLLM Model Orchestration Gateway for serving local Safetensors quantized LLMs and HuggingFace embeddings on an RTX 5090 GPU server.

Drop-in compatible with both **Ollama** (`/chat`, `/api/chat`, `/api/embeddings`, `/api/tags`) and **OpenAI** (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`).

---

## Architecture Highlights

1. **Sequential Single-Model vLLM Worker**:
   - Exactly one vLLM instance runs on GPU at a time (internal port `8001`).
   - Multiple concurrent incoming requests for the *same* model are processed concurrently by vLLM.
   - Incoming requests for a *different* model are queued. Active requests finish before the previous model is cleanly stopped and the new model loaded.
   - Queued requests are clustered by model to prevent unnecessary model-switching thrashing.
   - **Idle Auto-Unloading**: If no requests arrive for 5 minutes (`IDLE_UNLOAD_TIMEOUT=300`), the vLLM process is terminated to free 100% of GPU memory.

2. **Dedicated CPU Embedding Engine**:
   - Embeddings (`nomic-ai/nomic-embed-text-v1.5`) run on CPU via FastEmbed with zero GPU VRAM consumption.
   - Auto-downloads model weights if unavailable.
   - Embeddings are instantaneous (~10ms) and never trigger a vLLM model switch or interrupt running LLMs.

3. **Master Model Catalog**:
   - Manages all quantized local Safetensors models under:
     - `/home/ror-technologies/Quantization/models` (Qwen 2.5 3B, 7B, 14B; FP16, FP8, INT4 AWQ)
     - `/home/ror-technologies/IT-helpdesk-agent/quantized_models` (Llama 3.2 1B, 3B, Meta-Llama 3 8B; AWQ, BNB-4bit, BNB-8bit, GPTQ-4bit, GPTQ-8bit)
     - HuggingFace local models (`meta-llama/Llama-3.2-1B-Instruct`, etc.)
   - Master list exposed dynamically at `GET /api/models` for `platform-services`.

---

## Server Deployment Instructions

### 1. Requirements on GPU Server
- Python 3.10+
- NVIDIA CUDA 12.1+ and drivers
- `vllm >= 0.4.0` installed in the server environment

### 2. Setup Virtual Environment
```bash
git clone <your-repo-url> ekoi-llm-router
cd ekoi-llm-router

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install vllm
```

### 3. Environment Variables (Optional overrides)
Create a `.env` file or export environment variables:
```bash
ROUTER_HOST=0.0.0.0
ROUTER_PORT=8080
VLLM_INTERNAL_PORT=8001
IDLE_UNLOAD_TIMEOUT=300
DEFAULT_LLM_MODEL=Qwen2.5-7B-Instruct_INT4_AWQ
DEFAULT_EMBED_MODEL=nomic-ai/nomic-embed-text-v1.5
QWEN_MODELS_DIR=/home/ror-technologies/Quantization/models
LLAMA_MODELS_DIR=/home/ror-technologies/IT-helpdesk-agent/quantized_models
```

### 4. Running the Router
#### Direct execution:
```bash
python main.py
```

#### Production Systemd Service (`/etc/systemd/system/ekoi-llm-router.service`):
```ini
[Unit]
Description=EKOI vLLM Model Router
After=network.target

[Service]
Type=simple
User=ror-technologies
WorkingDirectory=/home/ror-technologies/ekoi-llm-router
ExecStart=/home/ror-technologies/ekoi-llm-router/.venv/bin/python main.py
Restart=always
RestartSec=5
Environment=ROUTER_PORT=8080
Environment=PATH=/home/ror-technologies/ekoi-llm-router/.venv/bin:/usr/local/cuda/bin:/usr/bin

[Install]
WantedBy=multi-user.target
```
Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ekoi-llm-router
```

---

## API Examples

### 1. Ollama Chat Format (`/chat` or `/api/chat`)
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct_INT4_AWQ",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in one sentence."}
    ],
    "stream": false
  }'
```

### 2. Ollama Embeddings (`/api/embeddings`)
```bash
curl -X POST http://localhost:8080/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-ai/nomic-embed-text-v1.5",
    "prompt": "Hello world"
  }'
```

### 3. List Registered Models (`/api/models`)
```bash
curl http://localhost:8080/api/models
```

### 4. OpenAI Chat Completions (`/v1/chat/completions`)
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct_INT4_AWQ",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### 5. Health & Telemetry (`/health`)
```bash
curl http://localhost:8080/health
```
