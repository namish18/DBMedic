# Tiny Aya Global — Setup Guide

## Model Information

| Field | Value |
|---|---|
| Model ID | `CohereLabs/tiny-aya-global` |
| Provider | Cohere Labs |
| HF Hub | https://huggingface.co/CohereLabs/tiny-aya-global |
| Access | Gated (requires HF account + agreement) |
| Architecture | Decoder-only causal LM |
| Use | Instruction following, tool-augmented reasoning |

## Prerequisites

### 1. Accept Model Access

The model is **gated** on Hugging Face. You must:

1. Create or log in to a Hugging Face account at https://huggingface.co
2. Navigate to https://huggingface.co/CohereLabs/tiny-aya-global
3. Click **"Agree and access repository"** (or similar gated access button)
4. Wait for access approval (usually immediate for research use)

> [!IMPORTANT]
> Without completing gated access, model download will fail with a 403 error.

### 2. Create a Hugging Face Access Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with **read** permission
3. Copy the token value (starts with `hf_`)

### 3. Set HF_TOKEN in .env

Add to your `.env` file (never commit this file):

```
HF_TOKEN=hf_your_token_here
```

> [!CAUTION]
> Never hardcode your HF_TOKEN in source files. Never pass it on the command line. The agent code will never log your token.

## Hardware Requirements

| Configuration | RAM | GPU VRAM | Speed |
|---|---|---|---|
| CPU only | 8+ GB | None | ~30-120s/response |
| GPU (float16) | 8+ GB | 4+ GB | ~2-10s/response |
| GPU (float32) | 16+ GB | 8+ GB | ~5-20s/response |

The model uses `device_map="auto"` by default, which automatically uses GPU if available.

## Installing Python Dependencies

```powershell
# Activate virtualenv
.venv\Scripts\Activate.ps1

# Install Phase 3 dependencies
pip install transformers>=4.40.0 accelerate>=0.27.0 torch>=2.1.0 huggingface_hub>=0.20.0
```

Or install from requirements.txt (after updating):
```powershell
pip install -r requirements.txt
```

> [!NOTE]
> `torch` installation varies by platform. For Windows with CUDA:
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

## Verifying Model Access

Test that your HF_TOKEN and gated access are working:

```powershell
.venv\Scripts\python.exe -c "
from huggingface_hub import model_info
import os
info = model_info('CohereLabs/tiny-aya-global', token=os.getenv('HF_TOKEN'))
print('Model access confirmed:', info.id)
print('SHA:', info.sha[:8])
"
```

Expected output:
```
Model access confirmed: CohereLabs/tiny-aya-global
SHA: <8-char commit hash>
```

## Loading the Model

```python
from agent.core.model_client import TinyAyaClient

# Initialize client
client = TinyAyaClient(
    model_id="CohereLabs/tiny-aya-global",
    generation_config={
        "max_new_tokens": 512,
        "do_sample": False,          # Deterministic for baseline
        "repetition_penalty": 1.1,
    },
    device_map="auto",
)

# Load (downloads ~2–4 GB on first run, then cached)
client.load()

print(f"Model loaded: {client.model_id}")
print(f"Revision: {client.model_revision}")
```

## First-Run Model Download

On first use, the model weights (~2–4 GB) are downloaded to the HF cache:
- **Windows**: `C:\Users\<user>\.cache\huggingface\hub`

Subsequent runs use the cached weights.

## Generation Configuration

| Parameter | Default | Notes |
|---|---|---|
| `max_new_tokens` | 512 | Maximum tokens to generate |
| `do_sample` | False | Deterministic for baseline evaluation |
| `temperature` | 1.0 | Ignored when do_sample=False |
| `repetition_penalty` | 1.1 | Reduces repetitive outputs |

For stochastic sampling (e.g., testing robustness):
```python
client = TinyAyaClient(generation_config={
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.3,
    "top_p": 0.9,
})
```

## Chat Template

Tiny Aya Global uses an official chat template accessible via:

```python
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

The `TinyAyaClient.format_messages()` method handles this automatically, with a fallback formatter if the template is unavailable.

## Troubleshooting

### 403 Forbidden on model download
- You have not accepted gated access on HF Hub
- Your HF_TOKEN doesn't have read permission
- Token is expired — generate a new one

### CUDA out of memory
- Reduce `max_new_tokens`
- Use `device_map="cpu"` to force CPU inference

### ImportError: No module named 'transformers'
```powershell
pip install transformers accelerate torch huggingface_hub
```

### Slow inference on CPU
- Expected: 30–120 seconds per response on CPU
- Use GPU if available for practical evaluation speeds

