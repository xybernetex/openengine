# LLM Provider Configuration

OpenEngine supports five LLM providers out of the box.  The active provider
is selected via the `LLM_PROVIDER` environment variable.  All providers are
accessed through a single `LLMClient` interface so the rest of the engine is
provider-agnostic.

---

## Quick-start

Copy `.env.example` to `.env` and fill in the variables for your chosen
provider.  Only the variables for your active provider are required.

```bash
cp .env.example .env
# edit .env
```

---

## 1. Cloudflare Workers AI  *(default)*

Cloudflare Workers AI runs inference at the edge with no GPU provisioning
required.  It is the default provider because it requires only a free
Cloudflare account to get started.

**Required environment variables:**

```dotenv
LLM_PROVIDER=cloudflare
CF_ACCOUNT_ID=your_cloudflare_account_id
CF_API_TOKEN=your_cloudflare_api_token
```

**Optional:**

```dotenv
LLM_MODEL=@cf/meta/llama-3.1-8b-instruct
# Default if omitted: @cf/meta/llama-3.1-8b-instruct
```

**Default model:** `@cf/meta/llama-3.1-8b-instruct`

**Useful alternative models:**

| Model ID | Notes |
|---|---|
| `@cf/meta/llama-3.3-70b-instruct-fp8-fast` | Higher quality, more tokens |
| `@cf/mistral/mistral-7b-instruct-v0.2` | Fast, Mistral architecture |
| `@cf/google/gemma-7b-it` | Google Gemma instruction-tuned |

**Notes:**
- `CF_API_TOKEN` needs the **AI** permission (Workers AI: Edit).
- The free tier includes generous monthly token limits suitable for
  development and small-scale production.
- Cloudflare does not require a separate `LLM_API_KEY` variable — use
  `CF_API_TOKEN` instead.

---

## 2. OpenAI

**Required environment variables:**

```dotenv
LLM_PROVIDER=openai
LLM_API_KEY=sk-...your_openai_api_key...
```

**Optional:**

```dotenv
LLM_MODEL=gpt-4o-mini
# Default if omitted: gpt-4o-mini
```

**Default model:** `gpt-4o-mini`

**Useful alternative models:**

| Model ID | Notes |
|---|---|
| `gpt-4o` | Highest capability, higher cost |
| `gpt-4o-mini` | Best cost/quality balance (default) |
| `gpt-4-turbo` | Previous generation flagship |
| `gpt-3.5-turbo` | Fastest, lowest cost |

**Notes:**
- The engine uses the `/v1/chat/completions` endpoint.
- Structured output (JSON mode) is enabled automatically when the engine
  needs a parseable response.
- If you use an Azure OpenAI deployment, set
  `LLM_BASE_URL=https://<your-resource>.openai.azure.com/` and
  `LLM_API_VERSION=2024-02-01`.

---

## 3. Mistral

**Required environment variables:**

```dotenv
LLM_PROVIDER=mistral
LLM_API_KEY=your_mistral_api_key
```

**Optional:**

```dotenv
LLM_MODEL=mistral-small-latest
# Default if omitted: mistral-small-latest
```

**Default model:** `mistral-small-latest`

**Useful alternative models:**

| Model ID | Notes |
|---|---|
| `mistral-large-latest` | Flagship, highest quality |
| `mistral-small-latest` | Good balance (default) |
| `open-mistral-nemo` | Open-weight, 128k context |
| `codestral-latest` | Optimised for code generation |

**Notes:**
- The engine uses the Mistral `/v1/chat/completions` compatible endpoint.
- Mistral supports function calling on `mistral-large-latest` and
  `mistral-small-latest`; the engine's structured prompting works on all
  models regardless.

---

## 4. Anthropic

**Required environment variables:**

```dotenv
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-ant-...your_anthropic_api_key...
```

**Optional:**

```dotenv
LLM_MODEL=claude-3-5-haiku-20241022
# Default if omitted: claude-3-5-haiku-20241022
```

**Default model:** `claude-3-5-haiku-20241022`

**Useful alternative models:**

| Model ID | Notes |
|---|---|
| `claude-opus-4-5` | Highest capability |
| `claude-sonnet-4-5` | Strong balance of quality and speed |
| `claude-3-5-haiku-20241022` | Fastest, lowest cost (default) |

**Notes:**
- The engine uses the `/v1/messages` endpoint (not the legacy completions API).
- Claude's default max output tokens are generous; the engine sets a
  reasonable limit per action type.
- Anthropic imposes rate limits by tier — if you see 529 errors during PPO
  rollouts, switch to a lower-cost tier or use Cloudflare for training.

---

## 5. Gemini (Google)

**Required environment variables:**

```dotenv
LLM_PROVIDER=gemini
LLM_API_KEY=your_google_ai_studio_api_key
```

**Optional:**

```dotenv
LLM_MODEL=gemini-1.5-flash
# Default if omitted: gemini-1.5-flash
```

**Default model:** `gemini-1.5-flash`

**Useful alternative models:**

| Model ID | Notes |
|---|---|
| `gemini-1.5-pro` | Highest quality, 2 M token context |
| `gemini-1.5-flash` | Fast, low cost (default) |
| `gemini-1.5-flash-8b` | Smallest, cheapest |

**Notes:**
- Get an API key from [Google AI Studio](https://aistudio.google.com/).
- The engine targets the `generateContent` REST endpoint.
- Gemini has a generous free tier suitable for development.

---

## Switching providers at runtime

You can override the provider for a single run by passing environment
variables directly:

```bash
LLM_PROVIDER=anthropic LLM_API_KEY=sk-ant-... python examples/run_goal.py
```

---

## Fallback and retry behaviour

The `LLMClient` will automatically retry failed requests up to 3 times with
exponential backoff for transient errors (HTTP 429, 500, 503).  Persistent
errors are surfaced as exceptions that the engine logs and handles gracefully
(the action is marked as failed and the episode continues).

---

## Adding a new provider

To add a provider not listed here, subclass `LLMClient` in
`engine/llm_client.py` and implement `_call(messages, **kwargs) -> str`.
Then add a branch to the factory function `build_llm_client()` and document
the required env vars here.
