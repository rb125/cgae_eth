"""
CGAE Model Configurations — aligned with CDCT evaluation models.

Three providers:
  - Azure OpenAI (GPT) via cognitiveservices endpoint
  - Azure AI Foundry (DeepSeek, Mistral, Grok, Phi, Llama, Kimi) via services.ai endpoint
  - AWS Bedrock (Nova, Claude, MiniMax, jury models) via ABSK bearer token
  - Gemma via Modal (self-hosted, OpenAI-compatible)

Environment variables:
  AZURE_API_KEY                - Shared Azure key
  AZURE_OPENAI_API_ENDPOINT    - Azure OpenAI (GPT models)
  FOUNDRY_MODELS_ENDPOINT      - Azure AI Foundry
  AWS_BEARER_TOKEN_BEDROCK     - Bedrock ABSK bearer token
  GEMMA_BASE_URL               - Modal endpoint for Gemma
  GEMMA_API_KEY                - Gemma API key (usually "not-needed")
"""

AVAILABLE_MODELS = [
    # --- Azure OpenAI ---
    {
        "model_name": "gpt-5.4",
        "deployment_name": "gpt-5.4",
        "provider": "azure_openai",
        "api_key_env_var": "AZURE_API_KEY",
        "endpoint_env_var": "AZURE_OPENAI_API_ENDPOINT",
        "api_version": "2025-03-01-preview",
        "family": "OpenAI",
        "tier_assignment": "contestant",
    },
    # --- Azure AI Foundry ---
    {
        "model_name": "DeepSeek-V3.2",
        "deployment_name": "DeepSeek-V3.2",
        "provider": "azure_ai",
        "api_key_env_var": "AZURE_API_KEY",
        "endpoint_env_var": "FOUNDRY_MODELS_ENDPOINT",
        "family": "DeepSeek",
        "tier_assignment": "contestant",
    },
    {
        "model_name": "Mistral-Large-3",
        "deployment_name": "Mistral-Large-3",
        "provider": "azure_ai",
        "api_key_env_var": "AZURE_API_KEY",
        "endpoint_env_var": "FOUNDRY_MODELS_ENDPOINT",
        "family": "Mistral",
        "tier_assignment": "contestant",
    },
    {
        "model_name": "grok-4-20-reasoning",
        "deployment_name": "grok-4-20-reasoning",
        "provider": "azure_ai",
        "api_key_env_var": "AZURE_API_KEY",
        "endpoint_env_var": "FOUNDRY_MODELS_ENDPOINT",
        "family": "xAI",
        "tier_assignment": "contestant",
    },
    {
        "model_name": "Phi-4",
        "deployment_name": "Phi-4",
        "provider": "azure_ai",
        "api_key_env_var": "AZURE_API_KEY",
        "endpoint_env_var": "FOUNDRY_MODELS_ENDPOINT",
        "family": "Microsoft",
        "tier_assignment": "contestant",
    },
    {
        "model_name": "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "deployment_name": "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "provider": "azure_ai",
        "api_key_env_var": "AZURE_API_KEY",
        "endpoint_env_var": "FOUNDRY_MODELS_ENDPOINT",
        "family": "Meta",
        "tier_assignment": "contestant",
    },
    {
        "model_name": "Kimi-K2.5",
        "deployment_name": "Kimi-K2.5",
        "provider": "azure_ai",
        "api_key_env_var": "AZURE_API_KEY",
        "endpoint_env_var": "FOUNDRY_MODELS_ENDPOINT",
        "family": "Moonshot",
        "tier_assignment": "contestant",
    },
    # --- Gemma via Modal ---
    {
        "model_name": "gemma-4-27b-it",
        "deployment_name": "google/gemma-4-26B-A4B-it",
        "provider": "azure_ai",
        "api_key_env_var": "GEMMA_API_KEY",
        "endpoint_env_var": "GEMMA_BASE_URL",
        "family": "Google",
        "tier_assignment": "contestant",
    },
    # --- AWS Bedrock (contestant) ---
    {
        "model_name": "nova-pro",
        "model_id": "amazon.nova-pro-v1:0",
        "provider": "bedrock",
        "region": "us-east-1",
        "family": "Amazon",
        "tier_assignment": "contestant",
    },
    {
        "model_name": "claude-sonnet-4.6",
        "model_id": "us.anthropic.claude-sonnet-4-6",
        "provider": "bedrock",
        "region": "us-east-1",
        "family": "Anthropic",
        "tier_assignment": "contestant",
    },
    {
        "model_name": "MiniMax-M2.5",
        "model_id": "minimax.minimax-m2.5",
        "provider": "bedrock",
        "region": "us-east-1",
        "family": "MiniMax",
        "tier_assignment": "contestant",
    },
    # --- AWS Bedrock (jury — zero family overlap with contestants) ---
    {
        "model_name": "Qwen3-32B",
        "model_id": "qwen.qwen3-32b-v1:0",
        "provider": "bedrock",
        "region": "us-east-1",
        "family": "Alibaba",
        "tier_assignment": "jury",
    },
    {
        "model_name": "GLM-5",
        "model_id": "zai.glm-5",
        "provider": "bedrock",
        "region": "us-east-1",
        "family": "Zhipu",
        "tier_assignment": "jury",
    },
    {
        "model_name": "Nemotron-Super-3-120B",
        "model_id": "nvidia.nemotron-super-3-120b",
        "provider": "bedrock",
        "region": "us-east-1",
        "family": "NVIDIA",
        "tier_assignment": "jury",
    },
]

JURY_MODELS = [m for m in AVAILABLE_MODELS if m["tier_assignment"] == "jury"]
CONTESTANT_MODELS = [m for m in AVAILABLE_MODELS if m["tier_assignment"] != "jury"]


def get_model_config(model_name: str) -> dict:
    for m in AVAILABLE_MODELS:
        if m["model_name"] == model_name:
            return m
    raise KeyError(f"Model '{model_name}' not found in AVAILABLE_MODELS")
