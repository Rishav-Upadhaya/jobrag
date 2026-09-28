import os

# Settings() validates required keys at import time; tests never call the real
# services, so placeholders are enough. Real values from .env still win.
for key, value in {
    "DATABASE_URL": "postgresql://unused:unused@localhost:1/unused",
    "LLAMA_CLOUD_API_KEY": "test",
    "LLM_PROVIDER": "gemini",
    "LLM_SYNTHESIZER_MODEL": "test",
    "LLM_CLASSIFIER_MODEL": "test",
    "LLM_JUDGE_MODEL": "test",
    "JINA_API_KEY": "test",
}.items():
    os.environ.setdefault(key, value)
