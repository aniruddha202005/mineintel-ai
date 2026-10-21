"""Chat model factory. The provider is chosen through configuration."""

from langchain_core.language_models import BaseChatModel

from app.core.config import Settings


def build_llm(settings: Settings) -> BaseChatModel:
    """Create the chat model for the configured provider.

    ``openai_compatible`` works with any server that speaks the OpenAI chat
    API (Groq, OpenRouter, the Hugging Face router, a local Ollama or vLLM
    endpoint). ``ollama`` uses the native Ollama client.
    """
    provider = settings.llm_provider.lower()
    if provider == "openai_compatible":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key or "not-set",
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=60,
            max_retries=2,
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
