from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from app.config import settings


def get_llm(
	purpose: Literal["classifier", "synthesizer", "judge"],
	max_tokens: int | None = None,
) -> BaseChatModel:
	"""
	Returns the configured LangChain BaseChatModel for the given purpose.
	Provider and model are read from environment variables.
	Supports: openai, gemini, openrouter

	Args:
		purpose: The purpose of the LLM call (classifier, synthesizer, or judge)
		max_tokens: Optional max output tokens for response limiting
	"""
	provider = settings.LLM_PROVIDER
	if purpose == "classifier":
		model = settings.LLM_CLASSIFIER_MODEL
	elif purpose == "synthesizer":
		model = settings.LLM_SYNTHESIZER_MODEL
	elif purpose == "judge":
		model = settings.LLM_JUDGE_MODEL
	else:
		raise ValueError(f"Unknown LLM purpose: {purpose}")

	if provider == "openai":
		return ChatOpenAI(
			model=model,
			api_key=settings.OPENAI_API_KEY,
			max_tokens=max_tokens,
			temperature=settings.LLM_TEMPERATURE,
		)
	if provider == "gemini":
		return ChatGoogleGenerativeAI(
			model=model,
			google_api_key=settings.GOOGLE_API_KEY,
			max_output_tokens=max_tokens,
			temperature=settings.LLM_TEMPERATURE,
		)
	if provider == "openrouter":
		return ChatOpenAI(
			model=model,
			base_url=settings.OPENROUTER_BASE_URL,
			api_key=settings.OPENROUTER_API_KEY,
			max_tokens=max_tokens,
			temperature=settings.LLM_TEMPERATURE,
		)

	raise ValueError(
		"Invalid LLM_PROVIDER. Expected one of: openai, gemini, openrouter. "
		f"Got: {provider}"
	)


if __name__ == "__main__":
	response = get_llm("classifier").invoke("hello")
	print(response)
