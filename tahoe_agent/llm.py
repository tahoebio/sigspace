"""LLM utilities for Tahoe Agent."""

import os
from typing import Literal, Optional

import openai
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

SourceType = Literal[
    "OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Lambda", "Custom"
]


def get_llm(
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.7,
    stop_sequences: Optional[list[str]] = None,
    source: Optional[SourceType] = None,
    base_url: Optional[str] = None,
    api_key: str = "EMPTY",
) -> BaseChatModel:
    """
    Get a language model instance based on the specified model name and source.
    This function supports models from OpenAI, Azure OpenAI, Anthropic, Ollama, Gemini, Lambda Labs, and custom model serving.

    Args:
        model: The model name to use
        temperature: Temperature setting for generation
        stop_sequences: Sequences that will stop generation
        source: Source provider ("OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Lambda", "Custom")
        base_url: Base URL for custom model serving (e.g., "http://localhost:8000/v1")
        api_key: API key for custom LLM

    Returns:
        BaseChatModel: Configured language model instance
    """
    # Auto-detect source from model name if not specified
    if source is None:
        if model.startswith("claude-"):
            source = "Anthropic"
        elif model.startswith("gpt-"):
            source = "OpenAI"
        elif model.startswith("gemini-"):
            source = "Gemini"
        elif base_url is not None and "lambda" in base_url.lower():
            source = "Lambda"
        elif base_url is not None:
            source = "Custom"
        elif "/" in model or any(
            name in model.lower()
            for name in [
                "llama",
                "mistral",
                "qwen",
                "gemma",
                "phi",
                "dolphin",
                "orca",
                "vicuna",
            ]
        ):
            source = "Ollama"
        else:
            raise ValueError(
                "Unable to determine model source. Please specify 'source' parameter."
            )

    # Create appropriate model based on source
    if source == "OpenAI":
        return ChatOpenAI(
            model=model, temperature=temperature, stop_sequences=stop_sequences
        )
    elif source == "AzureOpenAI":
        API_VERSION = "2024-12-01-preview"
        return AzureChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
            azure_deployment=model,
            api_version=API_VERSION,
            temperature=temperature,
        )
    elif source == "Anthropic":
        return ChatAnthropic(
            model_name=model,
            temperature=temperature,
            max_tokens_to_sample=8192,
            stop_sequences=stop_sequences,
        )
    elif source == "Gemini":
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
        )
    elif source == "Ollama":
        return ChatOllama(
            model=model,
            temperature=temperature,
        )
    elif source == "Lambda":
        # Lambda Labs AI endpoints - OpenAI compatible
        # Default to their API endpoint if no base_url provided
        if base_url is None:
            base_url = "https://api.lambdalabs.com/v1"

        # Use API key from environment if not provided
        if api_key == "EMPTY":
            lambda_api_key = os.getenv("LAMBDA_API_KEY")
            if not lambda_api_key:
                raise ValueError(
                    "Lambda API key not found. Set LAMBDA_API_KEY environment variable or provide api_key parameter."
                )
            api_key = lambda_api_key

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )
    elif source == "Custom":
        assert base_url is not None, "base_url must be provided for custom LLMs"
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=8192,
            stop_sequences=stop_sequences,
        )
        llm.client = openai.Client(base_url=base_url, api_key=api_key).chat.completions
        return llm
    else:
        raise ValueError(
            f"Invalid source: {source}. Valid options are 'OpenAI', 'AzureOpenAI', 'Anthropic', 'Gemini', 'Ollama', 'Lambda', or 'Custom'"
        )
