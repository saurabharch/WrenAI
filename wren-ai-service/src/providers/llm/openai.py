import logging
import os
from typing import Any, Dict, List, Optional

import backoff
import openai
from haystack import component
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.utils.auth import Secret
from openai import OpenAI

from src.core.provider import LLMProvider
from src.providers.loader import provider

logger = logging.getLogger("wren-ai-service")

OPENAI_BASE_URL = "https://api.openai.com/v1"
GENERATION_MODEL_NAME = "gpt-3.5-turbo"
GENERATION_MODEL_KWARGS = {
    "temperature": 0,
    "n": 1,
    "max_tokens": 4096,
    "response_format": {"type": "json_object"},
}
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
EMBEDDING_MODEL_DIMENSION = 3072


@component
class CustomOpenAIGenerator(OpenAIGenerator):
    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=60, max_tries=3)
    def run(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None):
        logger.debug(f"Running OpenAI generator with prompt: {prompt}")
        return super(CustomOpenAIGenerator, self).run(
            prompt=prompt, generation_kwargs=generation_kwargs
        )


@provider("openai")
class OpenAILLMProvider(LLMProvider):
    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        base_url: str = os.getenv("OPENAI_BASE_URL") or OPENAI_BASE_URL,
        generation_model: str = os.getenv("GENERATION_MODEL") or GENERATION_MODEL_NAME,
        embedding_model: str = os.getenv("EMBEDDING_MODEL") or EMBEDDING_MODEL_NAME,
        embedding_model_dim: int = int(os.getenv("EMBEDDING_MODEL_DIMENSION", 0))
        or EMBEDDING_MODEL_DIMENSION,
    ):
        def _verify_api_key(api_key: str, base_url: str) -> None:
            """
            this is a temporary solution to verify that the required environment variables are set
            """
            OpenAI(api_key=api_key, base_url=base_url).models.list()

        _verify_api_key(api_key.resolve_value(), base_url)
        logger.info(f"Using OpenAI Generation Model: {generation_model}")
        self._api_key = api_key
        self._base_url = base_url
        self._generation_model = generation_model
        self._embedding_model = embedding_model
        self._embedding_model_dim = embedding_model_dim

    def get_generator(
        self,
        model_kwargs: Optional[Dict[str, Any]] = GENERATION_MODEL_KWARGS,
        system_prompt: Optional[str] = None,
    ):
        def _get_generation_kwargs(
            model_kwargs: Optional[Dict[str, Any]] = GENERATION_MODEL_KWARGS,
            base_url: str = OPENAI_BASE_URL,
        ):
            if base_url == OPENAI_BASE_URL:
                return model_kwargs
            elif model_kwargs != GENERATION_MODEL_KWARGS:
                return model_kwargs
            return None

        return CustomOpenAIGenerator(
            api_key=self._api_key,
            model=self._generation_model,
            system_prompt=system_prompt,
            generation_kwargs=_get_generation_kwargs(model_kwargs, self._base_url),
        )

    def get_text_embedder(self):
        return OpenAITextEmbedder(
            api_key=self._api_key,
            api_base_url=self._base_url,
            model=self._embedding_model,
            dimensions=self._embedding_model_dim,
        )

    def get_document_embedder(self):
        return OpenAIDocumentEmbedder(
            api_key=self._api_key,
            api_base_url=self._base_url,
            model=self._embedding_model,
            dimensions=self._embedding_model_dim,
        )
