import logging
from typing import Any, List

from haystack import Document, Pipeline

from src.core.pipeline import BasicPipeline
from src.pipelines.ask.components.post_processors import init_generation_post_processor
from src.pipelines.ask.components.prompts import (
    TEXT_TO_SQL_RULES,
    init_text_to_sql_with_followup_prompt_builder,
)
from src.utils import init_providers, load_env_vars, timer
from src.web.v1.services.ask import AskRequest

load_env_vars()
logger = logging.getLogger("wren-ai-service")


class FollowUpGeneration(BasicPipeline):
    def __init__(
        self,
        generator: Any,
    ):
        self._pipeline = Pipeline()
        self._pipeline.add_component(
            "text_to_sql_prompt_builder",
            init_text_to_sql_with_followup_prompt_builder(),
        )
        self._pipeline.add_component("text_to_sql_generator", generator)
        self._pipeline.add_component("post_processor", init_generation_post_processor())

        self._pipeline.connect(
            "text_to_sql_prompt_builder.prompt", "text_to_sql_generator.prompt"
        )
        self._pipeline.connect(
            "text_to_sql_generator.replies", "post_processor.replies"
        )

        super().__init__(self._pipeline)

    @timer
    def run(
        self,
        query: str,
        contexts: List[Document],
        history: AskRequest.AskResponseDetails,
    ):
        logger.info("Ask FollowUpGeneration pipeline is running...")
        return self._pipeline.run(
            {
                "text_to_sql_prompt_builder": {
                    "query": query,
                    "documents": contexts,
                    "history": history,
                    "alert": TEXT_TO_SQL_RULES,
                },
            }
        )


if __name__ == "__main__":
    llm_provider, _ = init_providers()
    followup_generation_pipeline = FollowUpGeneration(
        generator=llm_provider.get_generator(),
    )

    print("generating followup_generation_pipeline.jpg to outputs/pipelines/ask...")
    followup_generation_pipeline.draw(
        "./outputs/pipelines/ask/followup_generation_pipeline.jpg"
    )
