import logging
from typing import Dict, List

from haystack import Document, Pipeline

from src.core.llm_provider import LLMProvider
from src.core.pipeline import BasicPipeline
from src.pipelines.ask.components.post_processors import init_generation_post_processor
from src.pipelines.ask.components.prompts import (
    TEXT_TO_SQL_RULES,
    init_sql_correction_prompt_builder,
    text_to_sql_system_prompt,
)
from src.utils import init_providers, timer

logger = logging.getLogger("wren-ai-service")


class SQLCorrection(BasicPipeline):
    def __init__(
        self,
        llm_provider: LLMProvider,
    ):
        self._pipeline = Pipeline()
        self._pipeline.add_component(
            "sql_correction_prompt_builder",
            init_sql_correction_prompt_builder(),
        )
        self._pipeline.add_component(
            "sql_correction_generator",
            llm_provider.get_generator(system_prompt=text_to_sql_system_prompt),
        )
        self._pipeline.add_component("post_processor", init_generation_post_processor())

        self._pipeline.connect(
            "sql_correction_prompt_builder.prompt", "sql_correction_generator.prompt"
        )
        self._pipeline.connect(
            "sql_correction_generator.replies", "post_processor.replies"
        )

        super().__init__(self._pipeline)

    @timer
    def run(
        self,
        contexts: List[Document],
        invalid_generation_results: List[Dict[str, str]],
    ):
        logger.info("Ask SQLCorrection pipeline is running...")
        return self._pipeline.run(
            {
                "sql_correction_prompt_builder": {
                    "invalid_generation_results": invalid_generation_results,
                    "documents": contexts,
                    "alert": TEXT_TO_SQL_RULES,
                },
            }
        )


if __name__ == "__main__":
    llm_provider, _ = init_providers()
    sql_correction_pipeline = SQLCorrection(
        llm_provider=llm_provider,
    )

    print("generating sql_correction_pipeline.jpg to outputs/pipelines/ask...")
    sql_correction_pipeline.draw("./outputs/pipelines/ask/sql_correction_pipeline.jpg")
