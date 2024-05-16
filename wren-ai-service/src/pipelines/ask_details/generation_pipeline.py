import logging

from haystack import Pipeline

from src.core.llm_provider import LLMProvider
from src.core.pipeline import BasicPipeline
from src.pipelines.ask_details.components.post_processors import (
    init_generation_post_processor,
)
from src.pipelines.ask_details.components.prompts import (
    ask_details_system_prompt,
    init_ask_details_prompt_builder,
)
from src.utils import init_providers, load_env_vars

load_env_vars()
logger = logging.getLogger("wren-ai-service")


class Generation(BasicPipeline):
    def __init__(
        self,
        llm_provider: LLMProvider,
    ):
        self._pipeline = Pipeline()
        self._pipeline.add_component(
            "ask_details_prompt_builder",
            init_ask_details_prompt_builder(),
        )
        self._pipeline.add_component(
            "ask_details_generator",
            llm_provider.get_generator(system_prompt=ask_details_system_prompt),
        )
        self._pipeline.add_component("post_processor", init_generation_post_processor())

        self._pipeline.connect(
            "ask_details_prompt_builder.prompt", "ask_details_generator.prompt"
        )
        self._pipeline.connect(
            "ask_details_generator.replies", "post_processor.replies"
        )

        super().__init__(self._pipeline)

    def run(self, sql: str):
        logger.info("Ask Details Generation pipeline is running...")
        return self._pipeline.run(
            {
                "ask_details_prompt_builder": {
                    "sql": sql,
                },
            }
        )


if __name__ == "__main__":
    llm_provider, _ = init_providers()
    generation_pipeline = Generation(
        llm_provider=llm_provider,
    )

    print("generating generation_pipeline.jpg to outputs/pipelines/ask_details...")
    generation_pipeline.draw("./outputs/pipelines/ask_details/generation_pipeline.jpg")
