import asyncio
import random

from async_pipeline.stage import PipelineStage, pipeline_operation


class Loader(PipelineStage):
    def __init__(self, conf, *args, **kwargs) -> None:
        self._operation = conf["load"]
        super().__init__(*args, **kwargs)

    @pipeline_operation
    async def print(self, message):
        print(f"[FINAL OUT]: {message}")
        await asyncio.sleep(random.randint(1, 5))  # simulated IO delay
