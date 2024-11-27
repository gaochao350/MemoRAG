from .generation import Generator
from .retrieval import Retrieval
from typing import Dict

class Pipeline:
    def __init__(self, generator: Generator, retrieval: Retrieval, config: Dict):
        self.generator = generator
        self.retrieval = retrieval
        self.config = config

    def __call__(self, query: str, context: str) -> str:
        pass
