from typing import Dict

class Memory:
    def __init__(self, config: Dict):
        self.config = config

    def __call__(self, context: str) -> str:
        pass

class GraphMemory(Memory):
    def __init__(self, config: Dict):
        super().__init__(config)

    def memorize(self, context: str) -> None:
        pass

    def __call__(self, query: str) -> str:
        pass

    def build_index(self) -> None:
        pass

    def summarize_index(self) -> None:
        pass

    def refine_query(self, query: str) -> str:
        pass

    def match_query(self, query: str) -> str:
        pass

    def map_answer(self, query: str, answer: str) -> str:
        pass

    def reduce_answer(self, query: str, answer: str) -> str:
        pass

    def graph_query(self, query: str) -> str:
        pass

class AgentMemory(Memory):
    def __init__(self, config: Dict):
        super().__init__(config)


class KVMemory(Memory):
    def __init__(self, config: Dict):
        super().__init__(config)

    def memorize(self, context: str) -> None:
        pass


if __name__ == "__main__":
    query = "什么是MemoRAG？"
    context = "MemoRAG是一种基于记忆的检索增强生成模型，它通过记忆来增强生成模型的生成能力。"
    memory = KVMemory({})
    memory.memorize(context)
    
    