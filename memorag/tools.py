from typing import Dict, List

class EntityExtractor:
    def __init__(self, config: Dict):
        self.config = config

    def __call__(self, context: str) -> List[str]:
        pass

class RelationExtractor:
    def __init__(self, config: Dict):
        self.config = config

    def __call__(self, context: str) -> List[str]:
        pass


