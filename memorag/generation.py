from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Union
import os
import json 
from .agent import Agent
from .prompt import zh_prompts

class Generator:
    def __init__(self, generator: Union[Agent, AutoModelForCausalLM], config: Dict):
        self.generator = generator
        self.config = config

    def __call__(self, prompt: str, generation_config: Dict) -> str:
        pass

    def generate(self, prompt: str, generation_config: Dict) -> str:
        pass

    def batch_generate(self, prompts: List[str], generation_config: Dict) -> List[str]:
        pass

if __name__ == "__main__":
    query = "什么是MemoRAG？"
    context = "MemoRAG是一种基于记忆的检索增强生成模型，它通过记忆来增强生成模型的生成能力。"
    prompt = zh_prompts["qa_gen"].format(context=context, input=query)
    model = AutoModelForCausalLM.from_pretrained("THUDM/chatglm-6b")
    config = {}
    generation_config = {}
    generator = Generator(model, config)
    response = generator.generate(prompt, generation_config)
    print(response)
