from typing import Dict, List
from .memorag import Model
from .retrieval import Retriever
from semantic_text_splitter import TextSplitter

class Memory:
    def __init__(self, config: Dict):
        self.config = config

    def __call__(self, context: str) -> str:
        pass

class GraphMemory(Memory):
    def __init__(self, retriever, config: Dict):
        super().__init__(retriever, config)
        self.retriever = retriever


    def memorize(self, context: str) -> None:
        # tools and deployed model: Spacy -> 
        # API: aliyun API stc. deepseek API  -> ner / deepseek

        # step 1: chunk the context into multiple chunks: semantic-splitting
        # step 2: extract concepts and their co-occurrences, normalize the concepts
        # step 3: uses graph statistics to optimize the concept graph and extract hierarchical community structure
        # 北京市政府工作报告 -> 经济，民生，科技，教育，医疗，环保，文化，体育，社会治理，国际合作
        # 刑法 -> 干啥会被判死刑？ 判了死刑还有救吗？
        # how to construct and store the graph? -> ask chatgpt
        graph = {"entity 1": ["entity 2", "entity 3"], "entity 2": ["entity 4", "entity 5"]}
        graph = []
        return graph
    
    def refine_query(self, query: str) -> str:
        # open-sourced: qwen 2.5 3B, llama3.2 3B 
        # API: gpt3.5 API stc. deepseek API 
        # prompt the model to refine the query into sub-questions
        # query：北京去年总体表现怎么样？
        # output: 北京去年经济怎么样？ -> top-10 chunks 5 out 10 chunks are to the query, 
        #        北京去年民生怎么样？
        #        北京去年科技怎么样？
        #        海淀区去年怎么样？

        output = ["sub-query 1", "sub-query 2", "sub-query 3"]
        return output

    def match_query(self, query: str, sub_queries: List[str]) -> List[str]:
        # for each sub-query, retrieve the most relevant chunks from the long document
        # build community structure for the chunks
        # rank the chunk communities by their relevance to the input query or the combined query 
           # query + all chunks -> long sequence -> main query -> similarity
           # query + each chunks -> long sequence -> main query -> similarity
        # for each chunk community, select relevant chunks by measuring the semantic similarity between the chunks and the query
            #threshold = 0.6 reranker, matcher
        output = [[]]
        return output
    
    def map_answer(self, query: str, answer: str, max_length: int=1024) -> str:
        # open-sourced: qwen 2.5 3B, llama3.2 3B 
        # API: gpt3.5 API stc. deepseek API 

        # recursively generate useful claims from each chunk community
        # untill fit the predefined length
        # claim: 北京去年经济稳定增长，增长率高达8%
        # claim: 北京去年民生持续改善，居民收入稳步增长
        pass

    def reduce_answer(self, query: str, answer: str) -> str:
        # open-sourced: qwen 2.5 3B, llama3.2 3B 
        # API: gpt3.5 API stc. deepseek API 

        # produce the final answer from the useful claims
        pass

    def __call__(self, query: str, context: str) -> str:
        answer = ""
        return answer


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
    
    