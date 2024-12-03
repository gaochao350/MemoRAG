from typing import Dict, List
from .memorag import Model
from .retrieval import Retriever
import os
import networkx as nx
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
import spacy
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import AgglomerativeClustering
from langchain_core.documents import Document
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import json

class Memory:
    def __init__(self, retriever, config: Dict, graph_file: str = "graph.graphml"):
        self.retriever = retriever
        self.graph = nx.Graph()
        self.graph_file = graph_file
        self.config = config
        self.nlp = spacy.load("zh_core_web_sm")

    def __call__(self, context: str) -> str:
        pass

class GraphMemory(Memory):
    def __init__(self, retriever, config: Dict):
        # 需要传入 graph_file 参数
        super().__init__(retriever, config, graph_file="graph.graphml")
        self.retriever = retriever
        self.retrieval_corpus = []
        
        # 从配置的存储位置加载已有语料库
        corpus_file = config.get('corpus_file')
        if corpus_file and os.path.exists(corpus_file):
            with open(corpus_file, 'r', encoding='utf-8') as f:
                self.retrieval_corpus = json.load(f)
        
        # 添加镜像站点配置
        mirror = "https://mirrors.aliyun.com/hugging-face"  
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-3B", 
            trust_remote_code=True,
            mirror=mirror
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B",
            device_map="auto",
            trust_remote_code=True,
            mirror=mirror
        ).eval()

    def memorize(self, context: str) -> None:
        # tools and deployed model: Spacy -> 
        # API: aliyun API stc. deepseek API  -> ner / deepseek

        # step 1: chunk the context into multiple chunks: semantic-splitting
        # step 2: extract concepts and their co-occurrences, normalize the concepts
        # step 3: uses graph statistics to optimize the concept graph and extract hierarchical community structure
        # 北京市政府工作报告 -> 经济，民生，科技，教育，医疗，环保，文化，体育，社会治理，国际合作
        # 刑法 -> 干啥会被判死刑？ 判了死刑还有救吗？
        # how to construct and store the graph? -> ask chatgpt
        """
        Constructs, updates, saves, or loads a graph based on the input context.
        """
        # Load the graph if it exists
        if os.path.exists(self.graph_file):
            print(f"Loading existing graph from {self.graph_file}...")
            self.graph = nx.read_graphml(self.graph_file)
        else:
            print(f"No existing graph found at {self.graph_file}, constructing a new graph...")
            self.graph = nx.Graph()
        
        # Step 2: 使用SpaCy进行语义分块
        doc = self.nlp(context)
        sentences = [sent.text for sent in doc.sents]
        sentence_vectors = [sent.vector for sent in doc.sents]
        
        # 使用层次聚类将相似句子组合在一起
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.5,  # 可调整的相似度阈值
            linkage='ward'
        ).fit(sentence_vectors)
        
        # 控制块大小并组合文本
        max_chunk_size = 1000  # 最大块字符数
        chunks = []
        current_chunk = []
        current_size = 0
        
        for idx, label in enumerate(clustering.labels_):
            sent_text = sentences[idx]
            # 如果当前块太大，开始新块
            if current_size + len(sent_text) > max_chunk_size:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                current_chunk = [sent_text]
                current_size = len(sent_text)
            else:
                current_chunk.append(sent_text)
                current_size += len(sent_text)
                
            # 如果是最后一个句子或下一个句子属于不同聚类
            if (idx == len(sentences)-1 or 
                clustering.labels_[idx] != clustering.labels_[idx+1]):
                if current_chunk:
                    chunks.append("".join(current_chunk))
                current_chunk = []
                current_size = 0
        
        # 为每个文本块创建Document对象
        texts = [Document(page_content=chunk) for chunk in chunks]

        # 将新的文本块添加到检索语料库
        self.retrieval_corpus.extend([chunk.page_content for chunk in texts])
        
        # 更新检索器的索引 - 使用 add_documents 而不是 update_index
        self.retriever.add_documents(self.retrieval_corpus)

        # Step 3: 为每个块提取实体并建立关系
        for text in texts:
            doc = self.nlp(text.page_content)
            entities = [ent.text for ent in doc.ents]
            
            # 构建实体关系
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    if not self.graph.has_edge(entities[i], entities[j]):
                        self.graph.add_edge(entities[i], entities[j], weight=1)
                    else:
                        self.graph[entities[i]][entities[j]]['weight'] += 1
        
        # Step 4: Optimize the graph using graph statistics
        weight_threshold = 2
        edges_to_remove = [(u, v) for u, v, w in self.graph.edges(data="weight") if w < weight_threshold]
        self.graph.remove_edges_from(edges_to_remove)

        # Remove isolated nodes
        isolated_nodes = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolated_nodes)
        # Step 5: Extract hierarchical community structure  
        #top-k communities
        top_level_communities = list(greedy_modularity_communities(self.graph))
        community_dict = {i: list(community) for i, community in enumerate(top_level_communities)}
        
        # Extract sub-communities within each top-level community
        hierarchical_structure = {}
        for community_id, nodes in community_dict.items():
            sub_graph = self.graph.subgraph(nodes)
            sub_communities = list(greedy_modularity_communities(sub_graph))
            hierarchical_structure[community_id] = [list(sc) for sc in sub_communities]

        # Save the optimized graph to file
        print(f"Saving optimized graph to {self.graph_file}...")
        nx.write_graphml(self.graph, self.graph_file)
        
        #Convert the graph to a dictionary format   
        graph_dict = {node: list(self.graph.neighbors(node)) for node in self.graph.nodes()}
        return {"graph": graph_dict, "hierarchical_communities": hierarchical_structure}

        '''
        graph = {"entity 1": ["entity 2", "entity 3"], "entity 2": ["entity 4", "entity 5"]}
        graph = []
        return graph
        '''
    
    def refine_query(self, query: str, context: str) -> List[str]:
        # Step 1: 生成子查询
        prompt = f"""请将以下问题分解成3到5个具体的子问题：
        原始问题：{query}
        
        要求：
        每个子问题必须是一个完整的问句
        每个子问题必须以"？"结尾
        
        请直接列出子问题："""

        max_attempts = 3  # 最多尝试3次
        for attempt in range(max_attempts):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 规范化子查询格式
            sub_queries = []
            for line in response.split('\n'):
                line = line.strip()
                if line and line[0].isdigit():
                    # 提取问分并确保以问号结尾
                    question = line.split('. ', 1)[1] if '. ' in line else line
                    question = question.strip('*[] \t\n')  # 移除特殊字符
                    if not question.endswith('？'):
                        question += '？'
                    sub_queries.append(question)
            
            # 只保留3-5个子查询
            if 3 <= len(sub_queries) <= 5:
                break
            elif len(sub_queries) > 5:
                sub_queries = sub_queries[:5]
                break
        
        # 如果多次尝试后仍未得到足够的子查询，补充通用子查询
        if len(sub_queries) < 3:
            default_queries = [
                f"{query}的具体实施情况如何？",
                f"{query}取得了哪些主要成效？",
                f"{query}面临哪些挑战和机遇？"
            ]
            sub_queries.extend(default_queries[:3 - len(sub_queries)])
        
        # Step 2: 利用知识图谱扩展子查询
        optimized_queries = []
        for sub_q in sub_queries:  
            doc = self.nlp(sub_q)
            entities = [ent.text for ent in doc.ents]
            
            related_entities = set()
            for entity in entities:
                if entity in self.graph:
                    neighbors = list(self.graph.neighbors(entity))
                    sorted_neighbors = sorted(
                        neighbors,
                        key=lambda x: self.graph[entity][x].get('weight', 0),
                        reverse=True
                    )[:3]
                    related_entities.update(sorted_neighbors)
            
            if related_entities:
                related_str = '、'.join(related_entities)
                optimized_query = f"{sub_q}（相关领域：{related_str}）"
            else:
                optimized_query = sub_q
                
            optimized_queries.append(optimized_query)  

        # 直接将子查询拼接作为合并查询，使用空格分隔
        merged_query = " ".join(optimized_queries)
        
        return optimized_queries, merged_query

    def match_query(self, query: str, sub_queries: List[str], 
                    top_k: int = 5,           # 每个社区选择的top-k文本块
                    zero_threshold: int = 3,   # z值：连续零相关社区的阈值
                    relevance_budget: int = 30 # 总相关性测试预算
                    ) -> List[str]:
        """
        为每个子查询检索和排序相关文本块。
        
        Args:
            query: 主查询
            sub_queries: 子查询列表 [3-5个]
            top_k: 每个社区选择的top-k文本块数
            zero_threshold: 触发子社区递归的连续零相关社区数
            relevance_budget: 总的相关性测试预算
        """
        # 计算每个子查询的预算
        budget_per_query = relevance_budget // len(sub_queries)
        final_chunks = []
        
        for sub_q in sub_queries:
            # 确保 sub_q 是字符串
            if isinstance(sub_q, list):
                sub_q = sub_q[0] if sub_q else ""
            
            # 1. 检索初始文本块
            scores, indices = self.retriever.search(sub_q)
            chunks = [self.retrieval_corpus[idx] for idx in indices[0]]
            chunk_vectors = [self.nlp(chunk).vector for chunk in chunks]
            
            # 2. 构建初始社区
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.5,
                linkage='ward'
            ).fit(chunk_vectors)
            
            # 3. 构建社区和文本块的映射关系
            communities = defaultdict(list)
            chunk_to_community = {}
            for idx, label in enumerate(clustering.labels_):
                communities[label].append(chunks[idx])
                chunk_to_community[chunks[idx]] = label
                
            # 4. 计算文本块与查询的相似度
            chunk_scores = []
            query_vec = self.nlp(sub_q).vector
            for chunk in chunks:
                chunk_vec = self.nlp(chunk).vector
                similarity = cosine_similarity([query_vec], [chunk_vec])[0][0]
                chunk_scores.append((chunk, similarity))
            
            # 5. 对文本块进行排序
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 6. 计算社区得分（基于top-k文本块）
            community_scores = defaultdict(float)
            for chunk, score in chunk_scores[:top_k]:
                comm_id = chunk_to_community[chunk]
                community_scores[comm_id] += score
                
            # 7. 对社区进行排序
            ranked_communities = sorted(
                communities.items(),
                key=lambda x: community_scores[x[0]],
                reverse=True
            )
            
            # 8. 迭代处理社区
            tested_chunks = set()
            relevant_chunks = []
            zero_count = 0
            current_budget = budget_per_query
            
            def process_community(comm_chunks, depth=0):
                nonlocal zero_count, current_budget
                
                if current_budget <= 0:
                    return []
                    
                # 对社区内文本块进行相关性评估
                relevant = []
                for chunk in comm_chunks:
                    if chunk in tested_chunks:
                        continue
                        
                    if current_budget <= 0:
                        break
                        
                    # 使用LLM评估相关性
                    prompt = f"""请评估以下文本与问题的相关性，返回0-1之间的分数。
                    问题：{sub_q}
                    文本：{chunk}
                    只返回分数，不要其他内容。"""
                    
                    response = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                    outputs = self.model.generate(
                        **response,
                        max_length=128,
                        temperature=0.0
                    )
                    # 解析生成的文本，提取分数
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    try:
                        relevance = float(generated_text.strip().split()[-1])  # 提取最后一个数字作为分数
                    except ValueError:
                        print(f"无法解析生成的文本为浮点数: {generated_text}")
                        continue
                    
                    tested_chunks.add(chunk)
                    current_budget -= 1
                    
                    if relevance >= 0.6:  # 相关性阈值
                        relevant.append(chunk)
                        zero_count = 0
                    else:
                        zero_count += 1
                        
                return relevant
                
            # 9. 处理主社区
            for comm_id, comm_chunks in ranked_communities:
                # 处理当前社区
                new_relevant = process_community(comm_chunks)
                relevant_chunks.extend(new_relevant)
                
                # 如果连续z个社区没有相关内容，进入子社区
                if zero_count >= zero_threshold:
                    # 构建子社区
                    sub_vectors = [self.nlp(chunk).vector for chunk in comm_chunks]
                    if len(sub_vectors) > 1:  # 确保有足够的样本进行聚类
                        sub_clustering = AgglomerativeClustering(
                            n_clusters=min(len(comm_chunks) // 2, 3),
                            linkage='ward'
                        ).fit(sub_vectors)
                        
                        # 处理子社区
                        sub_communities = defaultdict(list)
                        for idx, label in enumerate(sub_clustering.labels_):
                            sub_communities[label].append(comm_chunks[idx])
                        
                        for sub_comm_chunks in sub_communities.values():
                            new_relevant = process_community(sub_comm_chunks, depth=1)
                            relevant_chunks.extend(new_relevant)
                            
                            if current_budget <= 0:
                                break
                
                    zero_count = 0  # 重置计数器
                    
                if current_budget <= 0:
                    break
                    
            final_chunks.extend(relevant_chunks)
        
        return final_chunks

    def map_answer(self, query: str, answer: str, max_length: int=1024) -> str:
        """
        从相关文本块中提取并组织与查询相关的声明。
        
        Args:
            query: 主查询
            answer: 包含相关文本块的列表
            max_length: 最大上下文窗口大小
        
        Returns:
            str: 组织好的声明列表
        """
        # 将文本块列表转换为单独的文本块
        chunks = answer.split('\n') if isinstance(answer, str) else answer
        
        # 为每个文本块提取概念并构建子图
        concept_graph = nx.Graph()
        chunk_concepts = {}  # 存储每个文本块的概念
        
        for chunk in chunks:
            doc = self.nlp(chunk)
            # 提取实体作为概念
            concepts = [ent.text for ent in doc.ents]
            chunk_concepts[chunk] = concepts
            
            # 在概念之间建立边
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    if not concept_graph.has_edge(concepts[i], concepts[j]):
                        concept_graph.add_edge(concepts[i], concepts[j], weight=1)
                    else:
                        concept_graph[concepts[i]][concepts[j]]['weight'] += 1
        
        # 使用社区检测对概念进行分组
        communities = list(greedy_modularity_communities(concept_graph))
        
        # 基于概念社区对文本块进行分组
        chunk_groups = defaultdict(list)
        for chunk, concepts in chunk_concepts.items():
            # 找出该文本块概念所属的主要社区
            community_counts = defaultdict(int)
            for concept in concepts:
                for i, community in enumerate(communities):
                    if concept in community:
                        community_counts[i] += 1
            
            # 将文本块分配给包含其最多概念的社区
            if community_counts:
                main_community = max(community_counts.items(), key=lambda x: x[1])[0]
                chunk_groups[main_community].append(chunk)
        
        # 使用LLM从每个组中提取相关声明
        all_claims = []
        for group in chunk_groups.values():
            # 将组内文本块合并
            group_text = "\n".join(group)
            
            # 构建提取声明的提示
            prompt = f"""从以下文本中提取与问题相关的关键声明：
            问题：{query}
            文本：{group_text}
            
            请提取3-5个关键声明，每个声明应该：
            直接回答问题的某个方面
            具体且有信息量
            避免重复信息
            
            仅返回声明列表，每行一个声明。"""
            
            # 使用LLM生成声明
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            claims = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            all_claims.extend([c.strip() for c in claims.split('\n') if c.strip()])
        
        # 对声明进行排序和筛选
        def score_claim(claim):
            # 计算声明的相关性分数
            prompt = f"""评估以下声明与问题的相关性（0-1分）：
            问题：{query}
            声明：{claim}
            仅返回分数。"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                temperature=0.0
            )
            score_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            try:
                return float(score_text.strip())
            except:
                return 0.0
        
        # 对声明进行评分和排序
        scored_claims = [(claim, score_claim(claim)) for claim in all_claims]
        scored_claims.sort(key=lambda x: x[1], reverse=True)
        
        # 选择最相关的声明，确保总长度不超过max_length
        selected_claims = []
        current_length = 0
        
        for claim, _ in scored_claims:
            if current_length + len(claim) + 1 <= max_length:  # +1 为换行符
                selected_claims.append(claim)
                current_length += len(claim) + 1
            else:
                break
        
        # 返回筛选后的声明列表
        return "\n".join(selected_claims)

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
    
    