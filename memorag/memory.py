from typing import Dict, List, Tuple
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
from functools import lru_cache
import numpy as np
from typing import Set
import logging
from time import time
from contextlib import contextmanager
import requests
from cdlib import algorithms  
import networkx.algorithms.community as nx_comm

logger = logging.getLogger(__name__)

@contextmanager
def timer(name: str):
    start = time()
    yield
    logger.info(f"{name} took {time() - start:.2f} seconds")

class Memory:
    def __init__(self, retriever: Retriever, config: Dict, graph_file: str = "graph.graphml"):
        """
        基础内存类，用于存储、加载和调用检索器与知识图谱等信息。
        """
        self.retriever = retriever
        self.graph = nx.Graph()
        self.graph_file = graph_file
        self.config = config
        self.nlp = spacy.load("zh_core_web_sm")

    def __call__(self, context: str) -> str:
        """
        占位方法，子类实现具体逻辑。
        """
        pass

class GraphMemory(Memory):
    def __init__(self, retriever: Retriever, config: Dict):
        """初始化 LazyGraphRAG 内存系统"""
        default_config = {
            'chunk_size': 500,
            'chunk_overlap': 50,
            'concept_min_len': 2,
            'concept_max_len': 10,
            'weight_threshold': 2,
            'similarity_threshold': 0.6,
            'max_neighbors': 3,
            'max_communities': 3,
            'graph_file': "graph.graphml",
            'use_api': True,
            'api_config': {
                "model": "deepseek-chat",
                "base_url": "https://api.deepseek.com",
                "api_key": None
            },
            'nlp_model': "zh_core_web_sm"
        }
        
        self.config = {**default_config, **config}
        super().__init__(retriever, self.config, graph_file=self.config['graph_file'])
        
        self.retriever = retriever
        self.retrieval_corpus = []
        
       
        if 'corpus_file' in config and os.path.exists(config['corpus_file']):
            try:
                with open(config['corpus_file'], 'r', encoding='utf-8') as f:
                    self.retrieval_corpus = json.load(f)
                    if self.retrieval_corpus:
                        self.retriever.add_documents(self.retrieval_corpus)
            except Exception as e:
                logger.error(f"Failed to load corpus: {str(e)}")
                self.retrieval_corpus = []
        
        
        if self.config['use_api']:
            self.api_url = f"{self.config['api_config']['base_url']}/v1/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.config['api_config']['api_key']}",
                "Content-Type": "application/json"
            }
            self.model = self.config['api_config']['model']
        else:
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config['model_name'], 
                    trust_remote_code=True,
                    mirror=self.config['mirror']
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config['model_name'],
                    device_map="auto",
                    trust_remote_code=True,
                    mirror=self.config['mirror']
                ).eval()
            except Exception as e:
                logger.error(f"Failed to initialize language model: {str(e)}")
                raise
        
        # 初始化NLP工具和向量化模型
        try:
            self.nlp = spacy.load(self.config['nlp_model'])
            self.vectorizer = self.nlp
        except:
            self.nlp = spacy.load("zh_core_web_sm")
            self.vectorizer = self.nlp
            logger.warning("Failed to load advanced NLP model, fallback to basic model")
        
        # 缓存
        self._concept_cache = {}  # 文本到概念的映射
        self._vector_cache = {}   # 文本到向量的映射
        self._community_cache = None  # 社区检测结果缓存
        self._concept_to_community = {}  # 概念到社区的映射

    def reset(self):
        """重置内存状态"""
        try:
            self.retrieval_corpus = []  # 重置检索语料库
            self.retriever.remove_all()  # 清除检索器索引
            self.graph.clear()  # 清除图
            self._concept_cache.clear()  # 清除概念缓存
            self._vector_cache.clear()  # 清除向量缓存
            self._community_cache = None  # 清除社区缓存
            self._concept_to_community.clear()  # 清除概念到社区的映射
        except Exception as e:
            logger.error(f"Error in reset: {str(e)}")

    def _call_api(self, prompt: str) -> str:
        """调用 Deepseek API"""
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 2048
                }
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return ""

    def memorize(self, context: str) -> Dict:
        """优化的内存构建"""
        try:
            if not context.strip():
                logger.warning("Empty context provided")
                return {"graph": {}, "hierarchical_communities": {}}
            
            if os.path.exists(self.graph_file):
                print(f"Loading existing graph from {self.graph_file}...")
                self.graph = nx.read_graphml(self.graph_file)
            else:
                print(f"No existing graph found at {self.graph_file}, constructing a new graph...")
                self.graph = nx.Graph()
            
            # 使用文本分割器处理文本
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
            )
            chunks = text_splitter.split_text(context)
            
            # 更新检索语料库
            new_docs = [chunk for chunk in chunks]
            self.retrieval_corpus.extend(new_docs)
            
            
            if new_docs:
                print(f"Adding {len(new_docs)} new documents to retriever...")
                self.retriever.add_documents(new_docs)
            
            # 使用NLP抽取概念（实体）
            concept_pairs = defaultdict(int)
            
            # 预处理：提取概念和构建图
            for chunk in chunks:
                # 获取概念并转换为列表以便索引
                concepts = list(self._extract_concepts(chunk))
                self._concept_cache[chunk] = set(concepts)
                self._vector_cache[chunk] = self._get_text_vector(chunk)
                
                # 构建概念共现关系
                for i in range(len(concepts)):
                    for j in range(i + 1, len(concepts)):
                        pair = tuple(sorted([concepts[i], concepts[j]]))
                        concept_pairs[pair] += 1
            
            # 构建和优化图
            for (c1, c2), weight in concept_pairs.items():
                if weight >= self.config['weight_threshold']:
                    self.graph.add_edge(c1, c2, weight=weight)
            
            # 一次性社区检测（leiden）
            self._community_cache = algorithms.leiden(self.graph)
            
            # 构建概念到社区的映射
            for comm_id, community in enumerate(self._community_cache.communities):
                for concept in community:
                    self._concept_to_community[concept] = comm_id
            
            # 移除孤立节点
            isolated_nodes = list(nx.isolates(self.graph))
            self.graph.remove_nodes_from(isolated_nodes)

            # 使用贪婪模块度算法进行社区检测
            communities = list(greedy_modularity_communities(self.graph))
            
            # 构建层次社区结构
            hierarchical_structure = {}
            for i, community in enumerate(communities):
                sub_graph = self.graph.subgraph(community)
                sub_communities = list(greedy_modularity_communities(sub_graph))
                hierarchical_structure[i] = [list(sc) for sc in sub_communities]

            # 保存图结构
            print(f"Saving optimized graph to {self.graph_file}...")
            nx.write_graphml(self.graph, self.graph_file)

            return {
                "graph": {node: list(self.graph.neighbors(node)) for node in self.graph.nodes()},
                "hierarchical_communities": hierarchical_structure
            }
            
        except Exception as e:
            logger.error(f"Error in memorize: {str(e)}")
            raise

    def refine_query(self, query: str, context: str) -> tuple[List[str], str]:
        """
        LazyGraphRAG风格的查询优化：
        1. 使用NLP抽取查询中的关键概念
        2. 利用图结构找到相关概念组
        3. 使用LLM生成自然的子查询
        """
        # 提取查询概念和扩展概念
        query_concepts = self._extract_concepts(query)
        expanded_concepts = set()
        for concept in query_concepts:
            if concept in self.graph:
                neighbors = set(self.graph.neighbors(concept))
                for neighbor in list(neighbors):
                    if self.graph[concept][neighbor]['weight'] >= self.config['weight_threshold']:
                        expanded_concepts.add(neighbor)
        
        # 按重要性排序概念
        concept_scores = []
        for concept in (query_concepts | expanded_concepts):
            if concept in self.graph:
                score = (
                    self.graph.degree(concept) * 2 if concept in query_concepts else self.graph.degree(concept),
                    sum(self.graph[concept][n]['weight'] for n in self.graph[concept])
                )
                concept_scores.append((concept, score))
        
        # 选择最重要的概念对
        concept_scores.sort(key=lambda x: x[1], reverse=True)
        top_concepts = [c for c, _ in concept_scores[:6]]  # 取前6个概念
        
        # 使用LLM生成子查询
        concept_pairs = []
        for i in range(0, len(top_concepts), 2):
            if i + 1 < len(top_concepts):
                concept_pairs.append((top_concepts[i], top_concepts[i+1]))
        
        prompt = f"""基于原始问题和给定的概念对生成3-4个自然的子问题。

原始问题：{query}

概念对：
{' '.join([f'({c1}, {c2})' for c1, c2 in concept_pairs])}

要求：
子问题要自然流畅
保持与原问题的关联
每个子问题关注不同方面
"""

        try:
            # 使用LLM生成子查询
            if self.config['use_api']:
                sub_queries_text = self._call_api(prompt)
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(**inputs, max_length=256, temperature=0.7)
                sub_queries_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 解析子查询，去掉数字前缀
            sub_queries = []
            for line in sub_queries_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # 去掉数字前缀和点
                line = line.lstrip('0123456789. ')
                if line and not line.startswith('示例'):
                    sub_queries.append(line)
            
            # 添加原始查询
            sub_queries.append(query)
            
            # 合并查询
            merged_query = " ".join(sub_queries)
            return sub_queries, merged_query
            
        except Exception as e:
            logger.error(f"Failed to generate sub-queries: {str(e)}")
            
            sub_queries = [query]  # 至少保留原始查询
            for c1, c2 in concept_pairs[:2]:
                sub_queries.append(f"{query}中{c1}与{c2}的联系？")
            return sub_queries, " ".join(sub_queries)

    def match_query(self, query: str, sub_queries: List[str],
                    top_k: int = 5,
                    similarity_threshold: float = 0.4) -> List[str]:
        """优化的查询匹配"""
        with timer("match_query"):
            final_chunks = []
            seen_chunks = set()
            
            for sub_q in sub_queries:
                if not isinstance(sub_q, str):
                    continue
                
                # 1. 初始检索
                chunk_scores = self._initial_retrieval(sub_q, top_k * 3)
                logger.info(f"Initial retrieval found {len(chunk_scores)} chunks")
                
                # 2. 快速相似度评估
                candidates = self._fast_filter_candidates(sub_q, chunk_scores)
                logger.info(f"Fast filter kept {len(candidates)} candidates")
                
                # 3. 社区增强
                enhanced_candidates = self._enhance_with_communities(sub_q, candidates)
                logger.info(f"Community enhancement produced {len(enhanced_candidates)} candidates")
                
                # 4. LLM评估
                for chunk, score in enhanced_candidates[:top_k]:
                    if chunk not in seen_chunks and score >= similarity_threshold:
                        if self._llm_evaluate_relevance(sub_q, chunk):
                            final_chunks.append(chunk)
                            seen_chunks.add(chunk)
                            logger.info(f"Added chunk with score {score:.3f}")
            
            logger.info(f"Final chunks count: {len(final_chunks)}")
            return final_chunks[:top_k]

    def _fast_filter_candidates(self, query: str, chunk_scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """快速过滤候选文档"""
        # 获取查询概念（使用缓存）
        if query not in self._concept_cache:
            self._concept_cache[query] = self._extract_concepts(query)
        query_concepts = self._concept_cache[query]
        
        # 获取查询向量（使用缓存）
        if query not in self._vector_cache:
            self._vector_cache[query] = self._get_text_vector(query)
        query_vector = self._vector_cache[query]
        
        # 计算概念重叠度和向量相似度
        filtered_scores = []
        for chunk, initial_score in chunk_scores:
            # 使用缓存的概念
            chunk_concepts = self._concept_cache.get(chunk)
            if chunk_concepts is None:
                chunk_concepts = self._extract_concepts(chunk)
                self._concept_cache[chunk] = chunk_concepts
            
            # 计算概念重叠
            overlap = len(query_concepts & chunk_concepts)
            
            # 使用缓存的向量
            if chunk not in self._vector_cache:
                self._vector_cache[chunk] = self._get_text_vector(chunk)
            chunk_vector = self._vector_cache[chunk]
            
            # 计算向量相似度
            vector_sim = float(cosine_similarity(
                [query_vector],
                [chunk_vector]
            )[0][0])
            
            # 综合评分
            score = (initial_score * 0.4 + 
                    (overlap / max(len(query_concepts), 1)) * 0.3 + 
                    vector_sim * 0.3)
            
            filtered_scores.append((chunk, score))
        
        return sorted(filtered_scores, key=lambda x: x[1], reverse=True)

    def _enhance_with_communities(self, 
                            query: str, 
                            candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """使用社区信息增强候选文档"""
        query_concepts = self._concept_cache[query]
        query_communities = {
            self._concept_to_community[c]
            for c in query_concepts
            if c in self._concept_to_community
        }
        
        enhanced_scores = []
        for chunk, score in candidates:
            chunk_concepts = self._concept_cache[chunk]
            chunk_communities = {
                self._concept_to_community[c]
                for c in chunk_concepts
                if c in self._concept_to_community
            }
            
            # 计算社区重叠
            community_overlap = len(query_communities & chunk_communities)
            enhanced_score = score * (1 + 0.2 * community_overlap)
            enhanced_scores.append((chunk, enhanced_score))
        
        return sorted(enhanced_scores, key=lambda x: x[1], reverse=True)

    def _llm_evaluate_relevance(self, query: str, chunk: str) -> bool:
        """使用 LLM 评估文本块与查询的相关性"""
        try:
            # 构建评估提示词
            prompt = f"""判断以下文本块是否与问题相关。

问题：{query}
文本块：{chunk}

要求：
1. 如果文本块包含可能有助于回答问题的信息，返回"相关"
2. 如果文本块完全无关，返回"不相关"
3. 只返回"相关"或"不相关"，不要解释

回答："""

            # 调用 API 获取评估结果
            response = self._call_api(prompt)
            
            # 记录评估结果
            logger.info(f"LLM relevance evaluation: {response}")
            
            # 放宽判断标准
            return "相关" in response or "有关" in response or "是" in response
            
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {str(e)}")
            # 如果评估失败，默认保留该文本块
            return True

    def _initial_retrieval(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """初始检索和评分"""
        scores, indices = self.retriever.search(query, top_k=top_k)
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        
        chunk_scores = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.retrieval_corpus):
                chunk = self.retrieval_corpus[idx]
                chunk_scores.append((chunk, float(score)))
        
        return chunk_scores

    def _build_doc_community_graph(self, chunk_scores: List[Tuple[str, float]]) -> nx.Graph:
        """构建文档-社区二分图"""
        graph = nx.Graph()
        
        # 使用整数作为节点ID
        doc_nodes = {}  # 文档节点映射
        comm_nodes = {}  # 社区节点映射
        next_id = 0
        
        # 添加文档节点
        for chunk, score in chunk_scores:
            doc_id = next_id
            next_id += 1
            doc_nodes[doc_id] = chunk
            graph.add_node(doc_id, type='document', score=score)
            
            # 提取文档中的概念
            concepts = self._extract_concepts(chunk)
            
            # 找到概念所属的社区
            for concept in concepts:
                if concept in self.graph:
                    communities = nx_comm.label_propagation_communities(self.graph)
                    for comm_id, community in enumerate(communities):
                        if concept in community:
                            if comm_id not in comm_nodes:
                                comm_nodes[comm_id] = f"community_{comm_id}"
                                graph.add_node(next_id + comm_id, type='community')
                            graph.add_edge(doc_id, next_id + comm_id, weight=score)
        
        # 添加节点属性
        nx.set_node_attributes(graph, {k: {'name': v} for k, v in {**doc_nodes, **comm_nodes}.items()})
        
        return graph

    def _rank_communities(self, communities, chunk_scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """对社区进行排序"""
        community_scores = defaultdict(float)
        
        # 使用 communities.communities 获取社区列表
        for chunk, score in chunk_scores:
            for comm in communities.communities:
                # 获取社区中的实际文档内容
                try:
                    
                    community_docs = [
                        communities.graph.nodes[n].get('name', str(n)) 
                        for n in comm
                    ]
                    if chunk in community_docs:
                        community_scores[tuple(comm)] += score
                except KeyError:
                    continue
        
        return sorted(community_scores.items(), key=lambda x: x[1], reverse=True)

    def _iterative_deepening_search(self, query: str, ranked_communities: List[Tuple[str, float]], 
                                  community_budget: int, relevance_budget: int) -> List[str]:
        """迭代深入搜索相关文档"""
        relevant_chunks = []
        tested_chunks = set()
        zero_relevance_count = 0
        
        for comm_nodes, _ in ranked_communities:
            if zero_relevance_count >= community_budget:
                break
            
            # 获取社区中的实际文档内容
            try:
                community_docs = [
                    self.graph.nodes[n].get('name', str(n))
                    for n in comm_nodes
                    if self.graph.nodes[n].get('type') == 'document'
                ]
            except KeyError:
                continue
            
            # 获取未测试的文档
            untested_chunks = [
                doc for doc in community_docs 
                if doc not in tested_chunks
            ][:relevance_budget]
            
            # 评估相关性
            found_relevant = False
            for chunk in untested_chunks:
                if len(relevant_chunks) >= relevance_budget:
                    break
                    
                relevance = self._assess_chunk_relevance(query, chunk)
                tested_chunks.add(chunk)
                
                if relevance >= 0.6:  # 相关性阈值
                    relevant_chunks.append(chunk)
                    found_relevant = True
            
            if not found_relevant:
                zero_relevance_count += 1
            else:
                zero_relevance_count = 0
        
        return relevant_chunks

    def _assess_chunk_relevance(self, query: str, chunk: str) -> float:
        """使用LLM评估文档相关性"""
        prompt = f"""评估以下文本与问题的相关性（返回0-1分）：
问题：{query}
文本：{chunk}
仅返回分数，无需解释。"""

        try:
            if self.config['use_api']:
                score_text = self._call_api(prompt)
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(**inputs, max_length=128, temperature=0.0)
                score_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return float(score_text.strip())
        except:
            return 0.0

    def _compute_overlap(self, text1: str, text2: str) -> float:
        """计算两段文本的内容重叠度"""
        concepts1 = self._extract_concepts(text1)
        concepts2 = self._extract_concepts(text2)
        
        if not concepts1 or not concepts2:
            return 0.0
        
        overlap = len(concepts1 & concepts2)
        return overlap / min(len(concepts1), len(concepts2))

    def _evaluate_chunks_with_llm(self, query: str, chunks: List[str], budget: int) -> List[str]:
        """使用LLM评估文本块相关性的辅助方法"""
        relevant_chunks = []
        
        for chunk in chunks[:budget]:
            prompt = f"评估文本与问题的相关性（返回0-1分）：\n问题：{query}\n文本：{chunk}"
            
            if self.config['use_api']:
                score_text = self._call_api(prompt)
            else:
                
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    temperature=0.0
                )
                score_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            try:
                score = float(score_text.strip())
                if score >= 0.6:
                    relevant_chunks.append(chunk)
            except ValueError:
                continue
            
        return relevant_chunks

    def map_answer(self, query: str, matched_chunks: List[str]) -> str:
        """从匹配的文本块中提取关键声明"""
        try:
            
            prompt = f"""仔细阅读以下文本，并提取所有与问题相关的关键信息和声明：

文本：
{' '.join(matched_chunks)}

问题：{query}

要求：
1. 提取所有可能与问题相关的信息，包括直接和间接相关的内容
2. 每条声明单独一行，以数字编号
3. 声明要尽可能详细和具体
4. 包含相关的背景信息、时间、地点、人物、数字等具体细节
5. 如果有多个相似的信息，都要列出来
6. 保持声明的原始表述，不要过度概括或简化
7. 不要添加解释或评论
8. 确保每条声明都完整且独立

回答："""

            # 调用 API 生成声明
            claims = self._call_api(prompt)
            
            # 记录生成的声明
            logger.info(f"Generated claims: {claims}")
            
            return claims.strip()
            
        except Exception as e:
            logger.error(f"Error in map_answer: {str(e)}")
            return ""

    def reduce_answer(self, query: str, claims: str) -> str:
        """生成最终答案"""
        if not claims:
            logger.warning("No claims provided for reducing answer")
            return ""
        
        try:
            
            prompt = f"""基于以下声明回答问题：

声明：
{claims}

问题：{query}

要求：
1. 基于声明直接回答问题
2. 简洁明了，不要解释
3. 如果声明不足以回答问题，返回空字符串

回答："""

            # 调用 API 生成答案
            answer = self._call_api(prompt)
            
            # 记录生成的答案
            logger.info(f"Generated answer: {answer}")
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error in reduce_answer: {str(e)}")
            return ""

    def __call__(self, query: str, context: str) -> str:
        
        answer = ""
        return answer

    @lru_cache(maxsize=1000)
    def _get_concept_vector(self, text: str) -> np.ndarray:
        """缓存概念的向量表示"""
        return self.nlp(text).vector

    @lru_cache(maxsize=1000)
    def _extract_concepts(self, text: str) -> Set[str]:
        """通用的概念提取"""
        concepts = set()
        doc = self.nlp(text)
        
        # 提取命名实体
        for ent in doc.ents:
            if 2 <= len(ent.text) <= 10:
                concepts.add(ent.text)
        
        # 提取核心名词短语
        for token in doc:
            # 提取名词
            if token.pos_ in ['NOUN', 'PROPN'] and 2 <= len(token.text) <= 10:
                concepts.add(token.text)
            # 提取重要的依存关系
            if token.dep_ in ['nsubj', 'dobj', 'pobj'] and 2 <= len(token.text) <= 10:
                concepts.add(token.text)
        
        return concepts

    def _save_graph(self):
        """优化的图存储"""
        try:
            # 保存为二进制格式以提高效率
            nx.write_gpickle(self.graph, self.graph_file + '.pickle')
            # 同时保存可读格式用于测试
            nx.write_graphml(self.graph, self.graph_file)
        except Exception as e:
            logger.error(f"Failed to save graph: {str(e)}")

    def _load_graph(self):
        """优化的图加载"""
        try:
            # 优先尝试加载二进制格式
            if os.path.exists(self.graph_file + '.pickle'):
                return nx.read_gpickle(self.graph_file + '.pickle')
            elif os.path.exists(self.graph_file):
                return nx.read_graphml(self.graph_file)
            return nx.Graph()
        except Exception as e:
            logger.error(f"Failed to load graph: {str(e)}")
            return nx.Graph()

    @lru_cache(maxsize=1000)
    def _get_word_vector(self, word: str) -> np.ndarray:
        """获取词向量"""
        doc = self.nlp(word)
        return doc.vector

    @lru_cache(maxsize=1000)
    def _get_text_vector(self, text: str) -> np.ndarray:
        """统一的文本向量化接口"""
        try:
            doc = self.vectorizer(text)
            return doc.vector
        except Exception as e:
            logger.error(f"Failed to get vector for text: {str(e)}")
            return np.zeros(768)  # 返回零向量作为后备

    def _get_batch_vectors(self, texts: List[str]) -> np.ndarray:
        """批量获取文本向量"""
        try:
            docs = list(self.vectorizer.pipe(texts))
            return np.array([doc.vector for doc in docs])
        except Exception as e:
            logger.error(f"Failed to get batch vectors: {str(e)}")
            return np.zeros((len(texts), 768))

class AgentMemory(Memory):
    def __init__(self, config: Dict):
        super().__init__(None, config)


class KVMemory(Memory):
    def __init__(self, config: Dict):
        super().__init__(None, config)

    def memorize(self, context: str) -> None:
        
        pass


if __name__ == "__main__":
    query = "什么是MemoRAG？"
    context = "MemoRAG是一种基于记忆的检索增强生成模型，它通过记忆来增强生成模型的生成能力。"
    memory = KVMemory({})
    memory.memorize(context)
