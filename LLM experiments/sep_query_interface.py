# Step 5: Query Interface

#!/usr/bin/env python3
"""
sep_query_interface.py

Advanced query interface for the SEP Knowledge Graph
Optimized for Linux machine with RTX 2060 SUPER, 30GB RAM, and Ollama
"""

import json
import logging
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path
import pickle

from py2neo import Graph, Node, Relationship
import torch
from transformers import AutoTokenizer, AutoModel
import ollama
from tqdm import tqdm
import pandas as pd
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import re
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Structure for query results"""
    nodes: List[Dict] = field(default_factory=list)
    relationships: List[Dict] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    visualization_data: Optional[Dict] = None
    
@dataclass
class PhilosophicalDebate:
    """Structure for debate analysis"""
    topic: str
    camps: Dict[str, List[str]]  # camp_name -> list of philosophers/articles
    key_arguments: Dict[str, List[str]]  # camp_name -> arguments
    historical_development: List[Dict]  # timeline of debate
    central_texts: List[str]
    relationships: List[Dict]  # inter-camp relationships

class SEPGraphQuery:
    def __init__(self, 
                 neo4j_url: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password",
                 use_gpu: bool = True,
                 ollama_model: str = "nous-hermes2:10.7b",
                 cache_embeddings: bool = True):
        """
        Initialize the query interface
        
        Args:
            neo4j_url: Neo4j database URL
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            use_gpu: Whether to use GPU for embeddings
            ollama_model: Ollama model to use for query understanding
            cache_embeddings: Whether to cache embeddings to disk
        """
        # Connect to Neo4j or load NetworkX fallback
        self.use_neo4j = self._connect_to_graph(neo4j_url, neo4j_user, neo4j_password)
        
        # Set up embedding model for semantic search
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Load sentence transformer
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(self.device)
        self.model.eval()
        
        # Ollama for query understanding
        self.ollama_model = ollama_model
        self._check_ollama()
        
        # Caching
        self.cache_embeddings = cache_embeddings
        self.embedding_cache_file = Path("embeddings_cache.pkl")
        self.embedding_cache = self._load_embedding_cache()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=12)
        
        # Precompute article embeddings if needed
        self._initialize_article_embeddings()
    
    def _connect_to_graph(self, neo4j_url: str, user: str, password: str) -> bool:
        """Connect to Neo4j or load NetworkX graph"""
        try:
            self.graph = Graph(neo4j_url, auth=(user, password))
            # Test connection
            self.graph.run("MATCH (n) RETURN n LIMIT 1")
            logger.info("Connected to Neo4j successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j: {e}")
            logger.info("Looking for NetworkX fallback...")
            
            # Try to load NetworkX graph from pickle
            graph_files = list(Path(".").glob("sep_graph_backup_*.pkl"))
            if graph_files:
                latest_graph = max(graph_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"Loading NetworkX graph from {latest_graph}")
                
                with open(latest_graph, 'rb') as f:
                    backup_data = pickle.load(f)
                    self.nx_graph = backup_data.get('networkx_graph')
                    self.articles_data = backup_data.get('articles', {})
                
                if self.nx_graph:
                    logger.info(f"Loaded NetworkX graph with {self.nx_graph.number_of_nodes()} nodes")
                    self.graph = None
                    return False
            
            logger.error("No graph available!")
            self.graph = None
            self.nx_graph = None
            return False
    
    def _check_ollama(self):
        """Check if Ollama is available with the specified model"""
        try:
            models = ollama.list()
            model_names = [m['name'] for m in models['models']]
            if any(self.ollama_model in name for name in model_names):
                logger.info(f"Ollama model {self.ollama_model} is available")
            else:
                logger.warning(f"Ollama model {self.ollama_model} not found")
                logger.info(f"Available models: {model_names}")
        except Exception as e:
            logger.error(f"Ollama not available: {e}")
    
    def _load_embedding_cache(self) -> Dict:
        """Load embedding cache from disk"""
        if self.cache_embeddings and self.embedding_cache_file.exists():
            try:
                with open(self.embedding_cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(f"Loaded {len(cache)} cached embeddings")
                return cache
            except Exception as e:
                logger.warning(f"Could not load embedding cache: {e}")
        return {}
    
    def _save_embedding_cache(self):
        """Save embedding cache to disk"""
        if self.cache_embeddings:
            try:
                with open(self.embedding_cache_file, 'wb') as f:
                    pickle.dump(self.embedding_cache, f)
                logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
            except Exception as e:
                logger.warning(f"Could not save embedding cache: {e}")
    
    def analyze_query_intent(self, query: str) -> Dict:
        """Use Ollama to understand query intent and extract key concepts"""
        
        prompt = f"""
        Analyze this philosophical query and extract structured information.
        
        Query: "{query}"
        
        Please identify:
        1. Query type: (choose one)
           - debate_comparison (comparing opposing views)
           - concept_network (exploring related concepts)
           - philosopher_influence (tracing influences between thinkers)
           - historical_development (evolution of ideas over time)
           - school_comparison (comparing philosophical schools)
           - general (other philosophical queries)
        
        2. Key concepts mentioned (list all philosophical concepts)
        3. Philosophers mentioned (if any, with time periods if known)
        4. Schools of thought mentioned (if any)
        5. Relationships to explore (e.g., "critiques", "influences", "contrasts")
        6. Time period focus (if mentioned, e.g., "ancient", "modern", "20th century")
        7. Depth requested (surface-level overview vs deep analysis)
        
        Format as JSON.
        """
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={"temperature": 0.3, "num_predict": 500}
            )
            
            # Parse the response
            content = response['message']['content']
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                intent = json.loads(json_match.group())
                logger.info(f"Query intent analyzed: {intent}")
                return intent
            else:
                # Fallback parsing
                return self._fallback_intent_parsing(query)
                
        except Exception as e:
            logger.error(f"Failed to analyze query intent: {e}")
            return self._fallback_intent_parsing(query)
    
    def _fallback_intent_parsing(self, query: str) -> Dict:
        """Basic intent parsing without LLM"""
        query_lower = query.lower()
        
        intent = {
            'query_type': 'general',
            'key_concepts': [],
            'philosophers': [],
            'schools': [],
            'relationships': ['related_to'],
            'time_period': None,
            'depth': 'overview'
        }
        
        # Detect query type
        if any(word in query_lower for word in ['debate', 'vs', 'versus', 'compare']):
            intent['query_type'] = 'debate_comparison'
        elif any(word in query_lower for word in ['influence', 'influenced', 'impact']):
            intent['query_type'] = 'philosopher_influence'
        elif any(word in query_lower for word in ['history', 'development', 'evolution']):
            intent['query_type'] = 'historical_development'
        
        # Extract concepts (words ending in -ism, -ity, etc.)
        concept_pattern = r'\b(\w+(?:ism|ity|ology|ophy))\b'
        concepts = re.findall(concept_pattern, query_lower)
        intent['key_concepts'] = list(set(concepts))
        
        return intent
    
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Generate embeddings for texts using GPU acceleration with batching"""
        
        embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                # Check cache first
                if text in self.embedding_cache:
                    batch_embeddings.append(self.embedding_cache[text])
                else:
                    # Generate embedding
                    with torch.no_grad():
                        encoded = self.tokenizer(
                            text, 
                            padding=True, 
                            truncation=True, 
                            return_tensors='pt',
                            max_length=512
                        ).to(self.device)
                        
                        outputs = self.model(**encoded)
                        # Mean pooling
                        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                        
                        # Cache it
                        self.embedding_cache[text] = embedding
                        batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
        
        # Save cache periodically
        if len(embeddings) % 100 == 0:
            self._save_embedding_cache()
        
        return np.vstack(embeddings)
    
    def _initialize_article_embeddings(self):
        """Precompute embeddings for all articles"""
        if hasattr(self, '_article_embeddings'):
            return
        
        logger.info("Initializing article embeddings...")
        
        if self.use_neo4j:
            # Get all articles from Neo4j
            result = self.graph.run("""
                MATCH (a:Article)
                RETURN a.title as title, a.sections as sections, a.id as id
                LIMIT 2000
            """)
            
            articles = list(result)
        else:
            # Get from NetworkX
            articles = []
            for node, data in self.nx_graph.nodes(data=True):
                if data.get('type') == 'Article':
                    articles.append({
                        'id': node,
                        'title': data.get('title', ''),
                        'sections': data.get('sections', [])
                    })
        
        if articles:
            # Create text representations
            texts = []
            self._article_ids = []
            
            for article in articles:
                sections = article.get('sections', [])
                if isinstance(sections, str):
                    sections = json.loads(sections)
                
                text = f"{article['title']} {' '.join(sections[:5])}"
                texts.append(text)
                self._article_ids.append(article['id'])
            
            # Generate embeddings
            self._article_embeddings = self.get_embeddings(texts)
            logger.info(f"Computed embeddings for {len(articles)} articles")
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Find semantically similar articles using embeddings"""
        
        if not hasattr(self, '_article_embeddings'):
            self._initialize_article_embeddings()
        
        if not hasattr(self, '_article_embeddings') or self._article_embeddings is None:
            logger.warning("No article embeddings available")
            return []
        
        # Generate query embedding
        query_embedding = self.get_embeddings([query])[0]
        
        # Calculate cosine similarities
        similarities = np.dot(self._article_embeddings, query_embedding) / (
            np.linalg.norm(self._article_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            article_id = self._article_ids[idx]
            
            if self.use_neo4j:
                # Get article details from Neo4j
                article_data = self.graph.run("""
                    MATCH (a:Article {id: $id})
                    RETURN a
                """, id=article_id).data()
                
                if article_data:
                    node = article_data[0]['a']
                    results.append({
                        'id': article_id,
                        'title': node.get('title', ''),
                        'url': node.get('url', ''),
                        'similarity': float(similarities[idx]),
                        'authors': json.loads(node.get('authors', '[]'))
                    })
            else:
                # Get from NetworkX
                node_data = self.nx_graph.nodes.get(f"article:{article_id}", {})
                results.append({
                    'id': article_id,
                    'title': node_data.get('title', ''),
                    'url': node_data.get('url', ''),
                    'similarity': float(similarities[idx])
                })
        
        return results
    
    def find_debate_camps(self, concept1: str, concept2: str, depth: int = 2) -> QueryResult:
        """Find all articles and philosophers involved in a debate between two concepts"""
        
        logger.info(f"Analyzing debate: {concept1} vs {concept2}")
        
        if self.use_neo4j:
            # Neo4j query to find debate camps
            query = """
            // Find articles mentioning both concepts
            MATCH (a1:Article)-[:DISCUSSES|MENTIONS]-(c1)
            WHERE c1.name =~ $concept1_pattern
            WITH a1, c1
            
            MATCH (a2:Article)-[:DISCUSSES|MENTIONS]-(c2)
            WHERE c2.name =~ $concept2_pattern
            WITH a1, a2, c1, c2
            WHERE a1 = a2
            
            // Find philosophers and their positions
            MATCH (a1)-[:MENTIONS]-(p:Philosopher)
            
            // Find relationships between concepts
            OPTIONAL MATCH path = (c1)-[r*1..%d]-(c2)
            
            RETURN DISTINCT
                a1 as article,
                collect(DISTINCT p) as philosophers,
                c1, c2,
                relationships(path) as path_relationships
            LIMIT 50
            """ % depth
            
            results = self.graph.run(
                query,
                concept1_pattern=f"(?i).*{concept1}.*",
                concept2_pattern=f"(?i).*{concept2}.*"
            ).data()
            
            # Analyze positions using Ollama
            debate = self._analyze_debate_positions(results, concept1, concept2)
            
        else:
            # NetworkX implementation
            debate = self._find_debate_camps_networkx(concept1, concept2, depth)
        
        # Create visualization data
        vis_data = self._create_debate_visualization(debate)
        
        return QueryResult(
            nodes=debate.camps,
            relationships=debate.relationships,
            summary=self._generate_debate_summary(debate),
            confidence=0.8,
            sources=debate.central_texts,
            visualization_data=vis_data
        )
    
    def _analyze_debate_positions(self, results: List[Dict], concept1: str, concept2: str) -> PhilosophicalDebate:
        """Use Ollama to analyze philosophical positions in a debate"""
        
        debate = PhilosophicalDebate(
            topic=f"{concept1} vs {concept2}",
            camps={},
            key_arguments={},
            historical_development=[],
            central_texts=[],
            relationships=[]
        )
        
        # Collect all relevant text
        articles_text = []
        for result in results[:10]:  # Limit to top 10 for analysis
            article = result['article']
            philosophers = result.get('philosophers', [])
            
            article_summary = f"Article: {article.get('title', '')}\n"
            article_summary += f"Philosophers: {', '.join([p.get('name', '') for p in philosophers])}\n"
            articles_text.append(article_summary)
        
# sep_query_interface.py (continued from where it was cut off)

        # Use Ollama to analyze positions
        prompt = f"""
        Analyze the philosophical debate between {concept1} and {concept2} based on these articles:
        
        {chr(10).join(articles_text)}
        
        Identify:
        1. Main camps/positions in this debate
        2. Key philosophers in each camp
        3. Central arguments for each position
        4. How the debate has evolved historically
        
        Format as JSON with structure:
        {{
            "camps": {{"camp_name": ["philosopher1", "philosopher2"]}},
            "arguments": {{"camp_name": ["argument1", "argument2"]}},
            "timeline": [{{"period": "...", "development": "..."}}],
            "key_texts": ["text1", "text2"]
        }}
        """
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={"temperature": 0.3, "num_predict": 1000}
            )
            
            content = response['message']['content']
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                analysis = json.loads(json_match.group())
                debate.camps = analysis.get('camps', {})
                debate.key_arguments = analysis.get('arguments', {})
                debate.historical_development = analysis.get('timeline', [])
                debate.central_texts = analysis.get('key_texts', [])
                
        except Exception as e:
            logger.error(f"Failed to analyze debate positions: {e}")
        
        # Extract relationships between camps
        for result in results:
            if 'path_relationships' in result and result['path_relationships']:
                for rel in result['path_relationships']:
                    debate.relationships.append({
                        'source': rel.start_node.get('name', ''),
                        'target': rel.end_node.get('name', ''),
                        'type': type(rel).__name__
                    })
        
        return debate
    
    def _find_debate_camps_networkx(self, concept1: str, concept2: str, depth: int) -> PhilosophicalDebate:
        """Find debate camps using NetworkX graph"""
        
        debate = PhilosophicalDebate(
            topic=f"{concept1} vs {concept2}",
            camps={},
            key_arguments={},
            historical_development=[],
            central_texts=[],
            relationships=[]
        )
        
        # Find nodes related to each concept
        concept1_nodes = set()
        concept2_nodes = set()
        
        for node, data in self.nx_graph.nodes(data=True):
            node_name = data.get('name', '').lower()
            node_title = data.get('title', '').lower()
            
            if concept1.lower() in node_name or concept1.lower() in node_title:
                concept1_nodes.add(node)
            if concept2.lower() in node_name or concept2.lower() in node_title:
                concept2_nodes.add(node)
        
        # Find articles discussing both concepts
        common_articles = set()
        
        for c1_node in concept1_nodes:
            for c2_node in concept2_nodes:
                # Find shortest paths between concepts
                try:
                    paths = list(nx.all_shortest_paths(
                        self.nx_graph, c1_node, c2_node, cutoff=depth
                    ))
                    
                    for path in paths:
                        for node in path:
                            if self.nx_graph.nodes[node].get('type') == 'Article':
                                common_articles.add(node)
                except nx.NetworkXNoPath:
                    continue
        
        # Analyze each article for positions
        for article_node in list(common_articles)[:20]:  # Limit to 20 articles
            article_data = self.nx_graph.nodes[article_node]
            
            # Find connected philosophers
            philosophers = []
            for neighbor in self.nx_graph.neighbors(article_node):
                if self.nx_graph.nodes[neighbor].get('type') == 'Philosopher':
                    philosophers.append(self.nx_graph.nodes[neighbor].get('name', ''))
            
            # Simple classification based on title
            article_title = article_data.get('title', '')
            if concept1.lower() in article_title.lower():
                camp = f"Pro-{concept1}"
            elif concept2.lower() in article_title.lower():
                camp = f"Pro-{concept2}"
            else:
                camp = "Neutral/Synthetic"
            
            if camp not in debate.camps:
                debate.camps[camp] = []
            
            debate.camps[camp].extend(philosophers)
            debate.central_texts.append(article_title)
        
        return debate
    
    def find_concept_network(self, concept: str, depth: int = 2, max_nodes: int = 50) -> QueryResult:
        """Find network of related concepts"""
        
        logger.info(f"Building concept network for: {concept}")
        
        if self.use_neo4j:
            query = """
            MATCH (c:Concept)
            WHERE c.name =~ $concept_pattern
            WITH c
            MATCH path = (c)-[r*1..%d]-(related)
            WHERE related:Concept OR related:Article
            WITH c, related, relationships(path) as rels, length(path) as distance
            ORDER BY distance
            LIMIT %d
            RETURN 
                c,
                collect(DISTINCT {
                    node: related,
                    distance: distance,
                    path: [rel in rels | type(rel)]
                }) as connections
            """ % (depth, max_nodes)
            
            results = self.graph.run(
                query,
                concept_pattern=f"(?i).*{concept}.*"
            ).data()
            
        else:
            results = self._find_concept_network_networkx(concept, depth, max_nodes)
        
        # Build network structure
        nodes = []
        relationships = []
        
        for result in results:
            center = result.get('c', {})
            nodes.append({
                'id': center.get('name', concept),
                'type': 'Concept',
                'central': True
            })
            
            for conn in result.get('connections', []):
                related = conn['node']
                nodes.append({
                    'id': related.get('name', related.get('title', '')),
                    'type': 'Concept' if 'name' in related else 'Article',
                    'distance': conn['distance']
                })
                
                relationships.append({
                    'source': center.get('name', concept),
                    'target': related.get('name', related.get('title', '')),
                    'types': conn['path']
                })
        
        # Generate network summary
        summary = self._generate_network_summary(concept, nodes, relationships)
        
        return QueryResult(
            nodes=nodes,
            relationships=relationships,
            summary=summary,
            confidence=0.85,
            sources=[n['id'] for n in nodes if n.get('type') == 'Article'][:10]
        )
    
    def trace_influence_chain(self, philosopher: str, direction: str = "both") -> QueryResult:
        """Trace influence chains for a philosopher"""
        
        logger.info(f"Tracing influence chain for: {philosopher} (direction: {direction})")
        
        if self.use_neo4j:
            if direction == "forward":
                rel_pattern = "INFLUENCES"
            elif direction == "backward":
                rel_pattern = "INFLUENCED_BY"
            else:
                rel_pattern = "INFLUENCES|INFLUENCED_BY"
            
            query = """
            MATCH (p:Philosopher)
            WHERE p.name =~ $philosopher_pattern
            WITH p
            MATCH path = (p)-[r:%s*1..3]-(other:Philosopher)
            WITH p, other, relationships(path) as rels, length(path) as distance
            RETURN 
                p,
                collect(DISTINCT {
                    philosopher: other,
                    distance: distance,
                    relationship_chain: [rel in rels | type(rel)]
                }) as influences
            """ % rel_pattern
            
            results = self.graph.run(
                query,
                philosopher_pattern=f"(?i).*{philosopher}.*"
            ).data()
            
        else:
            results = self._trace_influence_networkx(philosopher, direction)
        
        # Build influence tree
        influence_tree = self._build_influence_tree(results)
        
        # Create visualization
        vis_data = self._create_influence_visualization(influence_tree)
        
        return QueryResult(
            nodes=influence_tree['nodes'],
            relationships=influence_tree['relationships'],
            summary=self._generate_influence_summary(philosopher, influence_tree),
            confidence=0.9,
            visualization_data=vis_data
        )
    
    def analyze_philosophical_schools(self) -> Dict[str, Any]:
        """Analyze all philosophical schools and their relationships"""
        
        logger.info("Analyzing philosophical schools...")
        
        if self.use_neo4j:
            # Get schools and their members
            schools_query = """
            MATCH (s:School)-[r]-(p:Philosopher)
            WITH s, collect(DISTINCT p.name) as members, count(DISTINCT p) as member_count
            ORDER BY member_count DESC
            RETURN s.name as school, members, member_count
            LIMIT 50
            """
            
            # Get school relationships
            relations_query = """
            MATCH (s1:School)-[r]-(s2:School)
            WHERE id(s1) < id(s2)
            RETURN s1.name as school1, s2.name as school2, type(r) as relationship
            """
            
            schools = self.graph.run(schools_query).data()
            relations = self.graph.run(relations_query).data()
            
        else:
            schools, relations = self._analyze_schools_networkx()
        
        # Cluster schools by similarity
        school_clusters = self._cluster_schools(schools, relations)
        
        return {
            'schools': schools,
            'relationships': relations,
            'clusters': school_clusters,
            'visualization': self._create_schools_visualization(schools, relations)
        }
    
    def execute_custom_query(self, natural_language_query: str) -> QueryResult:
        """Execute a custom query based on natural language"""
        
        # Analyze intent
        intent = self.analyze_query_intent(natural_language_query)
        
        # Route to appropriate method
        if intent['query_type'] == 'debate_comparison':
            concepts = intent.get('key_concepts', [])
            if len(concepts) >= 2:
                return self.find_debate_camps(concepts[0], concepts[1])
        
        elif intent['query_type'] == 'philosopher_influence':
            philosophers = intent.get('philosophers', [])
            if philosophers:
                return self.trace_influence_chain(philosophers[0])
        
        elif intent['query_type'] == 'concept_network':
            concepts = intent.get('key_concepts', [])
            if concepts:
                return self.find_concept_network(concepts[0])
        
        # Default: semantic search
        results = self.semantic_search(natural_language_query, top_k=20)
        
        return QueryResult(
            nodes=[{'id': r['id'], 'title': r['title'], 'score': r['similarity']} 
                   for r in results],
            summary=f"Found {len(results)} relevant articles for your query.",
            confidence=0.7,
            sources=[r['url'] for r in results[:5]]
        )
    
    def _create_debate_visualization(self, debate: PhilosophicalDebate) -> Dict:
        """Create visualization data for debate"""
        
        nodes = []
        edges = []
        
        # Add camp nodes
        for i, (camp_name, members) in enumerate(debate.camps.items()):
            nodes.append({
                'id': camp_name,
                'label': camp_name,
                'type': 'camp',
                'size': len(members) * 10,
                'color': ['#ff6b6b', '#4ecdc4', '#45b7d1'][i % 3]
            })
            
            # Add philosopher nodes
            for member in members:
                nodes.append({
                    'id': member,
                    'label': member,
                    'type': 'philosopher',
                    'size': 5
                })
                
                edges.append({
                    'source': member,
                    'target': camp_name,
                    'type': 'belongs_to'
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'layout': 'force-directed'
        }
    
    def _generate_debate_summary(self, debate: PhilosophicalDebate) -> str:
        """Generate a summary of the philosophical debate"""
        
        summary = f"## Philosophical Debate: {debate.topic}\n\n"
        
        # Describe camps
        summary += "### Main Positions:\n"
        for camp, members in debate.camps.items():
            summary += f"\n**{camp}**\n"
            summary += f"- Key figures: {', '.join(members[:5])}\n"
            
            if camp in debate.key_arguments:
                summary += f"- Main arguments:\n"
                for arg in debate.key_arguments[camp][:3]:
                    summary += f"  - {arg}\n"
        
        # Historical development
        if debate.historical_development:
            summary += "\n### Historical Development:\n"
            for item in debate.historical_development[:5]:
                summary += f"- **{item.get('period', 'Unknown period')}**: {item.get('development', '')}\n"
        
        # Central texts
        if debate.central_texts:
            summary += "\n### Key Texts:\n"
            for text in debate.central_texts[:10]:
                summary += f"- {text}\n"
        
        return summary