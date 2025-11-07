# Step 3: Graph Construction and Storage

# sep_knowledge_graph.py
from py2neo import Graph, Node, Relationship, NodeMatcher
import json
from typing import Dict, List, Set, Tuple
import logging
from datetime import datetime
import networkx as nx
from collections import defaultdict
import pickle

class SEPKnowledgeGraph:
    def __init__(self, neo4j_url="bolt://localhost:7687", 
                 username="neo4j", password="password",
                 fallback_to_networkx=True):
        self.logger = logging.getLogger(__name__)
        self.fallback_to_networkx = fallback_to_networkx
        self.nx_graph = None
        
        try:
            self.graph = Graph(neo4j_url, auth=(username, password))
            self.matcher = NodeMatcher(self.graph)
            self.logger.info("Connected to Neo4j successfully")
            self.use_neo4j = True
            
            # Create indexes for performance
            self._create_indexes()
        except Exception as e:
            self.logger.warning(f"Could not connect to Neo4j: {e}")
            if fallback_to_networkx:
                self.logger.info("Falling back to NetworkX")
                self.nx_graph = nx.DiGraph()
                self.use_neo4j = False
            else:
                raise
    
    def _create_indexes(self):
        """Create Neo4j indexes for better performance"""
        if not self.use_neo4j:
            return
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Article) ON (n.title)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Article) ON (n.url)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Philosopher) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Concept) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:School) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Argument) ON (n.name)",
        ]
        
        for index in indexes:
            try:
                self.graph.run(index)
            except Exception as e:
                self.logger.warning(f"Could not create index: {e}")
    
    def build_graph_from_extracted_data(self, all_articles_data: Dict):
        """Build the complete graph from extracted data"""
        self.logger.info(f"Building graph from {len(all_articles_data)} articles")
        
        if self.use_neo4j:
            self._build_neo4j_graph(all_articles_data)
        else:
            self._build_networkx_graph(all_articles_data)
        
        # Calculate graph statistics
        self._calculate_graph_stats()
        
        # Save backup
        self._save_backup(all_articles_data)
    
    def _build_neo4j_graph(self, all_articles_data: Dict):
        """Build graph in Neo4j"""
        # Clear existing data (optional)
        # self.graph.run("MATCH (n) DETACH DELETE n")
        
        # Batch transactions for performance
        tx = self.graph.begin()
        batch_size = 100
        node_count = 0
        
        try:
            # Phase 1: Create all article nodes
            self.logger.info("Creating article nodes...")
            for article_id, data in all_articles_data.items():
                metadata = data.get('metadata', {})
                
                article_node = Node(
                    "Article",
                    id=article_id,
                    title=metadata.get('title', ''),
                    url=metadata.get('url', ''),
                    authors=json.dumps(metadata.get('authors', [])),
                    pub_date=metadata.get('pubdate'),
                    last_updated=metadata.get('last_updated'),
                    word_count=data.get('word_count', 0),
                    sections=json.dumps(data.get('sections', []))
                )
                
                tx.create(article_node)
                node_count += 1
                
                if node_count % batch_size == 0:
                    tx.commit()
                    tx = self.graph.begin()
                    self.logger.info(f"Created {node_count} article nodes")
            
            # Phase 2: Create entity nodes
            self.logger.info("Creating entity nodes...")
            entity_nodes = {}
            
            for article_id, data in all_articles_data.items():
                entities = data.get('entities', {})
                
                # Create philosopher nodes
                for philosopher in entities.get('philosophers', []):
                    if philosopher not in entity_nodes:
                        node = Node("Philosopher", name=philosopher)
                        tx.merge(node, "Philosopher", "name")
                        entity_nodes[philosopher] = node
                
                # Create concept nodes
                for concept in entities.get('concepts', []):
                    if concept not in entity_nodes:
                        node = Node("Concept", name=concept)
                        tx.merge(node, "Concept", "name")
                        entity_nodes[concept] = node
                
                # Create school nodes
                for school in entities.get('schools', []):
                    if school not in entity_nodes:
                        node = Node("School", name=school)
                        tx.merge(node, "School", "name")
                        entity_nodes[school] = node
                
                # Create argument nodes
                for argument in entities.get('arguments', []):
                    if argument not in entity_nodes:
                        node = Node("Argument", name=argument)
                        tx.merge(node, "Argument", "name")
                        entity_nodes[argument] = node
                
                if len(entity_nodes) % batch_size == 0:
                    tx.commit()
                    tx = self.graph.begin()
                    self.logger.info(f"Created {len(entity_nodes)} entity nodes")
            
            # Phase 3: Create relationships
            self.logger.info("Creating relationships...")
            rel_count = 0
            
            for article_id, data in all_articles_data.items():
                # Get article node
                article_node = self.matcher.match("Article", id=article_id).first()
                if not article_node:
                    continue
                
                # Link article to entities
                entities = data.get('entities', {})
                
                for philosopher in entities.get('philosophers', []):
                    philosopher_node = self.matcher.match("Philosopher", name=philosopher).first()
                    if philosopher_node:
                        rel = Relationship(article_node, "MENTIONS", philosopher_node)
                        tx.create(rel)
                        rel_count += 1
                
                for concept in entities.get('concepts', []):
                    concept_node = self.matcher.match("Concept", name=concept).first()
                    if concept_node:
                        rel = Relationship(article_node, "DISCUSSES", concept_node)
                        tx.create(rel)
                        rel_count += 1
                
                # Create relationships between entities
                for rel_data in data.get('relationships', []):
                    source = rel_data.get('source')
                    target = rel_data.get('target')
                    rel_type = rel_data.get('type', 'RELATED_TO')
                    
                    # Find source and target nodes
                    source_node = None
                    target_node = None
                    
                    # Try to find nodes in different categories
                    for label in ['Article', 'Philosopher', 'Concept', 'School']:
                        if not source_node:
                            source_node = self.matcher.match(label, name=source).first()
                            if not source_node and label == 'Article':
                                source_node = self.matcher.match(label, title=source).first()
                        
                        if not target_node:
                            target_node = self.matcher.match(label, name=target).first()
                            if not target_node and label == 'Article':
                                target_node = self.matcher.match(label, title=target).first()
                    
                    if source_node and target_node:
                        rel = Relationship(
                            source_node, 
                            rel_type, 
                            target_node,
                            confidence=rel_data.get('confidence', 0.5),
                            context=rel_data.get('context', '')[:500]
                        )
                        tx.create(rel)
                        rel_count += 1
                
                if rel_count % batch_size == 0:
                    tx.commit()
                    tx = self.graph.begin()
                    self.logger.info(f"Created {rel_count} relationships")
            
            # Commit final batch
            tx.commit()
            self.logger.info(f"Graph building complete: {node_count} nodes, {rel_count} relationships")
            
        except Exception as e:
            self.logger.error(f"Error building graph: {e}")
            tx.rollback()
            raise
    
    def _build_networkx_graph(self, all_articles_data: Dict):
        """Build graph using NetworkX as fallback"""
        self.logger.info("Building NetworkX graph...")
        
        # Add nodes
        for article_id, data in all_articles_data.items():
            metadata = data.get('metadata', {})
            
            # Add article node
            self.nx_graph.add_node(
                f"article:{article_id}",
                type='Article',
                title=metadata.get('title', ''),
                url=metadata.get('url', ''),
                authors=metadata.get('authors', []),
                pub_date=metadata.get('pubdate'),
                word_count=data.get('word_count', 0)
            )
            
            # Add entity nodes
            entities = data.get('entities', {})
            
            for philosopher in entities.get('philosophers', []):
                self.nx_graph.add_node(f"philosopher:{philosopher}", type='Philosopher', name=philosopher)
                self.nx_graph.add_edge(f"article:{article_id}", f"philosopher:{philosopher}", type='MENTIONS')
            
            for concept in entities.get('concepts', []):
                self.nx_graph.add_node(f"concept:{concept}", type='Concept', name=concept)
                self.nx_graph.add_edge(f"article:{article_id}", f"concept:{concept}", type='DISCUSSES')
            
            for school in entities.get('schools', []):
                self.nx_graph.add_node(f"school:{school}", type='School', name=school)
                self.nx_graph.add_edge(f"article:{article_id}", f"school:{school}", type='DISCUSSES')
            
            # Add relationships
            for rel_data in data.get('relationships', []):
                source = rel_data.get('source')
                target = rel_data.get('target')
                rel_type = rel_data.get('type', 'RELATED_TO')
                
                # Create edge with attributes
                self.nx_graph.add_edge(
                    f"entity:{source}",
                    f"entity:{target}",
                    type=rel_type,
                    confidence=rel_data.get('confidence', 0.5),
                    context=rel_data.get('context', '')
                )
        
        self.logger.info(f"NetworkX graph built: {self.nx_graph.number_of_nodes()} nodes, {self.nx_graph.number_of_edges()} edges")
    
    def _calculate_graph_stats(self):
        """Calculate and log graph statistics"""
        if self.use_neo4j:
            stats = {
                'total_nodes': self.graph.run("MATCH (n) RETURN count(n) as count").data()[0]['count'],
                'total_relationships': self.graph.run("MATCH ()-[r]->() RETURN count(r) as count").data()[0]['count'],
                'articles': self.graph.run("MATCH (n:Article) RETURN count(n) as count").data()[0]['count'],
                'philosophers': self.graph.run("MATCH (n:Philosopher) RETURN count(n) as count").data()[0]['count'],
                'concepts': self.graph.run("MATCH (n:Concept) RETURN count(n) as count").data()[0]['count'],
                'schools': self.graph.run("MATCH (n:School) RETURN count(n) as count").data()[0]['count'],
            }
            
            # Most connected nodes
            most_connected = self.graph.run("""
                MATCH (n)-[r]-()
                RETURN n.name as name, labels(n)[0] as label, count(r) as degree
                ORDER BY degree DESC
                LIMIT 10
            """).data()
            
            stats['most_connected'] = most_connected
        else:
            stats = {
                'total_nodes': self.nx_graph.number_of_nodes(),
                'total_edges': self.nx_graph.number_of_edges(),
                'density': nx.density(self.nx_graph)
            }
            
            # Most connected nodes
            degree_dict = dict(self.nx_graph.degree())
            most_connected = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            stats['most_connected'] = most_connected
        
        self.logger.info(f"Graph statistics: {json.dumps(stats, indent=2)}")
        
        # Save stats
        with open('graph_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _save_backup(self, all_articles_data: Dict):
        """Save backup of the graph data"""
        backup_file = f"sep_graph_backup_{datetime.now():%Y%m%d_%H%M%S}.pkl"
        
        backup_data = {
            'articles': all_articles_data,
            'timestamp': datetime.now().isoformat(),
            'use_neo4j': self.use_neo4j
        }
        
        if not self.use_neo4j:
            backup_data['networkx_graph'] = self.nx_graph
        
        with open(backup_file, 'wb') as f:
            pickle.dump(backup_data, f)
        
        self.logger.info(f"Backup saved to {backup_file}")
    
    def export_to_formats(self, output_dir: str = "exports"):
        """Export graph to various formats"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if self.use_neo4j:
            # Export to Cypher
            cypher_file = os.path.join(output_dir, "sep_graph.cypher")
            self._export_to_cypher(cypher_file)
            
            # Export to GraphML
            graphml_file = os.path.join(output_dir, "sep_graph.graphml")
            self._export_to_graphml(graphml_file)
        else:
            # Export NetworkX graph
            nx.write_graphml(self.nx_graph, os.path.join(output_dir, "sep_graph.graphml"))
            nx.write_gexf(self.nx_graph, os.path.join(output_dir, "sep_graph.gexf"))
        
        self.logger.info(f"Graph exported to {output_dir}")
    
    def _export_to_cypher(self, filename: str):
        """Export graph to Cypher script"""
        with open(filename, 'w') as f:
            # Export nodes
            f.write("// Nodes\n")
            
            # Articles
            articles = self.graph.run("MATCH (n:Article) RETURN n").data()
            for record in articles:
                node = record['n']
                props = dict(node)
                f.write(f"CREATE (n:Article {props});\n")
            
            # Other entities
            for label in ['Philosopher', 'Concept', 'School', 'Argument']:
                nodes = self.graph.run(f"MATCH (n:{label}) RETURN n").data()
                for record in nodes:
                    node = record['n']
                    props = dict(node)
                    f.write(f"CREATE (n:{label} {props});\n")
            
            # Export relationships
            f.write("\n// Relationships\n")
            relationships = self.graph.run("MATCH (a)-[r]->(b) RETURN a, r, b").data()
            
            for record in relationships:
                a = dict(record['a'])
                b = dict(record['b'])
                r = record['r']
                rel_type = type(r).__name__
                rel_props = dict(r)
                
                f.write(f"MATCH (a {a}), (b {b}) CREATE (a)-[:{rel_type} {rel_props}]->(b);\n")
    
    def _export_to_graphml(self, filename: str):
        """Export to GraphML format"""
        # Convert to NetworkX first
        G = nx.DiGraph()
        
        # Add nodes
        nodes = self.graph.run("MATCH (n) RETURN n, labels(n) as labels").data()
        for record in nodes:
            node = record['n']
            labels = record['labels']
            node_id = f"{labels[0]}:{node.get('name', node.get('title', node.get('id', '')))}"
            
            G.add_node(node_id, **dict(node), label=labels[0])
        
        # Add edges
        relationships = self.graph.run("MATCH (a)-[r]->(b) RETURN a, r, b, labels(a) as la, labels(b) as lb").data()
        for record in relationships:
            a = record['a']
            b = record['b']
            r = record['r']
            la = record['la'][0]
            lb = record['lb'][0]
            
            source_id = f"{la}:{a.get('name', a.get('title', a.get('id', '')))}"
            target_id = f"{lb}:{b.get('name', b.get('title', b.get('id', '')))}"
            
            G.add_edge(source_id, target_id, type=type(r).__name__, **dict(r))
        
        nx.write_graphml(G, filename)