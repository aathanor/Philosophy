# Step 2: Automated Entity and Relationship Extraction

# philosophy_kg_builder.py
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch
import networkx as nx
import re
import ollama
import json
from typing import Dict, List, Set
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class PhilosophyKGBuilder:
    def __init__(self, use_gpu=True, use_ollama=True):
        self.logger = logging.getLogger(__name__)
        
        # Detect and use GPU if available
        self.device = 0 if torch.cuda.is_available() and use_gpu else -1
        if self.device == 0:
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("Using CPU")
        
        # Load spaCy with GPU support
        spacy.require_gpu()  # Use GPU for spaCy if available
        try:
            self.nlp = spacy.load("en_core_web_trf")  # Transformer model
            self.logger.info("Loaded spaCy transformer model")
        except:
            self.nlp = spacy.load("en_core_web_lg")
            self.logger.info("Loaded spaCy large model")
        
        # Load transformer models with optimization
        self.logger.info("Loading transformer models...")
        
        # NER model
        self.ner_pipeline = pipeline(
            "ner", 
            model="dslim/bert-large-NER",
            device=self.device,
            batch_size=16  # Process multiple texts at once
        )
        
        # Relation extraction - using smaller model for efficiency
        try:
            self.relation_pipeline = pipeline(
                "text2text-generation",
                model="Babelscape/rebel-large",
                device=self.device,
                max_length=256
            )
            self.use_rebel = True
        except:
            self.logger.warning("Could not load REBEL model, using patterns only")
            self.use_rebel = False
        
        # Ollama integration
        self.use_ollama = use_ollama and self._check_ollama()
        if self.use_ollama:
            self.llm_model = "nous-hermes2:10.7b"
            self.logger.info(f"Using Ollama model: {self.llm_model}")
        
        # Philosophy-specific patterns
        self._init_philosophy_patterns()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=8)
    
    def _check_ollama(self):
        """Check if Ollama is available"""
        try:
            ollama.list()
            return True
        except:
            self.logger.warning("Ollama not available")
            return False
    
    def _init_philosophy_patterns(self):
        """Initialize philosophy-specific patterns"""
        self.philosopher_patterns = [
            r"([A-Z][a-z]+ (?:van |de |von )?[A-Z][a-z]+)\s*\([\d\-–]+\)",
            r"According to ([A-Z][a-z]+ [A-Z][a-z]+)",
            r"([A-Z][a-z]+ [A-Z][a-z]+)(?:'s|'s) (?:theory|philosophy|argument|view)",
            r"([A-Z][a-z]+ [A-Z][a-z]+) (?:argues|argued|claims|claimed|maintains|maintained)",
        ]
        
        self.concept_patterns = [
            r"\b(\w+ism)\b",
            r"\b(\w+ology)\b",
            r"\b(meta\w+)\b",
            r"\b(\w+ness)\b(?:\s+(?:thesis|theory|principle))?",
            r"\bthe (\w+) (?:problem|paradox|principle|theory|thesis|argument)\b",
        ]
        
        self.school_indicators = [
            'school', 'movement', 'tradition', 'approach', 'philosophy',
            'ism', 'ists', 'ians', 'ean', 'ic philosophy'
        ]
        
        self.relation_keywords = {
            'CRITIQUES': ['criticize', 'critique', 'object', 'against', 'reject', 'refute', 'challenge', 'dispute'],
            'SUPPORTS': ['support', 'agree', 'endorse', 'advocate', 'defend', 'maintain', 'uphold'],
            'INFLUENCED_BY': ['influenced by', 'inspired by', 'follows', 'based on', 'derived from', 'student of'],
            'INFLUENCES': ['influenced', 'inspired', 'led to', 'gave rise to', 'teacher of'],
            'OPPOSES': ['oppose', 'contrary to', 'against', 'incompatible with', 'conflicts with'],
            'DEVELOPS': ['develops', 'extends', 'builds on', 'elaborates', 'expands'],
            'RELATED_TO': ['related to', 'connected to', 'associated with', 'linked to'],
        }
    
    def extract_entities(self, text: str, metadata: Dict) -> Dict[str, Set[str]]:
        """Extract philosophical entities using multiple methods"""
        entities = {
            'philosophers': set(),
            'concepts': set(),
            'schools': set(),
            'arguments': set(),
            'works': set(),
            'problems': set()
        }
        
        # Limit text length for processing
        text_chunk = text[:50000]  # Process first 50k chars
        
        # 1. Pattern-based extraction (fast)
        self._extract_with_patterns(text_chunk, entities)
        
        # 2. SpaCy NER
        self._extract_with_spacy(text_chunk, entities)
        
        # 3. Transformer NER (if GPU available)
        if self.device == 0:  # Only on GPU
            self._extract_with_transformers(text_chunk, entities)
        
        # 4. Extract from metadata and structure
        self._extract_from_metadata(metadata, entities)
        
        # 5. LLM enhancement (if available)
        if self.use_ollama and len(entities['philosophers']) < 5:
            self._enhance_with_llm(text_chunk[:5000], metadata, entities)
        
        # Clean and validate entities
        self._clean_entities(entities)
        
        return entities
    
    def _extract_with_patterns(self, text: str, entities: Dict[str, Set[str]]):
        """Fast pattern-based extraction"""
        # Philosophers
        for pattern in self.philosopher_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['philosophers'].update(m for m in matches if len(m.split()) <= 4)
        
        # Concepts
        for pattern in self.concept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['concepts'].update(m.lower() for m in matches if 4 < len(m) < 30)
        
        # Arguments and problems
        argument_pattern = r"the (\w+(?:\s+\w+){0,3}) argument"
        problem_pattern = r"the (?:problem|paradox) of (\w+(?:\s+\w+){0,3})"
        
        entities['arguments'].update(re.findall(argument_pattern, text, re.IGNORECASE))
        entities['problems'].update(re.findall(problem_pattern, text, re.IGNORECASE))
    
    def _extract_with_spacy(self, text: str, entities: Dict[str, Set[str]]):
        """SpaCy-based entity extraction"""
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Filter to likely philosophers
                if any(word in ent.text.lower() for word in ['philosopher', 'century', 'bc', 'ad']):
                    entities['philosophers'].add(ent.text)
            elif ent.label_ == "ORG":
                # Could be a philosophical school
                if any(indicator in ent.text.lower() for indicator in self.school_indicators):
                    entities['schools'].add(ent.text)
            elif ent.label_ == "WORK_OF_ART":
                entities['works'].add(ent.text)
        
        # Extract noun phrases that might be concepts
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ == "NOUN" and len(chunk.text) > 5:
                if any(chunk.text.lower().endswith(suffix) for suffix in ['ism', 'ity', 'ness']):
                    entities['concepts'].add(chunk.text.lower())
    
    def _extract_with_transformers(self, text: str, entities: Dict[str, Set[str]]):
        """Transformer-based NER extraction"""
        # Process in chunks for efficiency
        chunk_size = 512
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)][:20]  # Max 20 chunks
        
        try:
            results = self.ner_pipeline(chunks, batch_size=8, truncation=True)
            
            for chunk_results in results:
                current_entity = []
                current_label = None
                
                for item in chunk_results:
                    if item['entity'].startswith('B-'):
                        if current_entity and current_label == 'PER':
                            entities['philosophers'].add(' '.join(current_entity))
                        current_entity = [item['word'].replace('##', '')]
                        current_label = item['entity'][2:]
                    elif item['entity'].startswith('I-') and current_label:
                        current_entity.append(item['word'].replace('##', ''))
                
                if current_entity and current_label == 'PER':
                    entities['philosophers'].add(' '.join(current_entity))
        except Exception as e:
            self.logger.warning(f"Transformer NER failed: {e}")
    
    def _extract_from_metadata(self, metadata: Dict, entities: Dict[str, Set[str]]):
        """Extract entities from article metadata"""
        # Add article title as potential concept
        if 'title' in metadata:
            title = metadata['title']
            if any(suffix in title.lower() for suffix in ['ism', 'philosophy', 'theory']):
                entities['concepts'].add(title.lower())
        
        # Extract from internal links
        if 'internal_links' in metadata:
            for link in metadata['internal_links']:
                target = link['target'].replace('-', ' ')
                anchor = link['anchor_text']
                
                # Classify based on context and anchor text
                if any(word in anchor.lower() for word in ['philosopher', 'century']):
                    entities['philosophers'].add(anchor)
                elif anchor.lower().endswith('ism'):
                    entities['schools'].add(anchor)
                else:
                    entities['concepts'].add(anchor.lower())
        
        # Extract from sections
        if 'sections' in metadata:
            for section in metadata['sections']:
                if 'argument' in section.lower():
                    entities['arguments'].add(section)
    
    def _enhance_with_llm(self, text: str, metadata: Dict, entities: Dict[str, Set[str]]):
        """Use Ollama for sophisticated entity extraction"""
        try:
            prompt = f"""
            Analyze this philosophical text and extract entities.
            
            Article: {metadata.get('title', 'Unknown')}
            Text excerpt: {text[:2000]}
            
            Extract:
            1. Philosophers mentioned (full names)
            2. Philosophical concepts/theories
            3. Schools of thought
            4. Key arguments
            5. Important works cited
            
            Format as JSON with keys: philosophers, concepts, schools, arguments, works
            """
            
            response = ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={"temperature": 0.3, "num_predict": 500}
            )
            
            # Parse response
            content = response['message']['content']
            
            # Try to extract JSON
            import json
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group())
                
                for key in ['philosophers', 'concepts', 'schools', 'arguments', 'works']:
                    if key in extracted and isinstance(extracted[key], list):
                        entities[key].update(extracted[key])
            
        except Exception as e:
            self.logger.warning(f"LLM extraction failed: {e}")
    
    def _clean_entities(self, entities: Dict[str, Set[str]]):
        """Clean and validate extracted entities"""
        # Remove common false positives
        false_positives = {'the', 'a', 'an', 'this', 'that', 'these', 'those'}
        
        for category in entities:
            # Remove short entities and false positives
            entities[category] = {
                e for e in entities[category] 
                if len(e) > 2 and e.lower() not in false_positives
            }
            
            # Limit each category to top entities
            if len(entities[category]) > 50:
                # Keep only the 50 most common (this is simplified)
                entities[category] = set(list(entities[category])[:50])
    
    def extract_relationships(self, article_data: Dict) -> List[Dict]:
        """Extract relationships between entities"""
        relationships = []
        text = article_data.get('full_text', '')
        
        # 1. Extract from internal links context
        relationships.extend(self._extract_from_links(article_data))
        
        # 2. Pattern-based extraction
        relationships.extend(self._extract_with_patterns_relations(text[:20000]))
        
        # 3. Transformer-based (if available and on GPU)
        if self.use_rebel and self.device == 0:
            relationships.extend(self._extract_with_transformer_relations(text[:10000]))
        
        # 4. LLM-based deep analysis for key relationships
        if self.use_ollama and len(relationships) < 10:
            relationships.extend(self._extract_with_llm_relations(article_data))
        
        # Deduplicate and score relationships
        relationships = self._deduplicate_relationships(relationships)
        
        return relationships
    
    def _extract_from_links(self, article_data: Dict) -> List[Dict]:
        """Extract relationships from internal links"""
        relationships = []
        source_title = article_data.get('metadata', {}).get('title', '')
        
        if not source_title:
            return relationships
        
        for link in article_data.get('internal_links', []):
            context = link.get('context', '').lower()
            target = link.get('anchor_text', '')
            
            if not target:
                continue
            
            # Determine relationship type from context
            rel_type = 'RELATED_TO'
            confidence = 0.5
            
            for rel, keywords in self.relation_keywords.items():
                if any(kw in context for kw in keywords):
                    rel_type = rel
                    confidence = 0.8
                    break
            
            relationships.append({
                'source': source_title,
                'target': target,
                'type': rel_type,
                'confidence': confidence,
                'context': context[:200],
                'section': link.get('section', 'Unknown')
            })
        
        return relationships
    
    def _extract_with_patterns_relations(self, text: str) -> List[Dict]:
        """Pattern-based relationship extraction"""
        relationships = []
        
        # Influence patterns
        influence_patterns = [
            r"([A-Z][a-z]+ [A-Z][a-z]+) influenced ([A-Z][a-z]+ [A-Z][a-z]+)",
            r"([A-Z][a-z]+ [A-Z][a-z]+) was influenced by ([A-Z][a-z]+ [A-Z][a-z]+)",
            r"([A-Z][a-z]+ [A-Z][a-z]+)'s influence on ([A-Z][a-z]+ [A-Z][a-z]+)",
        ]
        
        for pattern in influence_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                relationships.append({
                    'source': match[0],
                    'target': match[1],
                    'type': 'INFLUENCES',
                    'confidence': 0.9,
                    'context': f"Pattern match: {pattern}"
                })
        
        # Critique patterns
        critique_patterns = [
            r"([A-Z][a-z]+ [A-Z][a-z]+) (?:criticizes|criticized|critiques) ([A-Z][a-z]+ [A-Z][a-z]+)",
            r"([A-Z][a-z]+ [A-Z][a-z]+)'s critique of ([A-Z][a-z]+ [A-Z][a-z]+)",
        ]
        
        for pattern in critique_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                relationships.append({
                    'source': match[0],
                    'target': match[1],
                    'type': 'CRITIQUES',
                    'confidence': 0.9,
                    'context': f"Pattern match: {pattern}"
                })
        
        return relationships
    
    def _extract_with_transformer_relations(self, text: str) -> List[Dict]:
        """Use REBEL for relationship extraction"""
        relationships = []
        
        try:
            # Split text into sentences
            sentences = text.split('.')[:50]  # Process first 50 sentences
            
            # Batch process
            batch_size = 8
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                
                outputs = self.relation_pipeline(
                    batch,
                    max_length=256,
                    num_beams=3,
                    num_return_sequences=1
                )
                
                for output in outputs:
                    # Parse REBEL output format
                    triplets = self._parse_rebel_output(output['generated_text'])
                    relationships.extend(triplets)
        
        except Exception as e:
            self.logger.warning(f"REBEL extraction failed: {e}")
        
        return relationships
    
    def _parse_rebel_output(self, text: str) -> List[Dict]:
        """Parse REBEL model output"""
        relationships = []
        
        # REBEL outputs in format: <triplet> subject | predicate | object </triplet>
        triplet_pattern = r'<triplet>\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*</triplet>'
        matches = re.findall(triplet_pattern, text)
        
        for match in matches:
            subject, predicate, obj = match
            
            # Map REBEL predicates to our relationship types
            rel_type = 'RELATED_TO'
            if 'influence' in predicate.lower():
                rel_type = 'INFLUENCES'
            elif 'critic' in predicate.lower():
                rel_type = 'CRITIQUES'
            elif 'student' in predicate.lower() or 'teacher' in predicate.lower():
                rel_type = 'INFLUENCED_BY'
            
            relationships.append({
                'source': subject.strip(),
                'target': obj.strip(),
                'type': rel_type,
                'confidence': 0.7,
                'context': f"REBEL: {predicate}"
            })
        
        return relationships
    
    def _extract_with_llm_relations(self, article_data: Dict) -> List[Dict]:
        """Use LLM for deep relationship analysis"""
        relationships = []
        
        try:
            title = article_data.get('metadata', {}).get('title', '')
            text = article_data.get('full_text', '')[:3000]
            
            prompt = f"""
            Analyze the philosophical relationships in this article about "{title}".
            
            Text: {text}
            
            Extract relationships between philosophers, concepts, and schools of thought.
            For each relationship, specify:
            - Source entity
            - Target entity  
            - Relationship type (influences, critiques, develops, opposes, etc.)
            - Brief explanation
            
            Format as JSON array with objects containing: source, target, type, explanation
            """
            
            response = ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={"temperature": 0.3, "num_predict": 800}
            )
            
            # Parse response
            content = response['message']['content']
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            
            if json_match:
                extracted = json.loads(json_match.group())
                
                for rel in extracted:
                    if all(k in rel for k in ['source', 'target', 'type']):
                        relationships.append({
                            'source': rel['source'],
                            'target': rel['target'],
                            'type': rel['type'].upper().replace(' ', '_'),
                            'confidence': 0.8,
                            'context': rel.get('explanation', '')[:200]
                        })
        
        except Exception as e:
            self.logger.warning(f"LLM relationship extraction failed: {e}")
        
        return relationships
    
    def _deduplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Remove duplicate relationships, keeping highest confidence"""
        unique = {}
        
        for rel in relationships:
            key = (rel['source'], rel['target'], rel['type'])
            
            if key not in unique or rel.get('confidence', 0) > unique[key].get('confidence', 0):
                unique[key] = rel
        
        return list(unique.values())
    
    def batch_process_articles(self, articles: List[Dict], batch_size: int = 10) -> Dict:
        """Process multiple articles in parallel"""
        results = {}
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all tasks
            future_to_article = {
                executor.submit(self._process_single_article, article): article_id
                for article_id, article in articles.items()
            }
            
            # Collect results
            for future in future_to_article:
                article_id = future_to_article[future]
                try:
                    result = future.result()
                    results[article_id] = result
                except Exception as e:
                    self.logger.error(f"Failed to process {article_id}: {e}")
        
        return results
    
    def _process_single_article(self, article_data: Dict) -> Dict:
        """Process a single article"""
        entities = self.extract_entities(
            article_data.get('full_text', ''),
            article_data.get('metadata', {})
        )
        
        relationships = self.extract_relationships(article_data)
        
        return {
            'entities': entities,
            'relationships': relationships
        }