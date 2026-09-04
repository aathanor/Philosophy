# Query Analysis Module

# sep_query_analysis.py
"""
Query analysis and intent understanding module
"""

import re
import json
import logging
from typing import Dict, List, Optional
import ollama

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    def __init__(self, ollama_model: str = "nous-hermes2:10.7b"):
        self.ollama_model = ollama_model
        self.query_patterns = self._init_query_patterns()
    
    def _init_query_patterns(self) -> Dict[str, List[str]]:
        """Initialize query pattern recognition"""
        return {
            'debate_comparison': [
                r'(\w+)\s+vs\.?\s+(\w+)',
                r'compare\s+(\w+)\s+and\s+(\w+)',
                r'debate\s+between\s+(\w+)\s+and\s+(\w+)',
                r'(\w+)\s+versus\s+(\w+)'
            ],
            'philosopher_influence': [
                r'influence\s+of\s+(\w+\s*\w*)',
                r'(\w+\s*\w*)\s+influenced',
                r'who\s+influenced\s+(\w+\s*\w*)',
                r'(\w+\s*\w*)\'s\s+influence'
            ],
            'concept_network': [
                r'concepts?\s+related\s+to\s+(\w+)',
                r'(\w+)\s+and\s+related',
                r'network\s+of\s+(\w+)',
                r'connections?\s+to\s+(\w+)'
            ],
            'historical_development': [
                r'history\s+of\s+(\w+)',
                r'development\s+of\s+(\w+)',
                r'evolution\s+of\s+(\w+)',
                r'how\s+(\w+)\s+developed'
            ]
        }
    
    def analyze_query(self, query: str) -> Dict:
        """Analyze query using both patterns and LLM"""
        
        # First try pattern matching
        pattern_result = self._pattern_based_analysis(query)
        
        # Then enhance with LLM if available
        try:
            llm_result = self._llm_based_analysis(query)
            
            # Merge results
            return self._merge_analyses(pattern_result, llm_result)
            
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            return pattern_result
    
    def _pattern_based_analysis(self, query: str) -> Dict:
        """Basic pattern-based query analysis"""
        
        query_lower = query.lower()
        result = {
            'query_type': 'general',
            'entities': [],
            'concepts': [],
            'philosophers': [],
            'relationships': [],
            'time_period': None
        }
        
        # Check query patterns
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    result['query_type'] = query_type
                    result['entities'].extend(match.groups())
                    break
        
        # Extract philosophical concepts
        concept_pattern = r'\b(\w+(?:ism|ity|ology|ophy))\b'
        concepts = re.findall(concept_pattern, query_lower)
        result['concepts'] = list(set(concepts))
        
        # Extract potential philosopher names
        name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        names = re.findall(name_pattern, query)
        result['philosophers'] = [n for n in names if len(n.split()) <= 3]
        
        return result
    
    def _llm_based_analysis(self, query: str) -> Dict:
        """LLM-based query analysis"""
        
        prompt = f"""
        Analyze this philosophical query and extract structured information:
        
        Query: "{query}"
        
        Extract:
        1. Query type (debate_comparison, concept_network, philosopher_influence, historical_development, school_comparison, general)
        2. Key philosophical concepts
        3. Philosophers mentioned
        4. Schools of thought
        5. Relationships to explore
        6. Time period (if any)
        7. Required depth (overview, detailed, comprehensive)
        
        Output as JSON only.
        """
        
        response = ollama.chat(
            model=self.ollama_model,
            messages=[{'role': 'user', 'content': prompt}],
            options={"temperature": 0.3, "num_predict": 500}
        )
        
        content = response['message']['content']
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        
        if json_match:
            return json.loads(json_match.group())
        
        return {}
    
    def _merge_analyses(self, pattern_result: Dict, llm_result: Dict) -> Dict:
        """Merge pattern and LLM analysis results"""
        
        merged = pattern_result.copy()
        
        # Use LLM query type if more specific
        if llm_result.get('query_type', 'general') != 'general':
            merged['query_type'] = llm_result['query_type']
        
        # Merge lists
        for key in ['concepts', 'philosophers', 'entities']:
            merged[key] = list(set(
                pattern_result.get(key, []) + 
                llm_result.get(key, [])
            ))
        
        # Take LLM values for these
        for key in ['time_period', 'depth', 'relationships']:
            if key in llm_result:
                merged[key] = llm_result[key]
        
        return merged