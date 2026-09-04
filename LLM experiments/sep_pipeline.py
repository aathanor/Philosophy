# Step 4: Automated Pipeline Runner

# sep_pipeline.py
import asyncio
import logging
from datetime import datetime
import json
from pathlib import Path
from tqdm import tqdm
import time
import sys
import os
import gc
import psutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# Import our modules
from sep_crawler import SEPCrawler
from philosophy_kg_builder import PhilosophyKGBuilder  
from sep_knowledge_graph import SEPKnowledgeGraph

class SEPGraphBuilder:
    def __init__(self, 
                 resume_from_checkpoint=True,
                 use_gpu=True,
                 use_ollama=True,
                 max_workers=8):
        
        # Set up logging
        self.setup_logging()
        
        self.resume_from_checkpoint = resume_from_checkpoint
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Performance settings for your machine
        self.max_workers = max_workers  # CPU workers
        self.use_gpu = use_gpu
        self.use_ollama = use_ollama
        
        # Initialize components
        self.logger.info("Initializing SEP Graph Builder...")
        self.logger.info(f"System info: {mp.cpu_count()} CPUs, {psutil.virtual_memory().total / 1e9:.1f}GB RAM")
        
        try:
            self.crawler = SEPCrawler(max_concurrent=20)  # High concurrency for your connection
            self.logger.info("✓ Crawler initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize crawler: {e}")
            raise
        
        try:
            self.kg_builder = PhilosophyKGBuilder(use_gpu=use_gpu, use_ollama=use_ollama)
            self.logger.info("✓ KG Builder initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize KG builder: {e}")
            self.logger.warning("Continuing with reduced functionality")
            self.kg_builder = None
        
        try:
            # Try to connect to Neo4j
            self.graph = SEPKnowledgeGraph(
                neo4j_url="bolt://localhost:7687",
                username="neo4j",
                password="password",  # Change this
                fallback_to_networkx=True
            )
            self.logger.info("✓ Knowledge Graph initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize graph: {e}")
            self.graph = None
        
        # Memory monitoring
        self.monitor_memory = True
        self.memory_threshold = 0.8  # Use max 80% of RAM
    
    def setup_logging(self):
        """Configure logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"sep_pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging to {log_file}")
    
    def check_memory(self):
        """Check memory usage and clean if needed"""
        memory_percent = psutil.virtual_memory().percent / 100
        
        if memory_percent > self.memory_threshold:
            self.logger.warning(f"High memory usage: {memory_percent:.1%}")
            gc.collect()
            time.sleep(2)
            
            # If still high, save checkpoint and exit
            if psutil.virtual_memory().percent / 100 > 0.9:
                self.logger.error("Critical memory usage - saving and exiting")
                return False
        
        return True
    
    def load_checkpoint(self):
        """Load the latest checkpoint if available"""
        checkpoints = list(self.checkpoint_dir.glob("sep_data_checkpoint_*.json"))
        if not checkpoints:
            return {}, 0
        
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        self.logger.info(f"Loading checkpoint: {latest_checkpoint}")
        
        with open(latest_checkpoint, 'r') as f:
            data = json.load(f)
        
        # Extract the index from filename
        index = int(latest_checkpoint.stem.split('_')[-1])
        
        return data, index
    
    def save_checkpoint(self, data: dict, index: int):
        """Save checkpoint"""
        checkpoint_file = self.checkpoint_dir / f'sep_data_checkpoint_{index}.json'
        
        self.logger.info(f"Saving checkpoint at index {index}...")
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints(keep_latest=5)
        
        self.logger.info("Checkpoint saved")
    
    def _cleanup_old_checkpoints(self, keep_latest=5):
        """Remove old checkpoints to save space"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("sep_data_checkpoint_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for old_checkpoint in checkpoints[keep_latest:]:
            old_checkpoint.unlink()
            self.logger.info(f"Removed old checkpoint: {old_checkpoint}")
    
    async def run_full_pipeline(self):
        """Run the complete automated pipeline with optimizations"""
        start_time = time.time()
        
        # Step 1: Get all entries
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: Fetching all SEP entries...")
        self.logger.info("=" * 60)
        
        entries = self.crawler.get_all_entries()
        self.logger.info(f"Found {len(entries)} entries")
        
        # Step 2: Check for existing progress
        all_data = {}
        start_index = 0
        
        if self.resume_from_checkpoint:
            all_data, start_index = self.load_checkpoint()
            if start_index > 0:
                self.logger.info(f"Resuming from entry {start_index}")
                # Skip already processed entries
                processed_ids = set(all_data.keys())
                entries = [e for e in entries if e['id'] not in processed_ids]
        
        # Step 3: Crawl articles (async for speed)
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: Crawling articles...")
        self.logger.info("=" * 60)
        
        # Batch crawling
        batch_size = 100
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i + batch_size]
            self.logger.info(f"Crawling batch {i//batch_size + 1}/{(len(entries) + batch_size - 1)//batch_size}")
            
            # Async crawl
            batch_data = await self.crawler.crawl_all_articles_async(batch)
            
            # Update all_data
            all_data.update(batch_data)
            
            # Check memory
            if not self.check_memory():
                self.save_checkpoint(all_data, start_index + i + len(batch))
                return
            
            # Save checkpoint
            if (i + batch_size) % 200 == 0:
                self.save_checkpoint(all_data, start_index + i + batch_size)
        
        # Step 4: Extract entities and relationships
        if self.kg_builder:
            self.logger.info("=" * 60)
            self.logger.info("STEP 3: Extracting entities and relationships...")
            self.logger.info("=" * 60)
            
            # Process in parallel
            articles_to_process = {
                aid: adata for aid, adata in all_data.items()
                if 'entities' not in adata
            }
            
            if articles_to_process:
                self.logger.info(f"Processing {len(articles_to_process)} articles for entity extraction")
                
                # Use process pool for CPU-intensive NLP
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit tasks in batches
                    batch_size = 50
                    
                    for i in range(0, len(articles_to_process), batch_size):
                        batch_ids = list(articles_to_process.keys())[i:i + batch_size]
                        batch_articles = {aid: articles_to_process[aid] for aid in batch_ids}
                        
                        self.logger.info(f"Processing entity batch {i//batch_size + 1}")
                        
                        # Process batch
                        futures = []
                        for article_id, article_data in batch_articles.items():
                            future = executor.submit(
                                self._process_article_entities,
                                article_id,
                                article_data
                            )
                            futures.append((article_id, future))
                        
                        # Collect results
                        for article_id, future in futures:
                            try:
                                entities, relationships = future.result(timeout=300)
                                all_data[article_id]['entities'] = entities
                                all_data[article_id]['relationships'] = relationships
                            except Exception as e:
                                self.logger.error(f"Failed to process {article_id}: {e}")
                        
                        # Save checkpoint
                        if (i + batch_size) % 100 == 0:
                            self.save_checkpoint(all_data, start_index + len(entries))
                        
                        # Check memory
                        if not self.check_memory():
                            self.save_checkpoint(all_data, start_index + len(entries))
                            return
        
        # Step 5: Build the graph
        if self.graph:
            self.logger.info("=" * 60)
            self.logger.info("STEP 4: Building knowledge graph...")
            self.logger.info("=" * 60)
            
            self.graph.build_graph_from_extracted_data(all_data)
            
            # Export graph
            self.logger.info("Exporting graph...")
            self.graph.export_to_formats()
        
        # Save final data
        final_file = Path("data") / f"sep_complete_data_{datetime.now():%Y%m%d}.json"
        final_file.parent.mkdir(exist_ok=True)
        
        with open(final_file, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        # Report completion
        elapsed_time = time.time() - start_time
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE COMPLETE!")
        self.logger.info(f"Total time: {elapsed_time/3600:.2f} hours")
        self.logger.info(f"Articles processed: {len(all_data)}")
        self.logger.info(f"Data saved to: {final_file}")
        self.logger.info("=" * 60)
        
        return all_data
    
    def _process_article_entities(self, article_id: str, article_data: dict) -> tuple:
        """Process a single article for entities and relationships"""
        # Create a new KG builder instance for each process
        kg_builder = PhilosophyKGBuilder(
            use_gpu=self.use_gpu,
            use_ollama=False  # Disable Ollama in subprocesses
        )
        
        entities = kg_builder.extract_entities(
            article_data.get('full_text', ''),
            article_data.get('metadata', {})
        )
        
        relationships = kg_builder.extract_relationships(article_data)
        
        return entities, relationships
    
    def run_analysis_pipeline(self):
        """Run analysis on existing graph"""
        if not self.graph:
            self.logger.error("No graph available for analysis")
            return
        
        self.logger.info("Running graph analysis...")
        
        # Analyze philosophical schools and their relationships
        schools_query = """
        MATCH (s:School)-[r]-(p:Philosopher)
        RETURN s.name as school, collect(distinct p.name) as philosophers, count(p) as count
        ORDER BY count DESC
        """
        
        # Find most influential philosophers
        influence_query = """
        MATCH (p:Philosopher)-[r:INFLUENCES|INFLUENCED_BY]-(other)
        RETURN p.name as philosopher, count(r) as influence_score
        ORDER BY influence_score DESC
        LIMIT 20
        """
        
        # Concept networks
        concept_query = """
        MATCH (c1:Concept)-[r]-(c2:Concept)
        WHERE c1.name < c2.name
        RETURN c1.name as concept1, c2.name as concept2, count(r) as strength
        ORDER BY strength DESC
        LIMIT 50
        """
        
        # Run analyses
        if self.graph.use_neo4j:
            results = {
                'schools': self.graph.graph.run(schools_query).data(),
                'influential_philosophers': self.graph.graph.run(influence_query).data(),
                'concept_networks': self.graph.graph.run(concept_query).data()
            }
            
            # Save results
            with open('sep_analysis_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info("Analysis complete - results saved to sep_analysis_results.json")

def main():
    """Main entry point"""
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Build SEP Knowledge Graph')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh, ignore checkpoints')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
    parser.add_argument('--no-ollama', action='store_true', help='Disable Ollama LLM usage')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--analyze-only', action='store_true', help='Run analysis on existing graph')
    
    args = parser.parse_args()
    
    # Create builder
    builder = SEPGraphBuilder(
        resume_from_checkpoint=not args.no_resume,
        use_gpu=not args.no_gpu,
        use_ollama=not args.no_ollama,
        max_workers=args.workers
    )
    
    if args.analyze_only:
        builder.run_analysis_pipeline()
    else:
        # Run async pipeline
        asyncio.run(builder.run_full_pipeline())

if __name__ == "__main__":
    main()