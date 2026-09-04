# Step 1: Initial Data Extraction
# sep_crawler.py

import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin, urlparse
import time
import asyncio
import aiohttp
from pathlib import Path
import logging
from datetime import datetime

class SEPCrawler:
    def __init__(self, cache_dir="cache", max_concurrent=20):
    	self.headers = {
    		'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
	}
        self.base_url = "https://plato.stanford.edu"
        self.entries_url = f"{self.base_url}/entries/"
        self.visited = set()
        self.articles = {}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_concurrent = max_concurrent  # Utilize your bandwidth
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_all_entries(self):
        """Get list of all SEP entries from index"""
        cache_file = self.cache_dir / "sep_entries.json"
        
        # Check cache first
        if cache_file.exists():
            self.logger.info("Loading entries from cache")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        self.logger.info("Fetching SEP index...")
        index_url = f"{self.base_url}/contents.html"
        response = requests.get(index_url, headers=self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        entries = []
        # Find all entry links
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('entries/') and 1 <= href.count('/') <= 3 and '..' not in href:
                entry_id = href.split('/')[-1].rstrip('/')
                if entry_id and not entry_id.startswith('#'):
                    entries.append({
                        'id': entry_id,
                        'title': link.text.strip(),
                        'url': urljoin(self.base_url, href)
                    })
        
        # Also get entries from the main entries page
        entries_page = requests.get(self.entries_url, headers=self.headers)
        soup = BeautifulSoup(entries_page.text, 'html.parser')
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if 'entries/' in href and not href.startswith('http'):
                entry_id = href.split('/')[-1].rstrip('/')
                if entry_id and not any(e['id'] == entry_id for e in entries):
                    entries.append({
                        'id': entry_id,
                        'title': link.text.strip() or entry_id.replace('-', ' ').title(),
                        'url': urljoin(self.base_url, href)
                    })
        
        # Remove duplicates
        seen = set()
        unique_entries = []
        for entry in entries:
            if entry['id'] not in seen:
                seen.add(entry['id'])
                unique_entries.append(entry)
        
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(unique_entries, f, indent=2)
        
        self.logger.info(f"Found {len(unique_entries)} unique entries")
        return unique_entries
    
    async def fetch_article_async(self, session, url):
        """Async fetch a single article"""
        try:
            async with session.get(url, timeout=30, headers=self.headers) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    self.logger.warning(f"Failed to fetch {url}: {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {e}")
            return None
    
    def extract_article_data(self, url, html_content=None):
        """Extract structured data from a single article"""
        try:
            if html_content is None:
                response = requests.get(url, timeout=30, headers=self.headers)
                html_content = response.text
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract metadata
            metadata = {
                'title': '',
                'authors': [],
                'pubdate': None,
                'url': url,
                'last_updated': None
            }
            
            # Get title
            h1 = soup.find('h1')
            if h1:
                metadata['title'] = h1.text.strip()
            
            # Get authors
            authors_div = soup.find('div', id='aueditable')
            if authors_div:
                for author in authors_div.find_all('a'):
                    metadata['authors'].append(author.text.strip())
            
            # Get publication info
            pubinfo = soup.find('div', id='pubinfo')
            if pubinfo:
                # Look for dates
                import re
                date_pattern = r'First published \w+ \d{1,2}, \d{4}'
                date_match = re.search(date_pattern, pubinfo.text)
                if date_match:
                    metadata['pubdate'] = date_match.group()
                
                # Look for last update
                update_pattern = r'last revised \w+ \d{1,2}, \d{4}'
                update_match = re.search(update_pattern, pubinfo.text)
                if update_match:
                    metadata['last_updated'] = update_match.group()
            
            # Extract internal links and their context
            internal_links = []
            main_content = soup.find('div', id='main-text')
            
            if main_content:
                for link in main_content.find_all('a', href=True):
                    href = link['href']
                    if href.startswith('/entries/'):
                        # Get surrounding context (parent paragraph)
                        parent = link.parent
                        while parent and parent.name not in ['p', 'li', 'div']:
                            parent = parent.parent
                        
                        context = parent.text if parent else ''
                        context = ' '.join(context.split())[:500]  # Clean and limit
                        
                        internal_links.append({
                            'target': href.split('/')[-1].rstrip('/'),
                            'anchor_text': link.text.strip(),
                            'context': context,
                            'section': self._find_section_heading(link)
                        })
            
            # Extract section headings for topic modeling
            sections = []
            toc = soup.find('div', id='toc')
            if toc:
                for link in toc.find_all('a'):
                    section_text = link.text.strip()
                    if section_text and not section_text.startswith('Bibliography'):
                        sections.append(section_text)
            
            # Extract bibliography entries
            bibliography = []
            bib_section = soup.find('div', id='bibliography')
            if bib_section:
                for li in bib_section.find_all('li'):
                    bib_text = li.text.strip()
                    if bib_text:
                        bibliography.append(bib_text)
            
            # Extract full text
            full_text = ''
            if main_content:
                # Remove script and style elements
                for script in main_content(['script', 'style']):
                    script.decompose()
                full_text = main_content.get_text(separator=' ', strip=True)
            
            # Extract academic references (citations to other works)
            citations = self._extract_citations(full_text)
            
            return {
                'metadata': metadata,
                'internal_links': internal_links,
                'sections': sections,
                'bibliography': bibliography,
                'full_text': full_text,
                'citations': citations,
                'word_count': len(full_text.split())
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting data from {url}: {e}")
            return None
    
    def _find_section_heading(self, element):
        """Find the section heading for an element"""
        current = element
        while current:
            # Look for previous sibling headings
            for sibling in current.find_previous_siblings():
                if sibling.name in ['h1', 'h2', 'h3', 'h4']:
                    return sibling.text.strip()
            current = current.parent
        return "Introduction"
    
    def _extract_citations(self, text):
        """Extract academic citations from text"""
        import re
        citations = []
        
        # Pattern for (Author Year) citations
        pattern = r'\(([A-Z][a-zA-Z\s&,]+)\s+(\d{4}[a-z]?)\)'
        matches = re.findall(pattern, text)
        
        for match in matches:
            citations.append({
                'authors': match[0].strip(),
                'year': match[1]
            })
        
        return citations
    
    async def crawl_all_articles_async(self, entries):
        """Crawl all articles using async/await for speed"""
        self.logger.info(f"Starting async crawl of {len(entries)} articles")
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            results = {}
            
            # Process in batches to avoid overwhelming the server
            batch_size = self.max_concurrent
            for i in range(0, len(entries), batch_size):
                batch = entries[i:i + batch_size]
                self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(entries) + batch_size - 1)//batch_size}")
                
                tasks = []
                for entry in batch:
                    task = self.fetch_article_async(session, entry['url'])
                    tasks.append((entry, task))
                
                # Gather results
                for entry, task in tasks:
                    html_content = await task
                    if html_content:
                        article_data = self.extract_article_data(entry['url'], html_content)
                        if article_data:
                            results[entry['id']] = article_data
                
                # Small delay between batches
                await asyncio.sleep(0.5)
        
        return results
