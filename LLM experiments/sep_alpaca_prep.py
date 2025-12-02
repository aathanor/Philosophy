#!/usr/bin/env python3
"""
SEP to Alpaca LLM Preparation Script

Downloads Stanford Encyclopedia of Philosophy articles,
cleans and organizes them into a folder structure suitable
for Alpaca LLM fine-tuning on Linux.

Usage:
    python sep_alpaca_prep.py --max-articles 100 --output-dir ./sep_data
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Optional
from html2text import HTML2Text
from tqdm import tqdm
import hashlib

class SEPAlpacaPrep:
    """Downloads and prepares SEP content for Alpaca LLM training"""

    def __init__(self, output_dir: str = "./sep_data", delay: float = 1.0):
        self.base_url = "https://plato.stanford.edu"
        self.output_dir = Path(output_dir)
        self.delay = delay  # Respectful crawling delay

        # Create directory structure
        self.setup_directories()

        # Setup logging
        self.setup_logging()

        # HTML to Markdown converter
        self.html2text = HTML2Text()
        self.html2text.ignore_links = False
        self.html2text.ignore_images = True
        self.html2text.ignore_emphasis = False
        self.html2text.body_width = 0  # Don't wrap lines

        # Categories for organization
        self.categories = self._load_categories()

        # Request headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def setup_directories(self):
        """Create output directory structure"""
        directories = [
            self.output_dir / "raw_html",
            self.output_dir / "markdown",
            self.output_dir / "plain_text",
            self.output_dir / "alpaca_datasets",
            self.output_dir / "metadata",
            self.output_dir / "by_topic",
            self.output_dir / "logs"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "logs" / f"sep_prep_{datetime.now():%Y%m%d_%H%M%S}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging to {log_file}")

    def _load_categories(self) -> Dict[str, List[str]]:
        """Define topic categories for organization"""
        return {
            "metaphysics": ["metaphysics", "ontology", "reality", "existence", "substance",
                          "identity", "causation", "time", "space", "modality"],
            "epistemology": ["knowledge", "epistemology", "belief", "justification", "truth",
                           "skepticism", "perception", "evidence", "reasoning"],
            "ethics": ["ethics", "morality", "virtue", "duty", "consequentialism", "deontology",
                      "utilitarianism", "moral", "right", "wrong", "good", "evil"],
            "political_philosophy": ["political", "justice", "rights", "liberty", "democracy",
                                   "social", "law", "authority", "power", "state"],
            "logic": ["logic", "formal", "reasoning", "argument", "inference", "proof",
                     "paradox", "fallacy", "validity", "soundness"],
            "philosophy_of_mind": ["mind", "consciousness", "intentionality", "mental", "cognitive",
                                  "perception", "qualia", "dualism", "materialism"],
            "philosophy_of_science": ["science", "scientific", "explanation", "theory", "hypothesis",
                                     "causation", "laws", "reduction", "emergence"],
            "philosophy_of_language": ["language", "meaning", "reference", "semantics", "pragmatics",
                                      "syntax", "communication", "translation"],
            "aesthetics": ["aesthetics", "beauty", "art", "taste", "aesthetic", "artistic"],
            "ancient_philosophy": ["plato", "aristotle", "socrates", "stoic", "epicurean",
                                 "presocratic", "hellenistic", "ancient"],
            "medieval_philosophy": ["medieval", "aquinas", "augustine", "scholastic", "ockham",
                                  "anselm", "boethius"],
            "modern_philosophy": ["descartes", "locke", "hume", "kant", "leibniz", "spinoza",
                                "berkeley", "empiricism", "rationalism"],
            "contemporary_philosophy": ["analytic", "phenomenology", "existentialism", "pragmatism",
                                      "postmodern", "contemporary", "twentieth", "21st"]
        }

    def get_all_entries(self) -> List[Dict]:
        """Fetch list of all SEP entries"""
        cache_file = self.output_dir / "metadata" / "sep_entries_list.json"

        # Check cache
        if cache_file.exists():
            self.logger.info("Loading entries from cache")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        self.logger.info("Fetching SEP table of contents...")
        entries = []

        # Create a session for better connection handling
        session = requests.Session()
        session.headers.update(self.headers)

        try:
            # Try multiple pages to get entries
            urls_to_try = [
                f"{self.base_url}/contents.html",
                f"{self.base_url}/index.html",
            ]

            # Also try alphabetical index pages
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                urls_to_try.append(f"{self.base_url}/contents.html#{letter}")

            self.logger.info("Attempting to fetch entry list...")

            for url in urls_to_try[:3]:  # Try first 3 URLs
                try:
                    self.logger.info(f"Trying {url}...")
                    response = session.get(url, timeout=30)

                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')

                        # Look for entry links in various formats
                        for link in soup.find_all('a', href=True):
                            href = link['href']

                            # Match patterns like: entries/epistemology/ or entries/kant/
                            if 'entries/' in href:
                                # Clean up the href
                                if href.startswith('/'):
                                    href = href[1:]
                                elif href.startswith('http'):
                                    # Extract path from full URL
                                    parsed = urlparse(href)
                                    href = parsed.path.lstrip('/')

                                # Extract entry ID
                                if href.startswith('entries/'):
                                    parts = href.split('/')
                                    if len(parts) >= 2:
                                        entry_id = parts[1]

                                        # Skip if already added or invalid
                                        if entry_id and not any(e['id'] == entry_id for e in entries):
                                            # Filter out navigation items
                                            if entry_id not in ['index', 'contents', 'new', 'archives']:
                                                title = link.text.strip() or entry_id.replace('-', ' ').title()
                                                entries.append({
                                                    'id': entry_id,
                                                    'title': title,
                                                    'url': f"{self.base_url}/entries/{entry_id}/"
                                                })

                        if entries:
                            break  # Found entries, stop trying

                    time.sleep(1)  # Respectful delay between tries

                except Exception as e:
                    self.logger.warning(f"Failed to fetch from {url}: {e}")
                    continue

            # If still no entries, use a predefined list of common entries
            if not entries:
                self.logger.warning("Could not fetch from SEP website, using predefined entry list...")
                entries = self._get_fallback_entries()

            self.logger.info(f"Found {len(entries)} entries")

            # Save to cache
            if entries:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(entries, f, indent=2)

            return entries

        except Exception as e:
            self.logger.error(f"Error fetching entries: {e}")
            # Return fallback list
            return self._get_fallback_entries()

    def _get_fallback_entries(self) -> List[Dict]:
        """Fallback list of major SEP entries when scraping fails"""
        self.logger.info("Using fallback entry list of major philosophy topics...")

        fallback_ids = [
            'epistemology', 'metaphysics', 'ethics', 'logic-classical',
            'plato', 'aristotle', 'kant', 'hume', 'descartes', 'locke',
            'nietzsche', 'wittgenstein', 'heidegger', 'quine', 'rawls',
            'consciousness', 'free-will', 'personal-identity', 'mind',
            'philosophy-science', 'causation', 'time', 'space',
            'truth', 'knowledge-analysis', 'skepticism', 'perception',
            'moral-realism', 'consequentialism', 'deontological-ethics',
            'virtue-ethics', 'justice', 'rights', 'political-obligation',
            'social-contract', 'liberalism', 'democracy',
            'aesthetic-judgment', 'art-definition', 'beauty',
            'existence', 'ontology', 'universals', 'tropes',
            'language-thought', 'meaning', 'reference', 'pragmatics',
            'logic-modal', 'logic-inductive', 'rationality',
            'scientific-method', 'laws-of-nature', 'reductionism',
            'quantum-mechanics', 'action', 'emotion', 'self-knowledge',
        ]

        entries = []
        for entry_id in fallback_ids:
            entries.append({
                'id': entry_id,
                'title': entry_id.replace('-', ' ').title(),
                'url': f"{self.base_url}/entries/{entry_id}/"
            })

        return entries

    def download_article(self, entry: Dict) -> Optional[Dict]:
        """Download and parse a single article"""
        try:
            self.logger.info(f"Downloading: {entry['title']}")

            response = requests.get(entry['url'], headers=self.headers, timeout=30)
            response.raise_for_status()

            # Save raw HTML
            html_file = self.output_dir / "raw_html" / f"{entry['id']}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(response.text)

            # Parse content
            article_data = self.parse_article(entry, response.text)

            # Respectful delay
            time.sleep(self.delay)

            return article_data

        except Exception as e:
            self.logger.error(f"Error downloading {entry['id']}: {e}")
            return None

    def parse_article(self, entry: Dict, html_content: str) -> Dict:
        """Parse article HTML and extract structured content"""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract metadata
        metadata = {
            'id': entry['id'],
            'title': entry['title'],
            'url': entry['url'],
            'authors': [],
            'published': None,
            'last_updated': None,
            'topics': []
        }

        # Get title (more accurate from page)
        h1 = soup.find('h1')
        if h1:
            metadata['title'] = h1.text.strip()

        # Get authors
        author_div = soup.find('div', id='aueditable')
        if author_div:
            for author_link in author_div.find_all('a'):
                metadata['authors'].append(author_link.text.strip())

        # Get publication info
        pubinfo = soup.find('div', id='pubinfo')
        if pubinfo:
            pubinfo_text = pubinfo.text
            # Extract dates
            first_pub = re.search(r'First published\s+(\w+\s+\d{1,2},\s+\d{4})', pubinfo_text)
            if first_pub:
                metadata['published'] = first_pub.group(1)

            last_rev = re.search(r'substantive revision\s+(\w+\s+\d{1,2},\s+\d{4})', pubinfo_text)
            if last_rev:
                metadata['last_updated'] = last_rev.group(1)

        # Extract main content
        main_text = soup.find('div', id='main-text')
        if not main_text:
            main_text = soup.find('div', id='article')

        if main_text:
            # Remove navigation, scripts, styles
            for element in main_text.find_all(['script', 'style', 'nav', 'noscript']):
                element.decompose()

            # Convert to markdown
            markdown_content = self.html2text.handle(str(main_text))

            # Get plain text
            plain_text = main_text.get_text(separator='\n', strip=True)

            # Extract sections
            sections = self._extract_sections(soup)

            # Extract bibliography
            bibliography = self._extract_bibliography(soup)

            # Categorize by topic
            metadata['topics'] = self._categorize_article(metadata['title'], plain_text)

            return {
                'metadata': metadata,
                'markdown': markdown_content,
                'plain_text': plain_text,
                'sections': sections,
                'bibliography': bibliography,
                'word_count': len(plain_text.split())
            }

        return {'metadata': metadata, 'markdown': '', 'plain_text': '',
                'sections': [], 'bibliography': [], 'word_count': 0}

    def _extract_sections(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract section headings and content"""
        sections = []
        main_text = soup.find('div', id='main-text') or soup.find('div', id='article')

        if main_text:
            for heading in main_text.find_all(['h2', 'h3', 'h4']):
                section_id = heading.get('id', '')
                section_title = heading.text.strip()

                # Get section content (next siblings until next heading)
                content = []
                for sibling in heading.find_next_siblings():
                    if sibling.name in ['h2', 'h3', 'h4']:
                        break
                    if sibling.name in ['p', 'ul', 'ol', 'blockquote']:
                        content.append(sibling.get_text(strip=True))

                sections.append({
                    'id': section_id,
                    'title': section_title,
                    'level': int(heading.name[1]),  # h2 -> 2, h3 -> 3
                    'content': '\n\n'.join(content)
                })

        return sections

    def _extract_bibliography(self, soup: BeautifulSoup) -> List[str]:
        """Extract bibliography entries"""
        bibliography = []
        bib_div = soup.find('div', id='bibliography')

        if bib_div:
            for item in bib_div.find_all('li'):
                bib_text = item.get_text(strip=True)
                if bib_text:
                    bibliography.append(bib_text)

        return bibliography

    def _categorize_article(self, title: str, content: str) -> List[str]:
        """Categorize article by topics"""
        text_to_check = (title + ' ' + content[:2000]).lower()
        topics = []

        for category, keywords in self.categories.items():
            # Check if any keyword appears in title or early content
            if any(keyword in text_to_check for keyword in keywords):
                topics.append(category)

        # Default category if none found
        if not topics:
            topics.append('general_philosophy')

        return topics

    def save_article(self, article_data: Dict):
        """Save article in various formats"""
        article_id = article_data['metadata']['id']

        # Save markdown
        md_file = self.output_dir / "markdown" / f"{article_id}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"# {article_data['metadata']['title']}\n\n")
            f.write(f"**Authors:** {', '.join(article_data['metadata']['authors'])}\n\n")
            if article_data['metadata']['published']:
                f.write(f"**Published:** {article_data['metadata']['published']}\n\n")
            f.write(f"**Source:** [{article_data['metadata']['url']}]({article_data['metadata']['url']})\n\n")
            f.write("---\n\n")
            f.write(article_data['markdown'])

        # Save plain text
        txt_file = self.output_dir / "plain_text" / f"{article_id}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(article_data['plain_text'])

        # Save metadata
        meta_file = self.output_dir / "metadata" / f"{article_id}.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(article_data['metadata'], f, indent=2)

        # Organize by topic
        for topic in article_data['metadata']['topics']:
            topic_dir = self.output_dir / "by_topic" / topic
            topic_dir.mkdir(parents=True, exist_ok=True)

            # Create symlink or copy
            topic_file = topic_dir / f"{article_id}.md"
            if not topic_file.exists():
                with open(topic_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {article_data['metadata']['title']}\n\n")
                    f.write(article_data['markdown'])

    def generate_alpaca_dataset(self, articles: List[Dict], output_file: str = "alpaca_philosophy.json"):
        """Generate Alpaca-format dataset from articles"""
        alpaca_data = []

        self.logger.info("Generating Alpaca instruction dataset...")

        for article in tqdm(articles, desc="Processing articles"):
            if not article:
                continue

            metadata = article['metadata']
            sections = article['sections']

            # Generate instruction-response pairs

            # 1. General article summary
            if article['plain_text']:
                alpaca_data.append({
                    'instruction': f"What is the philosophical topic of {metadata['title']}?",
                    'input': "",
                    'output': article['plain_text'][:1000] + "...",
                    'source': metadata['url'],
                    'topics': metadata['topics']
                })

            # 2. Section-based Q&A
            for section in sections[:5]:  # Limit to first 5 sections
                if section['content']:
                    # Clean section title for question
                    section_q = section['title'].strip('0123456789. ')

                    alpaca_data.append({
                        'instruction': f"Explain the concept of '{section_q}' in the context of {metadata['title']}.",
                        'input': "",
                        'output': section['content'][:800],
                        'source': metadata['url'],
                        'topics': metadata['topics']
                    })

            # 3. Author-based questions
            if metadata['authors']:
                authors_str = ', '.join(metadata['authors'])
                alpaca_data.append({
                    'instruction': f"Who wrote about {metadata['title']} in the Stanford Encyclopedia of Philosophy?",
                    'input': "",
                    'output': f"The article on {metadata['title']} was written by {authors_str}.",
                    'source': metadata['url'],
                    'topics': metadata['topics']
                })

            # 4. Comparison questions (for similar topics)
            if len(sections) >= 2:
                sec1, sec2 = sections[0], sections[1]
                if sec1['content'] and sec2['content']:
                    alpaca_data.append({
                        'instruction': f"Compare and contrast '{sec1['title']}' and '{sec2['title']}' in {metadata['title']}.",
                        'input': "",
                        'output': f"Regarding {sec1['title']}: {sec1['content'][:300]}...\n\nRegarding {sec2['title']}: {sec2['content'][:300]}...",
                        'source': metadata['url'],
                        'topics': metadata['topics']
                    })

        # Save Alpaca dataset
        output_path = self.output_dir / "alpaca_datasets" / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(alpaca_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Generated {len(alpaca_data)} instruction-response pairs")
        self.logger.info(f"Saved to {output_path}")

        # Also save in formats for different frameworks
        self._save_training_formats(alpaca_data)

        return alpaca_data

    def _save_training_formats(self, alpaca_data: List[Dict]):
        """Save in multiple training formats"""

        # 1. JSONL format (one JSON per line)
        jsonl_file = self.output_dir / "alpaca_datasets" / "alpaca_philosophy.jsonl"
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for item in alpaca_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # 2. Hugging Face format
        hf_data = []
        for item in alpaca_data:
            hf_data.append({
                'text': f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
            })

        hf_file = self.output_dir / "alpaca_datasets" / "alpaca_philosophy_hf.json"
        with open(hf_file, 'w', encoding='utf-8') as f:
            json.dump(hf_data, f, indent=2, ensure_ascii=False)

        # 3. CSV format
        try:
            import csv
            csv_file = self.output_dir / "alpaca_datasets" / "alpaca_philosophy.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['instruction', 'input', 'output', 'source', 'topics']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for item in alpaca_data:
                    writer.writerow({
                        'instruction': item['instruction'],
                        'input': item.get('input', ''),
                        'output': item['output'],
                        'source': item['source'],
                        'topics': ','.join(item.get('topics', []))
                    })
        except Exception as e:
            self.logger.warning(f"Could not save CSV format: {e}")

        self.logger.info(f"Saved training data in multiple formats (JSON, JSONL, HF, CSV)")

    def run(self, max_articles: Optional[int] = None):
        """Main execution function"""
        self.logger.info("="*60)
        self.logger.info("SEP Alpaca Preparation Script")
        self.logger.info("="*60)

        # Get all entries
        entries = self.get_all_entries()

        if max_articles:
            entries = entries[:max_articles]
            self.logger.info(f"Limited to {max_articles} articles")

        # Download and process articles
        articles = []
        for entry in tqdm(entries, desc="Downloading articles"):
            article_data = self.download_article(entry)
            if article_data:
                self.save_article(article_data)
                articles.append(article_data)

        # Generate statistics
        self._generate_statistics(articles)

        # Generate Alpaca dataset
        self.generate_alpaca_dataset(articles)

        self.logger.info("="*60)
        self.logger.info("Processing complete!")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Articles processed: {len(articles)}")
        self.logger.info("="*60)

    def _generate_statistics(self, articles: List[Dict]):
        """Generate statistics about the downloaded corpus"""
        stats = {
            'total_articles': len(articles),
            'total_words': sum(a.get('word_count', 0) for a in articles),
            'topics_distribution': {},
            'authors': set(),
            'date_range': {'earliest': None, 'latest': None}
        }

        # Count topics
        for article in articles:
            for topic in article['metadata'].get('topics', []):
                stats['topics_distribution'][topic] = stats['topics_distribution'].get(topic, 0) + 1

            # Collect authors
            stats['authors'].update(article['metadata'].get('authors', []))

        stats['unique_authors'] = len(stats['authors'])
        stats['authors'] = list(stats['authors'])[:50]  # Save only first 50

        # Save statistics
        stats_file = self.output_dir / "metadata" / "corpus_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"Statistics saved to {stats_file}")
        self.logger.info(f"Total words: {stats['total_words']:,}")
        self.logger.info(f"Unique authors: {stats['unique_authors']}")


def main():
    parser = argparse.ArgumentParser(
        description='Download and prepare SEP content for Alpaca LLM training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download first 50 articles
  python sep_alpaca_prep.py --max-articles 50

  # Download all articles (takes several hours)
  python sep_alpaca_prep.py --output-dir ./sep_complete

  # Fast mode with shorter delays (use carefully)
  python sep_alpaca_prep.py --delay 0.5 --max-articles 100
        """
    )

    parser.add_argument('--output-dir', type=str, default='./sep_data',
                       help='Output directory for downloaded content (default: ./sep_data)')
    parser.add_argument('--max-articles', type=int, default=None,
                       help='Maximum number of articles to download (default: all)')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between requests in seconds (default: 1.0)')

    args = parser.parse_args()

    # Run the preparation
    prep = SEPAlpacaPrep(output_dir=args.output_dir, delay=args.delay)
    prep.run(max_articles=args.max_articles)


if __name__ == "__main__":
    main()
