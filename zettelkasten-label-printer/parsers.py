"""
Data Parsers Module
Parses notes from Zotero and Highlighted app exports.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import glob


class Note:
    """Represents a single note/quote for printing."""

    def __init__(self, title: str, author: str = "", source: str = "",
                 body: str = "", page: str = "", metadata: Dict = None):
        """
        Initialize a note.

        Args:
            title: Main title/heading of the note
            author: Author name
            source: Source publication/book title
            body: Main body text/quote
            page: Page number
            metadata: Additional metadata dictionary
        """
        self.title = title
        self.author = author
        self.source = source
        self.body = body
        self.page = page
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        """Convert note to dictionary for rendering."""
        return {
            'title': self.title,
            'author': self.author,
            'source': self.source,
            'body': self.body,
            'page': self.page
        }

    def __repr__(self) -> str:
        return f"Note(title='{self.title[:30]}...', author='{self.author}')"


class ZoteroParser:
    """Parser for Zotero markdown note exports."""

    @staticmethod
    def parse_file(filepath: str) -> List[Note]:
        """
        Parse a Zotero markdown export file.

        Zotero markdown notes typically have format:
        # Title (or item title)
        Author (Year). Publication.
        > Quote text (p. 123)

        Args:
            filepath: Path to markdown file

        Returns:
            List of Note objects
        """
        notes = []

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Try to extract metadata from the top
        lines = content.split('\n')
        current_note = {}
        buffer = []

        for line in lines:
            line = line.strip()

            # Skip HTML comments and tags
            if line.startswith('<!--') or line.startswith('<') or not line:
                continue

            # H1 heading - could be source title
            if line.startswith('# '):
                if current_note.get('body'):
                    # Save previous note
                    notes.append(Note(**current_note))
                    current_note = {}
                current_note['source'] = line[2:].strip()

            # H2 heading - could be section or note title
            elif line.startswith('## '):
                if current_note.get('body'):
                    notes.append(Note(**current_note))
                    current_note = {}
                current_note['title'] = line[3:].strip()

            # Extract author/citation info
            # Format: Author (YYYY). Title.
            elif re.match(r'^[A-Z][^(]+\(\d{4}\)', line):
                match = re.match(r'^([^(]+)\((\d{4})\)\.?\s*(.*)', line)
                if match:
                    current_note['author'] = match.group(1).strip()
                    # Could use year in metadata
                    if not current_note.get('source'):
                        current_note['source'] = match.group(3).strip()

            # Blockquote - the actual annotation/highlight
            elif line.startswith('>'):
                quote = line[1:].strip()

                # Extract page number if present: (p. 123) or (pp. 123-125)
                page_match = re.search(r'\(pp?\.\s*(\d+(?:-\d+)?)\)', quote)
                if page_match:
                    current_note['page'] = page_match.group(1)
                    # Remove page number from quote
                    quote = re.sub(r'\s*\(pp?\.\s*\d+(?:-\d+)?\)', '', quote)

                buffer.append(quote)

            # Regular text line
            elif line and not line.startswith('#'):
                # Could be continuation of a quote or citation
                buffer.append(line)

        # Flush remaining note
        if buffer:
            if not current_note.get('title'):
                # Use first part of body as title if no title found
                current_note['title'] = buffer[0][:50]
            current_note['body'] = ' '.join(buffer).strip()
            notes.append(Note(**current_note))

        return notes

    @staticmethod
    def scan_folder(folder_path: str) -> List[Note]:
        """
        Scan a folder for Zotero markdown exports and parse all notes.

        Args:
            folder_path: Path to folder containing markdown files

        Returns:
            List of all Note objects found
        """
        all_notes = []
        folder = Path(folder_path).expanduser()

        if not folder.exists():
            return all_notes

        for filepath in folder.glob('*.md'):
            try:
                notes = ZoteroParser.parse_file(str(filepath))
                all_notes.extend(notes)
            except Exception as e:
                print(f"Error parsing {filepath}: {e}")

        return all_notes


class HighlightedParser:
    """Parser for Highlighted app markdown exports."""

    @staticmethod
    def parse_file(filepath: str) -> List[Note]:
        """
        Parse a Highlighted app markdown export file.

        Highlighted markdown exports typically have format:
        # Document Title
        ## Author Name

        > Quote text
        Page: 123

        Args:
            filepath: Path to markdown file

        Returns:
            List of Note objects
        """
        notes = []

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')
        doc_title = ""
        current_author = ""
        current_note = {}
        quote_buffer = []

        for i, line in enumerate(lines):
            line = line.strip()

            # Skip empty lines and HTML
            if not line or line.startswith('<'):
                continue

            # H1 - Document title (source)
            if line.startswith('# '):
                doc_title = line[2:].strip()

            # H2 - Usually author or section
            elif line.startswith('## '):
                current_author = line[3:].strip()

            # Blockquote - the highlight/annotation
            elif line.startswith('>'):
                quote = line[1:].strip()
                quote_buffer.append(quote)

            # Page number (often separate line after quote)
            elif line.lower().startswith('page:') or line.lower().startswith('p.'):
                page_match = re.search(r'(\d+)', line)
                if page_match and quote_buffer:
                    # Create note from buffered quote
                    note = Note(
                        title=quote_buffer[0][:50],  # First line as title
                        author=current_author,
                        source=doc_title,
                        body=' '.join(quote_buffer),
                        page=page_match.group(1)
                    )
                    notes.append(note)
                    quote_buffer = []

            # Other text might be continuation
            elif quote_buffer and line and not line.startswith('#'):
                quote_buffer.append(line)

            # If we hit a new section and have buffered quotes, flush them
            elif quote_buffer and (line.startswith('#') or i == len(lines) - 1):
                note = Note(
                    title=quote_buffer[0][:50],
                    author=current_author,
                    source=doc_title,
                    body=' '.join(quote_buffer),
                    page=""
                )
                notes.append(note)
                quote_buffer = []

        # Flush any remaining quote
        if quote_buffer:
            note = Note(
                title=quote_buffer[0][:50],
                author=current_author,
                source=doc_title,
                body=' '.join(quote_buffer),
                page=""
            )
            notes.append(note)

        return notes

    @staticmethod
    def scan_folder(folder_path: str) -> List[Note]:
        """
        Scan a folder for Highlighted app markdown exports and parse all notes.

        Args:
            folder_path: Path to folder containing markdown files

        Returns:
            List of all Note objects found
        """
        all_notes = []
        folder = Path(folder_path).expanduser()

        if not folder.exists():
            return all_notes

        for filepath in folder.glob('*.md'):
            try:
                notes = HighlightedParser.parse_file(str(filepath))
                all_notes.extend(notes)
            except Exception as e:
                print(f"Error parsing {filepath}: {e}")

        return all_notes


def get_all_notes(zotero_folder: str, highlighted_folder: str) -> List[Note]:
    """
    Get all notes from both Zotero and Highlighted export folders.

    Args:
        zotero_folder: Path to Zotero exports folder
        highlighted_folder: Path to Highlighted exports folder

    Returns:
        Combined list of all Note objects
    """
    notes = []

    # Get Zotero notes
    zotero_notes = ZoteroParser.scan_folder(zotero_folder)
    notes.extend(zotero_notes)

    # Get Highlighted notes
    highlighted_notes = HighlightedParser.scan_folder(highlighted_folder)
    notes.extend(highlighted_notes)

    return notes
