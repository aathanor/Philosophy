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

        Zotero markdown notes can have various formats:
        - "Add Note from Annotations" format
        - Manual markdown notes
        - Exported highlights with metadata

        Args:
            filepath: Path to markdown file

        Returns:
            List of Note objects
        """
        notes = []

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split into sections by headers or empty lines
        lines = content.split('\n')
        current_note = {}
        quote_buffer = []
        text_buffer = []

        def save_current_note():
            """Helper to save the current note if it has content."""
            nonlocal current_note, quote_buffer, text_buffer

            # Combine buffers
            all_text = []
            if quote_buffer:
                all_text.extend(quote_buffer)
            if text_buffer:
                all_text.extend(text_buffer)

            if all_text:
                # Ensure we have at least a title
                if not current_note.get('title'):
                    # Use first line as title, rest as body
                    current_note['title'] = all_text[0][:100] if all_text else "Untitled"
                    current_note['body'] = ' '.join(all_text).strip()
                else:
                    current_note['body'] = ' '.join(all_text).strip()

                # Create note with defaults for missing fields
                notes.append(Note(
                    title=current_note.get('title', 'Untitled'),
                    author=current_note.get('author', ''),
                    source=current_note.get('source', ''),
                    body=current_note.get('body', ''),
                    page=current_note.get('page', '')
                ))

            # Reset buffers
            quote_buffer = []
            text_buffer = []

        for i, line in enumerate(lines):
            original_line = line
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip HTML comments and some tags
            if line.startswith('<!--') or line.startswith('<!'):
                continue

            # H1 heading - usually the document/source title
            if line.startswith('# '):
                save_current_note()
                current_note = {}
                title = line[2:].strip()
                # H1 could be source or title depending on context
                current_note['source'] = title

            # H2 heading - usually section or annotation title
            elif line.startswith('## '):
                save_current_note()
                current_note = {}
                current_note['title'] = line[3:].strip()

            # H3 and lower - treat as part of content
            elif line.startswith('#'):
                text_buffer.append(line.lstrip('#').strip())

            # Blockquote - the actual annotation/highlight
            elif line.startswith('>'):
                quote = line[1:].strip()

                # Extract page number if present: (p. 123) or (pp. 123-125)
                page_match = re.search(r'\(pp?\.\s*(\d+(?:-\d+)?)\)', quote)
                if page_match:
                    current_note['page'] = page_match.group(1)
                    # Remove page number from quote
                    quote = re.sub(r'\s*\(pp?\.\s*\d+(?:-\d+)?\)', '', quote)

                quote_buffer.append(quote)

            # Check for author/citation format: Author (YYYY). Title.
            elif re.match(r'^[A-Z][\w\s,.-]+\(\d{4}\)', line):
                match = re.match(r'^([^(]+)\((\d{4})\)\.?\s*(.*)', line)
                if match:
                    current_note['author'] = match.group(1).strip()
                    year = match.group(2)
                    rest = match.group(3).strip()
                    if rest and not current_note.get('source'):
                        current_note['source'] = rest

            # Page references on separate line
            elif re.match(r'^(?:p\.|pp\.|page:?)\s*\d+', line, re.IGNORECASE):
                page_match = re.search(r'(\d+(?:-\d+)?)', line)
                if page_match:
                    current_note['page'] = page_match.group(1)

            # Regular text line - add to buffer
            else:
                # Check if it looks like metadata
                if ':' in line and len(line.split(':')[0]) < 20:
                    # Might be metadata like "Author: Name" or "Page: 123"
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()

                    if 'author' in key:
                        current_note['author'] = value
                    elif 'page' in key:
                        page_match = re.search(r'(\d+(?:-\d+)?)', value)
                        if page_match:
                            current_note['page'] = page_match.group(1)
                    elif 'source' in key or 'title' in key:
                        if not current_note.get('source'):
                            current_note['source'] = value
                    else:
                        # Not recognized metadata, add as text
                        text_buffer.append(line)
                else:
                    text_buffer.append(line)

        # Save the final note
        save_current_note()

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
