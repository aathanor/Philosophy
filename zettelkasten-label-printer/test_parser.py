#!/usr/bin/env python3
"""
Debug script to test parser on actual Zotero exports
"""

import sys
from pathlib import Path
from parsers import ZoteroParser, HighlightedParser

def test_zotero_file(filepath):
    """Test parsing a Zotero markdown file."""
    print(f"\n{'='*60}")
    print(f"Testing file: {filepath}")
    print(f"{'='*60}\n")

    if not Path(filepath).exists():
        print(f"❌ File not found: {filepath}")
        return

    # Read raw content
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    print("📄 Raw file content:")
    print("-" * 60)
    print(content[:500])  # First 500 chars
    if len(content) > 500:
        print(f"\n... ({len(content) - 500} more characters)")
    print("-" * 60)

    # Try parsing
    print("\n🔍 Parsing notes...")
    try:
        notes = ZoteroParser.parse_file(filepath)
        print(f"\n✅ Found {len(notes)} notes\n")

        for i, note in enumerate(notes, 1):
            print(f"Note {i}:")
            print(f"  Title: {note.title[:100]}")
            print(f"  Author: {note.author}")
            print(f"  Source: {note.source}")
            print(f"  Body: {note.body[:100]}...")
            print(f"  Page: {note.page}")
            print()
    except Exception as e:
        print(f"❌ Error parsing: {e}")
        import traceback
        traceback.print_exc()

def test_folder(folder_path):
    """Test parsing all files in a folder."""
    folder = Path(folder_path).expanduser()

    if not folder.exists():
        print(f"❌ Folder not found: {folder}")
        return

    md_files = list(folder.glob('*.md'))
    print(f"\n📂 Found {len(md_files)} markdown files in {folder}\n")

    for md_file in md_files:
        test_zotero_file(str(md_file))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Test single file: python test_parser.py <filepath>")
        print("  Test folder: python test_parser.py <folder_path>")
        sys.exit(1)

    path = sys.argv[1]
    path_obj = Path(path).expanduser()

    if path_obj.is_file():
        test_zotero_file(path)
    elif path_obj.is_dir():
        test_folder(path)
    else:
        print(f"❌ Path not found: {path}")
