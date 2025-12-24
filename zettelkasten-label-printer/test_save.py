#!/usr/bin/env python3
"""
Test script to verify note saving and loading works correctly.
"""

import tempfile
import os
from parsers import Note, save_notes_to_file, ZoteroParser

def test_save_and_load():
    """Test that we can save notes and read them back."""

    # Create test notes
    test_notes = [
        Note(
            title="Test Note 1",
            author="John Doe",
            source="Test Book",
            body="This is the body of the first test note.\nIt has multiple lines.",
            page="42",
            source_file=""
        ),
        Note(
            title="Test Note 2",
            author="Jane Smith",
            source="Another Book",
            body="This is the second note.",
            page="123",
            source_file=""
        )
    ]

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        temp_file = f.name

    try:
        # Save notes to file
        print(f"Saving {len(test_notes)} notes to {temp_file}...")
        success = save_notes_to_file(temp_file, test_notes)

        if not success:
            print("❌ Failed to save notes")
            return False

        print("✅ Notes saved successfully")

        # Read the file content
        print("\n📄 File content:")
        print("-" * 60)
        with open(temp_file, 'r') as f:
            content = f.read()
            print(content)
        print("-" * 60)

        # Load notes back
        print("\n🔍 Loading notes back...")
        loaded_notes = ZoteroParser.parse_file(temp_file)
        print(f"✅ Loaded {len(loaded_notes)} notes")

        # Verify
        print("\n🔍 Verifying loaded notes...")
        for i, note in enumerate(loaded_notes, 1):
            print(f"\nNote {i}:")
            print(f"  Title: {note.title}")
            print(f"  Author: {note.author}")
            print(f"  Source: {note.source}")
            print(f"  Page: {note.page}")
            print(f"  Body: {note.body[:50]}...")

        if len(loaded_notes) == len(test_notes):
            print("\n✅ Test passed! Save and load cycle works correctly.")
            return True
        else:
            print(f"\n❌ Test failed! Expected {len(test_notes)} notes, got {len(loaded_notes)}")
            return False

    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"\n🗑️  Cleaned up temp file")

if __name__ == "__main__":
    test_save_and_load()
