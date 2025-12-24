"""
Print History Tracker
Tracks which notes have been printed to show new/unprinted notes first.
"""

import json
from pathlib import Path
from typing import Set, Dict
from datetime import datetime


class PrintHistory:
    """Manages history of printed notes."""

    def __init__(self, history_file: str = "print_history.json"):
        """
        Initialize print history tracker.

        Args:
            history_file: Path to JSON file storing print history
        """
        self.history_file = Path(history_file)
        self.history = self._load_history()

    def _load_history(self) -> Dict:
        """Load print history from file."""
        if not self.history_file.exists():
            return {"printed": {}, "version": "1.0"}

        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {"printed": {}, "version": "1.0"}

    def _save_history(self):
        """Save print history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Error saving print history: {e}")

    def _note_id(self, note) -> str:
        """Generate unique ID for a note."""
        # Use combination of title, author, and first 50 chars of body
        return f"{note.title[:50]}|{note.author}|{note.body[:50]}"

    def mark_printed(self, note) -> None:
        """
        Mark a note as printed.

        Args:
            note: Note object that was printed
        """
        note_id = self._note_id(note)
        self.history["printed"][note_id] = {
            "timestamp": datetime.now().isoformat(),
            "title": note.title[:100]
        }
        self._save_history()

    def is_printed(self, note) -> bool:
        """
        Check if a note has been printed.

        Args:
            note: Note object to check

        Returns:
            True if note has been printed before
        """
        note_id = self._note_id(note)
        return note_id in self.history["printed"]

    def get_print_time(self, note) -> str:
        """
        Get when a note was printed.

        Args:
            note: Note object

        Returns:
            ISO timestamp string or None
        """
        note_id = self._note_id(note)
        if note_id in self.history["printed"]:
            return self.history["printed"][note_id].get("timestamp")
        return None

    def clear_history(self):
        """Clear all print history."""
        self.history = {"printed": {}, "version": "1.0"}
        self._save_history()

    def get_stats(self) -> Dict:
        """Get statistics about print history."""
        return {
            "total_printed": len(self.history["printed"]),
        }
