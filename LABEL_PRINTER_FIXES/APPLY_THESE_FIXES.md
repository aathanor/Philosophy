# Manual Fixes to Apply

Since the git sync isn't working properly, apply these changes manually on your Mac:

## 1. Fix config.yaml (Line 7)
Change PID from 0x209b to 0x209c:
```yaml
connection: "usb://0x04f9:0x209c"
```

## 2. Replace HighlightedParser class in parsers.py

Replace the entire `HighlightedParser` class (starting around line 232) with the version from `UPDATED_HighlightedParser.py` in this directory.

The key changes:
- Now recognizes `**Author:**` metadata
- Now recognizes `**Source:**` metadata
- Now recognizes `**Page:**` metadata
- Uses H2 (`##`) as note title

## 3. Fix save_notes_to_file() in parsers.py (Line 499)

The function needs to handle both .txt (Scapple) and .md (Zotero/Highlighted) formats.
See `UPDATED_save_notes_to_file.py` for the complete function.

## 4. Fix save_note_to_file() in app.py (Line 378)

Change line 390 from:
```python
notes_from_file = [n for n in st.session_state.notes if n.source_file == note.source_file]
```

To:
```python
all_notes = st.session_state.notes + st.session_state.scapple_notes
notes_from_file = [n for n in all_notes if n.source_file == note.source_file]
```

This ensures Scapple notes can be saved too.

## Quick Apply Script

Or just copy these three files from this directory:
1. parsers.py
2. app.py
3. config.yaml

All have the fixes applied.
