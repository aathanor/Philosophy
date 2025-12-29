"""
Zettelkasten Label Printer - Main Streamlit Application
A web interface for printing zettelkasten cards on Brother QL-810W label printer.
"""

import streamlit as st
import logging
from pathlib import Path
import json

from config_loader import ConfigLoader
from parsers import get_all_notes, get_scapple_notes, Note, ZoteroParser, HighlightedParser, ScappleParser, save_notes_to_file
from label_renderer import LabelRenderer
from printer import create_printer, BROTHER_QL_AVAILABLE
from print_history import PrintHistory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="Zettelkasten Label Printer",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for more compact layout
st.markdown("""
<style>
    /* Reduce padding and margins - add enough top padding to clear Streamlit toolbar */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 0.5rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    /* Smaller headings with minimal spacing */
    h1 {
        font-size: 1.5rem !important;
        margin-top: 0.2rem !important;
        margin-bottom: 0.3rem !important;
    }

    h2 {
        font-size: 1.2rem !important;
        margin-top: 0.2rem !important;
        margin-bottom: 0.3rem !important;
    }

    h3 {
        font-size: 1rem !important;
        margin-top: 0.2rem !important;
        margin-bottom: 0.3rem !important;
    }

    /* Compact metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
    }

    /* Tighter button spacing */
    .stButton button {
        padding: 0.25rem 0.75rem !important;
        font-size: 0.9rem !important;
    }

    /* Reduce divider margins */
    hr {
        margin: 0.3rem 0 !important;
    }

    /* Compact sidebar */
    [data-testid="stSidebar"] {
        padding-top: 0.5rem;
    }

    /* Smaller expander headers */
    .streamlit-expanderHeader {
        font-size: 0.9rem !important;
    }

    /* Compact forms */
    .stTextInput label, .stTextArea label {
        font-size: 0.85rem !important;
        margin-bottom: 0.2rem !important;
    }

    .stTextInput input, .stTextArea textarea {
        font-size: 0.9rem !important;
        padding: 0.3rem 0.5rem !important;
    }

    /* Reduce spacing in columns */
    [data-testid="column"] {
        padding: 0 0.5rem !important;
    }

    /* Ensure images fit in viewport */
    img {
        max-height: 70vh !important;
        object-fit: contain !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'config' not in st.session_state:
        st.session_state.config = ConfigLoader('config.yaml')

    if 'notes' not in st.session_state:
        st.session_state.notes = []

    if 'scapple_notes' not in st.session_state:
        st.session_state.scapple_notes = []

    if 'notes_loaded' not in st.session_state:
        st.session_state.notes_loaded = False

    if 'current_note_index' not in st.session_state:
        st.session_state.current_note_index = None

    if 'current_scapple_note_index' not in st.session_state:
        st.session_state.current_scapple_note_index = None

    if 'printer' not in st.session_state:
        config = st.session_state.config.get_all()
        st.session_state.printer = create_printer(config, use_mock=not BROTHER_QL_AVAILABLE)

    if 'renderer' not in st.session_state:
        config = st.session_state.config.get_all()
        st.session_state.renderer = LabelRenderer(config)

    if 'print_history' not in st.session_state:
        st.session_state.print_history = PrintHistory()

    if 'edit_mode' not in st.session_state:
        st.session_state.edit_mode = False


def load_notes():
    """Load notes from configured folders."""
    config = st.session_state.config
    zotero_folder = config.get('data_sources.zotero_export_folder', '~/Documents/Zotero-Exports')
    highlighted_folder = config.get('data_sources.highlighted_export_folder', '~/Documents/Highlighted-Exports')
    scapple_folder = config.get('data_sources.scapple_export_folder', '~/Documents/Scapple-Exports')

    # Expand paths
    zotero_path = config.expand_path(zotero_folder)
    highlighted_path = config.expand_path(highlighted_folder)
    scapple_path = config.expand_path(scapple_folder)

    # Create folders if they don't exist
    zotero_path.mkdir(parents=True, exist_ok=True)
    highlighted_path.mkdir(parents=True, exist_ok=True)
    scapple_path.mkdir(parents=True, exist_ok=True)

    # Log folder info
    logger.info(f"Loading notes from:")
    logger.info(f"  Zotero: {zotero_path}")
    logger.info(f"  Highlighted: {highlighted_path}")
    logger.info(f"  Scapple: {scapple_path}")

    # Count files
    zotero_files = list(zotero_path.glob('*.md'))
    highlighted_files = list(highlighted_path.glob('*.md'))
    scapple_files = list(scapple_path.glob('*.txt'))
    logger.info(f"Found {len(zotero_files)} Zotero files, {len(highlighted_files)} Highlighted files, {len(scapple_files)} Scapple files")

    # Load notes (Zotero + Highlighted only, Scapple kept separate)
    notes = get_all_notes(str(zotero_path), str(highlighted_path))
    logger.info(f"Parsed {len(notes)} notes from Zotero/Highlighted")

    # Load Scapple notes separately
    scapple_notes = get_scapple_notes(str(scapple_path))
    logger.info(f"Parsed {len(scapple_notes)} Scapple notes")

    # Sort: unprinted first, then by title
    notes.sort(key=lambda n: (
        st.session_state.print_history.is_printed(n),  # False (unprinted) sorts first
        n.title.lower()
    ))

    scapple_notes.sort(key=lambda n: (
        st.session_state.print_history.is_printed(n),
        n.title.lower()
    ))

    st.session_state.notes = notes
    st.session_state.scapple_notes = scapple_notes
    st.session_state.notes_loaded = True
    return notes


def render_compact_sidebar():
    """Render compact sidebar with stats and controls."""
    with st.sidebar:
        st.markdown("### 🏷️ Label Printer")

        # Stats in columns for compactness
        col1, col2 = st.columns(2)
        with col1:
            total = len(st.session_state.notes) + len(st.session_state.scapple_notes)
            st.metric("Total", total)
        with col2:
            unprinted_regular = sum(1 for n in st.session_state.notes
                                   if not st.session_state.print_history.is_printed(n))
            unprinted_scapple = sum(1 for n in st.session_state.scapple_notes
                                   if not st.session_state.print_history.is_printed(n))
            st.metric("New", unprinted_regular + unprinted_scapple)

        # Reload button (prominent)
        if st.button("🔄 Reload Notes", type="primary", width='stretch'):
            with st.spinner("Loading notes..."):
                load_notes()
                st.rerun()

        st.divider()

        # Collapsible Settings
        with st.expander("⚙️ Printer Settings", expanded=False):
            if not BROTHER_QL_AVAILABLE:
                st.warning("⚠️ brother_ql not installed")
                st.caption("Using mock printer")
            else:
                printer_model = st.session_state.config.get('printer.model', 'QL-810W')
                st.success(f"✅ {printer_model}")

            if st.button("🔍 Test Connection", width='stretch'):
                if st.session_state.printer.test_connection():
                    st.success("✅ Connected!")
                else:
                    st.error("❌ Failed")

        with st.expander("📂 Data Sources", expanded=False):
            zotero_folder = st.text_input(
                "Zotero Folder",
                value=st.session_state.config.get('data_sources.zotero_export_folder', ''),
                key="zotero_input"
            )

            highlighted_folder = st.text_input(
                "Highlighted Folder",
                value=st.session_state.config.get('data_sources.highlighted_export_folder', ''),
                key="highlighted_input"
            )

            if st.button("💾 Save", width='stretch'):
                st.session_state.config.set('data_sources.zotero_export_folder', zotero_folder)
                st.session_state.config.set('data_sources.highlighted_export_folder', highlighted_folder)
                st.session_state.config.save()
                st.success("Saved!")

        with st.expander("🗑️ Clear History", expanded=False):
            st.caption("Clear print tracking")
            if st.button("Clear Print History", width='stretch'):
                st.session_state.print_history.clear_history()
                st.success("History cleared!")
                st.rerun()

        st.divider()

        # Shutdown button at bottom
        if st.button("🛑 Stop App", width='stretch', type="secondary"):
            st.warning("Shutting down...")
            import os
            import signal
            os.kill(os.getpid(), signal.SIGTERM)


def render_note_list():
    """Render note list in right column with tabs for Notes and Scapple."""

    # Create tabs
    tab1, tab2 = st.tabs(["📝 Notes", "🔷 Scapple"])

    # Tab 1: Regular Notes (Zotero + Highlighted)
    with tab1:
        # Search
        search = st.text_input("🔍", placeholder="Filter...", key="search_notes", label_visibility="collapsed")

        # Filter notes
        filtered_notes = st.session_state.notes
        if search:
            search_lower = search.lower()
            filtered_notes = [
                n for n in st.session_state.notes
                if search_lower in n.title.lower()
                or search_lower in n.author.lower()
                or search_lower in n.body.lower()
            ]

        st.caption(f"{len(filtered_notes)}/{len(st.session_state.notes)} notes")

        # Scrollable note list
        for i, note in enumerate(filtered_notes):
            is_printed = st.session_state.print_history.is_printed(note)

            # Find actual index in full list
            actual_index = st.session_state.notes.index(note)
            is_selected = st.session_state.current_note_index == actual_index

            # Status indicator
            if is_printed:
                style = "🔖"
            else:
                style = "🆕"

            # Button for each note - more compact
            button_label = f"{style} {note.title[:35]}..."

            if st.button(
                button_label,
                key=f"note_{actual_index}",
                width='stretch',
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.current_note_index = actual_index
                st.session_state.current_scapple_note_index = None  # Deselect Scapple
                st.session_state.edit_mode = False
                st.rerun()

    # Tab 2: Scapple Notes
    with tab2:
        # Search
        search_scapple = st.text_input("🔍", placeholder="Filter...", key="search_scapple", label_visibility="collapsed")

        # Filter Scapple notes
        filtered_scapple = st.session_state.scapple_notes
        if search_scapple:
            search_lower = search_scapple.lower()
            filtered_scapple = [
                n for n in st.session_state.scapple_notes
                if search_lower in n.title.lower()
                or search_lower in n.body.lower()
            ]

        st.caption(f"{len(filtered_scapple)}/{len(st.session_state.scapple_notes)} notes")

        # Scrollable Scapple note list
        for i, note in enumerate(filtered_scapple):
            is_printed = st.session_state.print_history.is_printed(note)

            # Find actual index in full list
            actual_index = st.session_state.scapple_notes.index(note)
            is_selected = st.session_state.current_scapple_note_index == actual_index

            # Status indicator
            if is_printed:
                style = "🔖"
            else:
                style = "🆕"

            # Button for each note
            button_label = f"{style} {note.title[:35]}..."

            if st.button(
                button_label,
                key=f"scapple_{actual_index}",
                width='stretch',
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.current_scapple_note_index = actual_index
                st.session_state.current_note_index = None  # Deselect regular notes
                st.session_state.edit_mode = False
                st.rerun()


def save_note_to_file(note: Note, note_data: dict):
    """Save edited note back to its source file."""
    # Update note object in memory
    note.title = note_data['title']
    note.author = note_data['author']
    note.source = note_data['source']
    note.body = note_data['body']
    note.page = note_data['page']

    # If note has a source file, save it back to disk
    if note.source_file:
        # Find all notes from the same source file (check both regular and Scapple notes)
        all_notes = st.session_state.notes + st.session_state.scapple_notes
        notes_from_file = [n for n in all_notes if n.source_file == note.source_file]

        # Save all notes from this file back to disk
        success = save_notes_to_file(note.source_file, notes_from_file)

        if success:
            logger.info(f"Saved {len(notes_from_file)} note(s) to {note.source_file}")
        else:
            logger.error(f"Failed to save notes to {note.source_file}")

        return success
    else:
        logger.warning(f"Note has no source_file, cannot save to disk")
        return False


def render_main_preview():
    """Render main preview and print area."""
    # Determine which note to show (regular or Scapple)
    note = None
    if st.session_state.current_note_index is not None:
        note = st.session_state.notes[st.session_state.current_note_index]
    elif st.session_state.current_scapple_note_index is not None:
        note = st.session_state.scapple_notes[st.session_state.current_scapple_note_index]

    if note is None:
        st.info("👈 Select a note to preview")
        return

    is_printed = st.session_state.print_history.is_printed(note)

    # Compact header with status
    col1, col2, col3 = st.columns([4, 1, 1])

    with col1:
        if is_printed:
            st.markdown(f"**✓ {note.title[:70]}**")
        else:
            st.markdown(f"**🆕 {note.title[:70]}**")

    with col2:
        if st.button("✏️", width='stretch', help="Edit note"):
            st.session_state.edit_mode = not st.session_state.edit_mode
            st.rerun()

    with col3:
        if st.button("🖨️", type="primary", width='stretch', help="Print label"):
            with st.spinner("Printing..."):
                try:
                    label_image = st.session_state.renderer.create_label(note.to_dict())
                    success = st.session_state.printer.print_label(label_image, rotate=90)

                    if success:
                        st.session_state.print_history.mark_printed(note)
                        st.success("✅ Printed!")
                        st.rerun()
                    else:
                        st.error("❌ Failed")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Edit mode or preview mode
    if st.session_state.edit_mode:
        with st.form("edit_note_form"):
            title = st.text_input("Title", value=note.title)
            author = st.text_input("Author", value=note.author)
            source = st.text_input("Source", value=note.source)
            body = st.text_area("Body", value=note.body, height=150)
            page = st.text_input("Page", value=note.page)

            col1, col2 = st.columns(2)

            with col1:
                if st.form_submit_button("💾 Save Changes", width='stretch'):
                    note_data = {
                        'title': title,
                        'author': author,
                        'source': source,
                        'body': body,
                        'page': page
                    }
                    success = save_note_to_file(note, note_data)
                    st.session_state.edit_mode = False
                    if success:
                        st.success("Saved to file!")
                    else:
                        st.warning("Note updated in memory only (file not saved)")
                    st.rerun()

            with col2:
                if st.form_submit_button("❌ Cancel", width='stretch'):
                    st.session_state.edit_mode = False
                    st.rerun()


    # Preview
    try:
        label_image = st.session_state.renderer.create_label(note.to_dict())

        # Preview options
        col1, col2 = st.columns([1, 4])
        with col1:
            rotate_preview = st.checkbox("Rotate 90°", value=False, help="Rotate to match print orientation")
        with col2:
            # Calculate actual size in mm
            width_mm = 62  # 62mm tape
            height_mm = int(label_image.height * 62 / label_image.width)
            st.caption(f"📏 ~{width_mm}×{height_mm}mm")

        # Rotate image if needed
        if rotate_preview:
            # Rotate 90 degrees clockwise to match printer output
            from PIL import Image
            rotated_image = label_image.rotate(-90, expand=True)
            st.image(rotated_image, width='stretch', caption="Preview (rotated 90°)")
        else:
            st.image(label_image, width='stretch', caption="Label Preview")

        # Compact note details
        with st.expander("ℹ️ Details", expanded=False):
            st.caption(f"**Author:** {note.author}")
            st.caption(f"**Source:** {note.source}")
            st.caption(f"**Page:** {note.page}")
            st.caption("**Text:**")
            st.caption(note.body)

    except Exception as e:
        st.error(f"Error generating preview: {e}")


def main():
    """Main application entry point."""
    init_session_state()

    # Auto-load notes on first run
    if not st.session_state.notes_loaded:
        load_notes()

    # Layout: sidebar + two columns (preview + note list)
    render_compact_sidebar()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        render_main_preview()

    with col2:
        render_note_list()


if __name__ == "__main__":
    main()
