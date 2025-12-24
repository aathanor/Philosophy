"""
Zettelkasten Label Printer - Main Streamlit Application
A web interface for printing zettelkasten cards on Brother QL-810W label printer.
"""

import streamlit as st
import logging
from pathlib import Path
import json

from config_loader import ConfigLoader
from parsers import get_all_notes, Note, ZoteroParser, HighlightedParser
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


def init_session_state():
    """Initialize session state variables."""
    if 'config' not in st.session_state:
        st.session_state.config = ConfigLoader('config.yaml')

    if 'notes' not in st.session_state:
        st.session_state.notes = []

    if 'current_note_index' not in st.session_state:
        st.session_state.current_note_index = None

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

    # Expand paths
    zotero_path = config.expand_path(zotero_folder)
    highlighted_path = config.expand_path(highlighted_folder)

    # Create folders if they don't exist
    zotero_path.mkdir(parents=True, exist_ok=True)
    highlighted_path.mkdir(parents=True, exist_ok=True)

    # Log folder info
    logger.info(f"Loading notes from:")
    logger.info(f"  Zotero: {zotero_path}")
    logger.info(f"  Highlighted: {highlighted_path}")

    # Count files
    zotero_files = list(zotero_path.glob('*.md'))
    highlighted_files = list(highlighted_path.glob('*.md'))
    logger.info(f"Found {len(zotero_files)} Zotero files, {len(highlighted_files)} Highlighted files")

    # Load notes
    notes = get_all_notes(str(zotero_path), str(highlighted_path))
    logger.info(f"Parsed {len(notes)} total notes")

    # Sort: unprinted first, then by title
    notes.sort(key=lambda n: (
        st.session_state.print_history.is_printed(n),  # False (unprinted) sorts first
        n.title.lower()
    ))

    st.session_state.notes = notes
    return notes


def render_compact_sidebar():
    """Render compact sidebar with stats and controls."""
    with st.sidebar:
        st.title("🏷️ Label Printer")

        # Stats first
        st.metric("Total Notes", len(st.session_state.notes))
        unprinted = sum(1 for n in st.session_state.notes
                       if not st.session_state.print_history.is_printed(n))
        st.metric("Unprinted", unprinted)

        st.divider()

        # Reload button (prominent)
        if st.button("🔄 Reload Notes", type="primary", use_container_width=True):
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

            if st.button("🔍 Test Connection", use_container_width=True):
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

            if st.button("💾 Save", use_container_width=True):
                st.session_state.config.set('data_sources.zotero_export_folder', zotero_folder)
                st.session_state.config.set('data_sources.highlighted_export_folder', highlighted_folder)
                st.session_state.config.save()
                st.success("Saved!")

        with st.expander("🗑️ Clear History", expanded=False):
            st.caption("Clear print tracking")
            if st.button("Clear Print History", use_container_width=True):
                st.session_state.print_history.clear_history()
                st.success("History cleared!")
                st.rerun()


def render_note_list():
    """Render note list in right column."""
    st.subheader("📝 Notes")

    # Search
    search = st.text_input("🔍 Search", placeholder="Filter notes...", key="search")

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

    st.caption(f"{len(filtered_notes)} of {len(st.session_state.notes)} notes")

    # Scrollable note list
    for i, note in enumerate(filtered_notes):
        is_printed = st.session_state.print_history.is_printed(note)

        # Find actual index in full list
        actual_index = st.session_state.notes.index(note)
        is_selected = st.session_state.current_note_index == actual_index

        # Status indicator
        if is_printed:
            status = "✓"
            style = "🔖"
        else:
            status = "●"
            style = "🆕"

        # Button for each note
        button_label = f"{style} {status} {note.title[:40]}..."

        if st.button(
            button_label,
            key=f"note_{actual_index}",
            use_container_width=True,
            type="primary" if is_selected else "secondary"
        ):
            st.session_state.current_note_index = actual_index
            st.session_state.edit_mode = False
            st.rerun()

        # Show author under button if not printed
        if not is_printed:
            st.caption(f"   {note.author}")


def save_note_to_file(note: Note, note_data: dict):
    """Save edited note back to its source file."""
    # Update note object in memory
    note.title = note_data['title']
    note.author = note_data['author']
    note.source = note_data['source']
    note.body = note_data['body']
    note.page = note_data['page']


def render_main_preview():
    """Render main preview and print area."""
    if st.session_state.current_note_index is None:
        st.info("👈 Select a note from the list to preview and print")
        return

    note = st.session_state.notes[st.session_state.current_note_index]
    is_printed = st.session_state.print_history.is_printed(note)

    # Header with status
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        if is_printed:
            st.subheader(f"✓ {note.title[:60]}")
            print_time = st.session_state.print_history.get_print_time(note)
            if print_time:
                st.caption(f"Printed: {print_time[:16]}")
        else:
            st.subheader(f"🆕 {note.title[:60]}")
            st.caption("Not yet printed")

    with col2:
        if st.button("✏️ Edit", use_container_width=True):
            st.session_state.edit_mode = not st.session_state.edit_mode
            st.rerun()

    with col3:
        if st.button("🖨️ Print", type="primary", use_container_width=True):
            with st.spinner("Printing..."):
                try:
                    label_image = st.session_state.renderer.create_label(note.to_dict())
                    success = st.session_state.printer.print_label(label_image, rotate=90)

                    if success:
                        st.session_state.print_history.mark_printed(note)
                        st.success("✅ Printed!")
                        st.rerun()
                    else:
                        st.error("❌ Print failed")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    # Edit mode or preview mode
    if st.session_state.edit_mode:
        st.subheader("✏️ Edit Note")

        with st.form("edit_note_form"):
            title = st.text_input("Title", value=note.title)
            author = st.text_input("Author", value=note.author)
            source = st.text_input("Source", value=note.source)
            body = st.text_area("Body", value=note.body, height=200)
            page = st.text_input("Page", value=note.page)

            col1, col2 = st.columns(2)

            with col1:
                if st.form_submit_button("💾 Save Changes", use_container_width=True):
                    note_data = {
                        'title': title,
                        'author': author,
                        'source': source,
                        'body': body,
                        'page': page
                    }
                    save_note_to_file(note, note_data)
                    st.session_state.edit_mode = False
                    st.success("Saved!")
                    st.rerun()

            with col2:
                if st.form_submit_button("❌ Cancel", use_container_width=True):
                    st.session_state.edit_mode = False
                    st.rerun()

        st.divider()

    # Preview
    st.subheader("👁️ Label Preview")

    try:
        label_image = st.session_state.renderer.create_label(note.to_dict())
        st.image(label_image, use_column_width=True)

        # Note details below preview
        with st.expander("📄 Note Details", expanded=False):
            st.write(f"**Author:** {note.author}")
            st.write(f"**Source:** {note.source}")
            st.write(f"**Page:** {note.page}")
            st.write(f"**Body:**")
            st.write(note.body)

    except Exception as e:
        st.error(f"Error generating preview: {e}")


def main():
    """Main application entry point."""
    init_session_state()

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
