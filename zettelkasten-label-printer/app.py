"""
Zettelkasten Label Printer - Main Streamlit Application
A web interface for printing zettelkasten cards on Brother QL-810W label printer.
"""

import streamlit as st
import logging
from pathlib import Path
from io import BytesIO
import os

from config_loader import ConfigLoader
from parsers import get_all_notes, Note, ZoteroParser, HighlightedParser
from label_renderer import LabelRenderer
from printer import create_printer, BROTHER_QL_AVAILABLE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="Zettelkasten Label Printer",
    page_icon="🏷️",
    layout="wide"
)


def init_session_state():
    """Initialize session state variables."""
    if 'config' not in st.session_state:
        st.session_state.config = ConfigLoader('config.yaml')

    if 'notes' not in st.session_state:
        st.session_state.notes = []

    if 'selected_notes' not in st.session_state:
        st.session_state.selected_notes = []

    if 'printer' not in st.session_state:
        config = st.session_state.config.get_all()
        # Use mock printer if brother_ql is not available
        st.session_state.printer = create_printer(config, use_mock=not BROTHER_QL_AVAILABLE)

    if 'renderer' not in st.session_state:
        config = st.session_state.config.get_all()
        st.session_state.renderer = LabelRenderer(config)


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

    # Load notes
    notes = get_all_notes(str(zotero_path), str(highlighted_path))

    st.session_state.notes = notes
    return notes


def render_sidebar():
    """Render sidebar with configuration and controls."""
    st.sidebar.title("⚙️ Settings")

    # Printer status
    st.sidebar.subheader("Printer Status")

    if not BROTHER_QL_AVAILABLE:
        st.sidebar.warning("⚠️ brother_ql library not installed. Using mock printer.")
        st.sidebar.info("Install with: `pip install brother-ql`")
    else:
        printer_model = st.session_state.config.get('printer.model', 'QL-810W')
        printer_conn = st.session_state.config.get('printer.connection', 'usb')
        st.sidebar.success(f"✅ Printer: {printer_model}")
        st.sidebar.text(f"Connection: {printer_conn}")

    # Test connection button
    if st.sidebar.button("🔍 Test Printer Connection"):
        with st.spinner("Testing printer connection..."):
            if st.session_state.printer.test_connection():
                st.sidebar.success("✅ Printer connection OK!")
            else:
                st.sidebar.error("❌ Printer connection failed")

    st.sidebar.divider()

    # Data sources
    st.sidebar.subheader("Data Sources")

    zotero_folder = st.sidebar.text_input(
        "Zotero Export Folder",
        value=st.session_state.config.get('data_sources.zotero_export_folder', '')
    )

    highlighted_folder = st.sidebar.text_input(
        "Highlighted Export Folder",
        value=st.session_state.config.get('data_sources.highlighted_export_folder', '')
    )

    # Save config button
    if st.sidebar.button("💾 Save Configuration"):
        st.session_state.config.set('data_sources.zotero_export_folder', zotero_folder)
        st.session_state.config.set('data_sources.highlighted_export_folder', highlighted_folder)
        st.session_state.config.save()
        st.sidebar.success("Configuration saved!")

    st.sidebar.divider()

    # Reload notes button
    if st.sidebar.button("🔄 Reload Notes", type="primary"):
        with st.spinner("Loading notes..."):
            notes = load_notes()
            st.sidebar.success(f"Loaded {len(notes)} notes")

    # Note statistics
    st.sidebar.subheader("📊 Statistics")
    st.sidebar.metric("Total Notes", len(st.session_state.notes))
    st.sidebar.metric("Selected", len(st.session_state.selected_notes))


def render_main_content():
    """Render main content area."""
    st.title("🏷️ Zettelkasten Label Printer")
    st.markdown("Print notes from Zotero and Highlighted app to Brother QL-810W labels")

    # Check if we have notes
    if not st.session_state.notes:
        st.info("👈 Click 'Reload Notes' in the sidebar to load notes from your export folders")

        # Show folder locations
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📚 Zotero Exports")
            zotero_path = st.session_state.config.expand_path(
                st.session_state.config.get('data_sources.zotero_export_folder', '')
            )
            st.code(str(zotero_path))
            if zotero_path.exists():
                md_files = list(zotero_path.glob('*.md'))
                st.write(f"Found {len(md_files)} markdown files")
            else:
                st.warning("Folder does not exist")

        with col2:
            st.subheader("📱 Highlighted Exports")
            highlighted_path = st.session_state.config.expand_path(
                st.session_state.config.get('data_sources.highlighted_export_folder', '')
            )
            st.code(str(highlighted_path))
            if highlighted_path.exists():
                md_files = list(highlighted_path.glob('*.md'))
                st.write(f"Found {len(md_files)} markdown files")
            else:
                st.warning("Folder does not exist")

        return

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📝 Browse Notes", "🎨 Preview & Print", "✏️ Create Custom Note"])

    with tab1:
        render_browse_notes_tab()

    with tab2:
        render_preview_print_tab()

    with tab3:
        render_create_note_tab()


def render_browse_notes_tab():
    """Render the browse notes tab."""
    st.subheader("Browse and Select Notes")

    # Search/filter
    search_term = st.text_input("🔍 Search notes", placeholder="Search by title, author, or content...")

    # Filter notes based on search
    filtered_notes = st.session_state.notes
    if search_term:
        search_lower = search_term.lower()
        filtered_notes = [
            note for note in st.session_state.notes
            if search_lower in note.title.lower()
            or search_lower in note.author.lower()
            or search_lower in note.body.lower()
            or search_lower in note.source.lower()
        ]

    st.write(f"Showing {len(filtered_notes)} of {len(st.session_state.notes)} notes")

    # Display notes with selection
    for i, note in enumerate(filtered_notes):
        with st.expander(f"**{note.title[:60]}** - {note.author}"):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"**Author:** {note.author}")
                st.write(f"**Source:** {note.source}")
                st.write(f"**Page:** {note.page}")
                st.write(f"**Quote:**")
                st.write(note.body)

            with col2:
                # Select button
                is_selected = note in st.session_state.selected_notes

                if is_selected:
                    if st.button(f"✅ Selected", key=f"deselect_{i}"):
                        st.session_state.selected_notes.remove(note)
                        st.rerun()
                else:
                    if st.button(f"Select", key=f"select_{i}"):
                        st.session_state.selected_notes.append(note)
                        st.rerun()

    # Bulk actions
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Select All Visible", use_container_width=True):
            st.session_state.selected_notes = list(set(st.session_state.selected_notes + filtered_notes))
            st.rerun()

    with col2:
        if st.button("Clear Selection", use_container_width=True):
            st.session_state.selected_notes = []
            st.rerun()

    with col3:
        st.write(f"**{len(st.session_state.selected_notes)} notes selected**")


def render_preview_print_tab():
    """Render the preview and print tab."""
    st.subheader("Preview and Print Labels")

    if not st.session_state.selected_notes:
        st.info("No notes selected. Go to 'Browse Notes' tab to select notes.")
        return

    st.write(f"**{len(st.session_state.selected_notes)} labels ready to print**")

    # Preview options
    col1, col2 = st.columns([2, 1])

    with col1:
        preview_index = st.slider(
            "Preview label",
            0,
            len(st.session_state.selected_notes) - 1,
            0
        )

    with col2:
        rotation = st.selectbox("Rotation", [0, 90, 180, 270], index=1)

    # Generate and show preview
    note = st.session_state.selected_notes[preview_index]

    with st.spinner("Generating preview..."):
        try:
            label_image = st.session_state.renderer.create_label(note.to_dict())

            # Display preview
            st.image(label_image, caption=f"Preview: {note.title[:50]}", use_container_width=True)

            # Note details
            with st.expander("Note Details"):
                st.write(f"**Title:** {note.title}")
                st.write(f"**Author:** {note.author}")
                st.write(f"**Source:** {note.source}")
                st.write(f"**Page:** {note.page}")
                st.write(f"**Body:** {note.body}")

        except Exception as e:
            st.error(f"Error generating preview: {e}")
            return

    # Print buttons
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🖨️ Print This Label", type="primary", use_container_width=True):
            with st.spinner("Printing..."):
                try:
                    label_image = st.session_state.renderer.create_label(note.to_dict())
                    success = st.session_state.printer.print_label(label_image, rotate=rotation)

                    if success:
                        st.success("✅ Label printed successfully!")
                    else:
                        st.error("❌ Failed to print label")
                except Exception as e:
                    st.error(f"Error printing: {e}")

    with col2:
        if st.button("🖨️ Print All Selected", use_container_width=True):
            with st.spinner(f"Printing {len(st.session_state.selected_notes)} labels..."):
                try:
                    images = []
                    for note in st.session_state.selected_notes:
                        label_image = st.session_state.renderer.create_label(note.to_dict())
                        images.append(label_image)

                    success_count = st.session_state.printer.print_labels(images, rotate=rotation)

                    st.success(f"✅ Printed {success_count} of {len(images)} labels")

                except Exception as e:
                    st.error(f"Error printing: {e}")


def render_create_note_tab():
    """Render the create custom note tab."""
    st.subheader("Create Custom Note")

    st.write("Create a one-off label without importing from files")

    # Input form
    with st.form("custom_note_form"):
        title = st.text_input("Title", placeholder="Note title or main concept")
        author = st.text_input("Author", placeholder="Author name")
        source = st.text_input("Source", placeholder="Book/article title")
        body = st.text_area("Body", placeholder="Quote or note content", height=150)
        page = st.text_input("Page", placeholder="Page number")

        col1, col2 = st.columns(2)

        with col1:
            preview_btn = st.form_submit_button("👁️ Preview", use_container_width=True)

        with col2:
            print_btn = st.form_submit_button("🖨️ Print", use_container_width=True, type="primary")

    if preview_btn or print_btn:
        if not title and not body:
            st.warning("Please enter at least a title or body text")
        else:
            note = Note(
                title=title or "Untitled",
                author=author,
                source=source,
                body=body,
                page=page
            )

            try:
                label_image = st.session_state.renderer.create_label(note.to_dict())

                if preview_btn:
                    st.image(label_image, caption="Preview", use_container_width=True)

                if print_btn:
                    with st.spinner("Printing..."):
                        success = st.session_state.printer.print_label(label_image)

                        if success:
                            st.success("✅ Label printed successfully!")
                        else:
                            st.error("❌ Failed to print label")

            except Exception as e:
                st.error(f"Error: {e}")


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
