# Zettelkasten Label Printer

A Python application for printing zettelkasten cards on a Brother QL-810W label printer from Zotero and Highlighted app exports.

## Features

- Print labels from Zotero annotations/highlights
- Print labels from Highlighted app (iOS) exports
- Custom template with H1 title, H2 author/source, body text, and page numbers
- Web-based UI using Streamlit
- Support for Brother QL-810W label printer using raster language
- Preview labels before printing
- Configurable fonts and layout

## Installation

### Quick Setup (Recommended)

1. Clone or download this repository
2. Run the setup script:
   ```bash
   cd zettelkasten-label-printer
   ./setup.sh
   ```

The setup script will:
- Check your Python version
- Create a virtual environment (venv/)
- Install all dependencies in the virtual environment

### Manual Installation

If you prefer manual setup:

1. Install Python 3.8 or higher
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your printer and folders in `config.yaml`

## Usage

### Run the Streamlit App

**Using the run script (recommended):**
```bash
./run.sh
```

**Or manually:**
```bash
source venv/bin/activate
streamlit run app.py
```

This will open a web interface where you can:
- Select notes from Zotero or Highlighted exports
- Preview labels before printing
- Print individual labels or batch print

### Export from Zotero

1. In Zotero, select items with annotations
2. Right-click → "Add Note from Annotations"
3. Export notes as Markdown to the configured folder

### Export from Highlighted App

1. In Highlighted app, select your annotations
2. Export as Markdown
3. Save to the configured folder (can be synced via iCloud)

## Configuration

Edit `config.yaml` to customize:
- Printer connection (USB or network)
- Font sizes for H1, H2, body text
- Label dimensions
- Export folder locations

## Printer Setup

For Brother QL-810W on Mac:

### USB Connection
The printer should be automatically detected when connected via USB.
Connection string: `usb://0x04f9:0x209b`

### Network Connection
If using Wi-Fi, find your printer's IP address and use:
Connection string: `tcp://192.168.1.XXX` (replace with your printer's IP)

## Label Size

Recommended: 62mm continuous length tape (DK-22205 or compatible)

## Troubleshooting

- If printer is not detected, check USB connection or network IP
- Ensure the Brother QL-810W is powered on and in ready state
- Check that the correct label roll is loaded
- Preview labels before printing to verify formatting

## Sources

This project uses:
- [brother_ql](https://github.com/pklaus/brother_ql) - Python package for Brother QL printers
- [Pillow](https://python-pillow.org/) - Image generation
- [Streamlit](https://streamlit.io/) - Web interface
