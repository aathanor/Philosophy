"""
Label Renderer Module
Creates label images from note data using PIL/Pillow with the zettelkasten template.
"""

from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Tuple, List
import textwrap


class LabelRenderer:
    """Renders labels with H1 title, H2 author/source, body text, and page number."""

    def __init__(self, config: Dict):
        """
        Initialize the label renderer with configuration.

        Args:
            config: Configuration dictionary with fonts and layout settings
        """
        self.config = config
        self.fonts = config.get('fonts', {})
        self.layout = config.get('layout', {})
        self.label_size = config.get('printer', {}).get('label_size', '62')

        # Label dimensions (62mm tape = ~696 pixels at 300 DPI)
        # Height will be calculated based on content
        self.label_width = 696  # For 62mm tape
        self.padding = self.layout.get('padding', 20)
        self.line_spacing = self.layout.get('line_spacing', 1.2)

        # Load fonts - using default fonts if custom ones aren't available
        try:
            self.h1_font = ImageFont.truetype(
                self._get_font_path(),
                self.fonts.get('h1_size', 48)
            )
            self.h2_font = ImageFont.truetype(
                self._get_font_path(),
                self.fonts.get('h2_size', 32)
            )
            self.body_font = ImageFont.truetype(
                self._get_font_path(),
                self.fonts.get('body_size', 24)
            )
            self.page_font = ImageFont.truetype(
                self._get_font_path(),
                self.fonts.get('page_size', 20)
            )
        except Exception:
            # Fallback to default font
            self.h1_font = ImageFont.load_default()
            self.h2_font = ImageFont.load_default()
            self.body_font = ImageFont.load_default()
            self.page_font = ImageFont.load_default()

    def _get_font_path(self) -> str:
        """Get system font path based on font family."""
        font_family = self.fonts.get('font_family', 'Arial')

        # Common font paths on macOS
        font_paths = {
            'Arial': '/System/Library/Fonts/Supplemental/Arial.ttf',
            'Helvetica': '/System/Library/Fonts/Helvetica.ttc',
            'Times': '/System/Library/Fonts/Supplemental/Times New Roman.ttf',
            'Courier': '/System/Library/Fonts/Supplemental/Courier New.ttf',
        }

        return font_paths.get(font_family, font_paths['Arial'])

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        """
        Wrap text to fit within max_width.

        Args:
            text: Text to wrap
            font: Font to use for measuring
            max_width: Maximum width in pixels

        Returns:
            List of wrapped text lines
        """
        lines = []
        paragraphs = text.split('\n')

        for paragraph in paragraphs:
            if not paragraph.strip():
                lines.append('')
                continue

            words = paragraph.split()
            current_line = []

            for word in words:
                test_line = ' '.join(current_line + [word])
                bbox = font.getbbox(test_line)
                width = bbox[2] - bbox[0]

                if width <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        # Word is too long, force it
                        lines.append(word)

            if current_line:
                lines.append(' '.join(current_line))

        return lines

    def _calculate_text_height(self, lines: List[str], font: ImageFont.FreeTypeFont) -> int:
        """Calculate total height needed for wrapped text lines."""
        if not lines:
            return 0

        line_height = font.getbbox('Ay')[3] * self.line_spacing
        return int(line_height * len(lines))

    def create_label(self, note_data: Dict) -> Image.Image:
        """
        Create a label image from note data.

        Args:
            note_data: Dictionary containing:
                - title: Main title (H1)
                - author: Author name
                - source: Source/publication title
                - body: Quote/note body text
                - page: Page number

        Returns:
            PIL Image object of the rendered label
        """
        title = note_data.get('title', 'Untitled')
        author = note_data.get('author', '')
        source = note_data.get('source', '')
        body = note_data.get('body', '')
        page = note_data.get('page', '')

        # Calculate available width for text
        text_width = self.label_width - (2 * self.padding)

        # Wrap all text sections
        title_lines = self._wrap_text(title, self.h1_font, text_width)

        # Combine author and source for H2 line
        author_source = f"{author}"
        if source and author:
            author_source += f" - {source}"
        elif source:
            author_source = source
        author_lines = self._wrap_text(author_source, self.h2_font, text_width)

        body_lines = self._wrap_text(body, self.body_font, text_width)

        page_text = f"p. {page}" if page else ""

        # Calculate heights for each section
        title_height = self._calculate_text_height(title_lines, self.h1_font)
        author_height = self._calculate_text_height(author_lines, self.h2_font)
        body_height = self._calculate_text_height(body_lines, self.body_font)
        page_height = self.page_font.getbbox(page_text)[3] if page_text else 0

        # Calculate total label height with spacing between sections
        section_spacing = 15
        total_height = (
            self.padding * 2 +
            title_height + section_spacing +
            author_height + section_spacing +
            body_height +
            (section_spacing + page_height if page_text else 0)
        )

        # Enforce maximum height if configured
        max_height = self.layout.get('max_label_height', 2000)
        total_height = min(total_height, max_height)

        # Create the image
        image = Image.new('RGB', (self.label_width, int(total_height)), 'white')
        draw = ImageDraw.Draw(image)

        # Draw text sections
        y_position = self.padding

        # Draw title (H1)
        for line in title_lines:
            draw.text((self.padding, y_position), line, font=self.h1_font, fill='black')
            y_position += self.h1_font.getbbox(line)[3] * self.line_spacing

        y_position += section_spacing

        # Draw author/source (H2)
        for line in author_lines:
            draw.text((self.padding, y_position), line, font=self.h2_font, fill='black')
            y_position += self.h2_font.getbbox(line)[3] * self.line_spacing

        y_position += section_spacing

        # Draw body text
        for line in body_lines:
            draw.text((self.padding, y_position), line, font=self.body_font, fill='black')
            y_position += self.body_font.getbbox(line)[3] * self.line_spacing

        # Draw page number
        if page_text:
            y_position += section_spacing
            draw.text((self.padding, y_position), page_text, font=self.page_font, fill='black')

        return image

    def save_preview(self, image: Image.Image, filename: str) -> str:
        """
        Save label preview to file.

        Args:
            image: Label image to save
            filename: Output filename

        Returns:
            Path to saved file
        """
        image.save(filename, 'PNG')
        return filename
