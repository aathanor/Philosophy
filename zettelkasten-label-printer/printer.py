"""
Printer Module
Interfaces with Brother QL-810W label printer using brother_ql library.
"""

from PIL import Image
from typing import Optional, List
import logging
import warnings

# Add compatibility shim for Pillow 10+ (brother_ql uses deprecated ANTIALIAS)
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

# Suppress deprecation warnings from brother_ql library
warnings.filterwarnings('ignore', message='.*brother_ql.devicedependent.*')

try:
    from brother_ql.conversion import convert
    from brother_ql.backends.helpers import send
    from brother_ql.raster import BrotherQLRaster
    BROTHER_QL_AVAILABLE = True
except ImportError:
    BROTHER_QL_AVAILABLE = False
    logging.warning("brother_ql library not found. Printing will not be available.")


class BrotherQLPrinter:
    """Interface for Brother QL series label printers."""

    def __init__(self, model: str = "QL-810W", connection: str = "usb://0x04f9:0x209b",
                 label_size: str = "62"):
        """
        Initialize the printer interface.

        Args:
            model: Printer model (e.g., "QL-810W")
            connection: Connection string (usb://... or tcp://...)
            label_size: Label size in mm (e.g., "62" for 62mm continuous)
        """
        self.model = model
        self.connection = connection
        self.label_size = label_size
        self.logger = logging.getLogger(__name__)

        if not BROTHER_QL_AVAILABLE:
            raise ImportError(
                "brother_ql library is required for printing. "
                "Install it with: pip install brother-ql"
            )

    def print_label(self, image: Image.Image, rotate: int = 90) -> bool:
        """
        Print a label image to the Brother QL printer.

        Args:
            image: PIL Image object to print
            rotate: Rotation angle (0, 90, 180, 270). Default 90 for landscape.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Debug: Check available backends
            try:
                from brother_ql.backends import available_backends
                backends = available_backends()
                self.logger.info(f"Available backends: {backends}")
            except Exception as e:
                self.logger.warning(f"Could not check available backends: {e}")

            # Debug: Check pyusb availability
            try:
                import usb.core
                devices = list(usb.core.find(find_all=True, idVendor=0x04f9))
                self.logger.info(f"Found {len(devices)} Brother USB device(s)")
                for dev in devices:
                    self.logger.info(f"  Device: VID=0x04f9, PID=0x{dev.idProduct:04x}")
            except ImportError:
                self.logger.error("pyusb module not installed! Install with: pip install pyusb")
            except Exception as e:
                self.logger.warning(f"USB device detection failed: {e}")

            self.logger.info(f"Attempting to print to: {self.connection}")

            # Create raster data
            qlr = BrotherQLRaster(self.model)
            qlr.exception_on_warning = True

            # Convert image to raster format
            instructions = convert(
                qlr=qlr,
                images=[image],
                label=self.label_size,
                rotate=rotate,
                threshold=70.0,
                dither=False,
                compress=False,
                red=False,
                dpi_600=False,
                hq=True,
                cut=True
            )

            # Determine backend
            backend = 'pyusb' if self.connection.startswith('usb') else 'network'
            self.logger.info(f"Using backend: {backend}")

            # Send to printer
            send(
                instructions=instructions,
                printer_identifier=self.connection,
                backend_identifier=backend,
                blocking=True
            )

            self.logger.info(f"Successfully printed label")
            return True

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Full error details: {type(e).__name__}: {error_msg}")

            if "No backend available" in error_msg:
                self.logger.error(
                    "Printer backend not available. Troubleshooting:\n"
                    "1. Install pyusb: pip install pyusb\n"
                    "2. Connect Brother QL printer via USB\n"
                    "3. Check USB permissions (may need sudo on some systems)\n"
                    f"4. Try running: brother_ql discover usb"
                )
            elif "Access denied" in error_msg or "Permission denied" in error_msg:
                self.logger.error(
                    "USB permission denied. On macOS, you may need to:\n"
                    "1. Disconnect and reconnect the printer\n"
                    "2. Grant USB permissions in System Settings\n"
                    "3. Try running with sudo (not recommended)"
                )
            return False

    def print_labels(self, images: List[Image.Image], rotate: int = 90) -> int:
        """
        Print multiple label images.

        Args:
            images: List of PIL Image objects to print
            rotate: Rotation angle for all labels

        Returns:
            Number of successfully printed labels
        """
        success_count = 0

        for i, image in enumerate(images):
            self.logger.info(f"Printing label {i + 1} of {len(images)}")
            if self.print_label(image, rotate):
                success_count += 1
            else:
                self.logger.error(f"Failed to print label {i + 1}")

        return success_count

    def test_connection(self) -> bool:
        """
        Test if the printer is accessible.

        Returns:
            True if printer responds, False otherwise
        """
        try:
            # Create a small test image
            test_image = Image.new('RGB', (100, 100), 'white')

            # Try to create raster data (doesn't send to printer)
            qlr = BrotherQLRaster(self.model)
            convert(
                qlr=qlr,
                images=[test_image],
                label=self.label_size,
                rotate=90,
                threshold=70.0,
                dither=False,
                compress=False,
                red=False,
                dpi_600=False,
                hq=True,
                cut=True
            )

            self.logger.info("Printer connection test successful (raster creation)")
            return True

        except Exception as e:
            self.logger.error(f"Printer connection test failed: {e}")
            return False

    @staticmethod
    def list_available_printers() -> List[str]:
        """
        List available Brother QL printers.

        Returns:
            List of printer identifiers
        """
        if not BROTHER_QL_AVAILABLE:
            return []

        printers = []

        # Try to detect USB printers
        try:
            import usb.core

            # Brother vendor ID: 0x04f9
            devices = usb.core.find(find_all=True, idVendor=0x04f9)

            for device in devices:
                identifier = f"usb://0x04f9:0x{device.idProduct:04x}"
                printers.append(identifier)

        except Exception as e:
            logging.warning(f"Could not scan for USB printers: {e}")

        return printers

    def get_supported_label_sizes(self) -> List[str]:
        """
        Get list of supported label sizes for this printer model.

        Returns:
            List of label size identifiers
        """
        # Common sizes for QL-810W
        # These are the DK tape widths
        return [
            "12",      # 12mm
            "29",      # 29mm
            "38",      # 38mm
            "50",      # 50mm
            "54",      # 54mm
            "62",      # 62mm (recommended for zettelkasten cards)
            "102",     # 102mm (wide)
        ]


class MockPrinter:
    """Mock printer for testing without hardware."""

    def __init__(self, model: str = "QL-810W", connection: str = "mock",
                 label_size: str = "62"):
        """Initialize mock printer."""
        self.model = model
        self.connection = connection
        self.label_size = label_size
        self.logger = logging.getLogger(__name__)
        self.printed_images = []

    def print_label(self, image: Image.Image, rotate: int = 90) -> bool:
        """Mock print operation - saves to memory."""
        self.logger.info(f"[MOCK] Printing label {len(self.printed_images) + 1}")
        self.printed_images.append(image)
        return True

    def print_labels(self, images: List[Image.Image], rotate: int = 90) -> int:
        """Mock batch print."""
        for image in images:
            self.print_label(image, rotate)
        return len(images)

    def test_connection(self) -> bool:
        """Mock connection test - always succeeds."""
        self.logger.info("[MOCK] Printer connection OK")
        return True

    @staticmethod
    def list_available_printers() -> List[str]:
        """Mock printer list."""
        return ["mock://printer"]

    def get_supported_label_sizes(self) -> List[str]:
        """Mock label sizes."""
        return ["12", "29", "38", "50", "54", "62", "102"]


def create_printer(config: dict, use_mock: bool = False):
    """
    Create a printer instance from configuration.

    Args:
        config: Configuration dictionary with printer settings
        use_mock: If True, create mock printer for testing

    Returns:
        Printer instance (BrotherQLPrinter or MockPrinter)
    """
    printer_config = config.get('printer', {})
    model = printer_config.get('model', 'QL-810W')
    connection = printer_config.get('connection', 'usb://0x04f9:0x209b')
    label_size = printer_config.get('label_size', '62')

    if use_mock or not BROTHER_QL_AVAILABLE:
        return MockPrinter(model, connection, label_size)
    else:
        return BrotherQLPrinter(model, connection, label_size)
