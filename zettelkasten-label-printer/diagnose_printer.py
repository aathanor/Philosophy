#!/usr/bin/env python3
"""
Diagnostic script to check printer connectivity and backend availability.
Run this to diagnose why printing isn't working.
"""

print("=" * 60)
print("Brother QL Printer Diagnostics")
print("=" * 60)

# 1. Check pyusb installation
print("\n1. Checking pyusb installation...")
try:
    import usb.core
    import usb.util
    print("   ✓ pyusb is installed")
    print(f"   Version: {usb.__version__ if hasattr(usb, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"   ✗ pyusb NOT installed: {e}")
    print("   Install with: pip install pyusb")

# 2. Check brother_ql installation
print("\n2. Checking brother_ql installation...")
try:
    import brother_ql
    print("   ✓ brother_ql is installed")
    print(f"   Version: {brother_ql.__version__ if hasattr(brother_ql, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"   ✗ brother_ql NOT installed: {e}")

# 3. Check available backends
print("\n3. Checking available backends...")
try:
    from brother_ql.backends import available_backends
    backends = available_backends()
    print(f"   Available backends: {backends}")
    if 'pyusb' in backends:
        print("   ✓ pyusb backend is available")
    else:
        print("   ✗ pyusb backend is NOT available")
except Exception as e:
    print(f"   ✗ Error checking backends: {e}")

# 4. Scan for Brother USB devices
print("\n4. Scanning for Brother USB devices...")
try:
    import usb.core
    # Brother vendor ID
    devices = list(usb.core.find(find_all=True, idVendor=0x04f9))
    if devices:
        print(f"   ✓ Found {len(devices)} Brother USB device(s):")
        for dev in devices:
            print(f"      - VID: 0x04f9, PID: 0x{dev.idProduct:04x}")
            print(f"        Bus: {dev.bus}, Address: {dev.address}")
            try:
                manufacturer = usb.util.get_string(dev, dev.iManufacturer)
                product = usb.util.get_string(dev, dev.iProduct)
                print(f"        Manufacturer: {manufacturer}")
                print(f"        Product: {product}")
            except usb.core.USBError as ue:
                print(f"        ⚠ Cannot read device info: {ue}")
                if "Access denied" in str(ue) or "Permission denied" in str(ue):
                    print(f"        ⚠ USB PERMISSION ISSUE DETECTED")
            except:
                pass
    else:
        print("   ✗ No Brother USB devices found")
        print("   Possible reasons:")
        print("   - Printer not connected")
        print("   - Printer turned off")
        print("   - USB cable issue")
        print("   - USB permissions issue")
except ImportError:
    print("   ✗ Cannot scan: pyusb not installed")
except Exception as e:
    print(f"   ✗ Error scanning USB: {e}")

# 5. Try brother_ql discover command
print("\n5. Testing brother_ql discover command...")
try:
    from brother_ql.backends import available_backends
    from brother_ql import BrotherQLRaster

    if 'pyusb' in available_backends():
        print("   Running: brother_ql discover usb")
        import subprocess
        result = subprocess.run(['brother_ql', 'discover', 'usb'],
                              capture_output=True, text=True, timeout=5)
        if result.stdout:
            print(f"   Output: {result.stdout}")
        if result.stderr:
            print(f"   Errors: {result.stderr}")
    else:
        print("   ✗ Skipped: pyusb backend not available")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 6. Check libusb on macOS
print("\n6. Checking libusb (macOS)...")
try:
    import subprocess
    result = subprocess.run(['brew', 'list', 'libusb'],
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("   ✓ libusb is installed via Homebrew")
    else:
        print("   ✗ libusb NOT installed")
        print("   Install with: brew install libusb")
except FileNotFoundError:
    print("   ℹ Homebrew not found (or not in PATH)")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 7. Check for network connectivity (QL-810W has WiFi)
print("\n7. Checking for network printer...")
print("   ℹ Your QL-810W supports WiFi. You can use network instead of USB!")
print("   To find your printer's IP address:")
print("   - Check your printer's display/settings menu")
print("   - Check your router's connected devices list")
print("   - Try: brother_ql discover network")
try:
    import subprocess
    result = subprocess.run(['brother_ql', 'discover', 'network'],
                          capture_output=True, text=True, timeout=10)
    if result.stdout and result.stdout.strip():
        print(f"   ✓ Network discovery result:")
        print(f"      {result.stdout}")
    else:
        print("   ℹ No network printers found (or discovery timed out)")
except subprocess.TimeoutExpired:
    print("   ℹ Network discovery timed out (this is normal)")
except Exception as e:
    print(f"   ℹ Network discovery not available: {e}")

print("\n" + "=" * 60)
print("Diagnosis complete!")
print("=" * 60)
print("\n💡 RECOMMENDATIONS:")
print("   If USB has permission issues, use network connection instead:")
print("   1. Get your printer's IP address from printer settings or router")
print("   2. Update config.yaml connection to: tcp://192.168.x.x")
print("   3. This bypasses all USB permission issues!")
print("=" * 60)
