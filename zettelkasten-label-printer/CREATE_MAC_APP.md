# Creating a macOS Application Shortcut

Follow these steps to create a double-clickable macOS application for the Label Printer:

## Method 1: Using Script Editor (Recommended)

1. **Open Script Editor**
   - Open `/Applications/Utilities/Script Editor.app`

2. **Load the AppleScript**
   - In Script Editor, click `File` → `Open`
   - Navigate to this folder and select `Label Printer.applescript`

3. **Save as Application**
   - Click `File` → `Export`
   - Set **File Format** to: `Application`
   - **Name**: `Label Printer`
   - **Where**: Choose `Applications` folder or `Desktop`
   - Uncheck "Show startup screen" (optional)
   - Click `Save`

4. **Use the App**
   - Double-click the `Label Printer.app` to start
   - Terminal will open and launch the web interface
   - Your default browser will open to the label printer
   - Use the **🛑 Stop App** button in the sidebar to shut down

## Method 2: Using Automator (Alternative)

1. **Open Automator**
   - Open `/Applications/Automator.app`
   - Choose `Application` as document type

2. **Add Run Shell Script**
   - Search for "Run Shell Script" in the actions library
   - Drag it to the workflow area
   - Paste this code:
   ```bash
   cd "$(dirname "$0")/../../.."
   cd zettelkasten-label-printer
   ./launch-label-printer.sh
   ```

3. **Save Application**
   - Click `File` → `Save`
   - Name: `Label Printer`
   - Save to: `Applications` or `Desktop`

## Method 3: Command Line Shortcut

If you prefer command line access, you can create an alias:

```bash
# Add to ~/.zshrc or ~/.bash_profile
alias label-printer='cd ~/Philosophy/zettelkasten-label-printer && ./launch-label-printer.sh'
```

Then simply run:
```bash
label-printer
```

## Custom Icon (Optional)

To add a custom printer icon to your app:

1. Find a printer icon image (PNG or ICNS)
2. Right-click the `Label Printer.app` → `Get Info`
3. Drag the icon to the small icon in the top-left of the Get Info window
4. Close the window

## Troubleshooting

**Permission Denied Error:**
```bash
chmod +x ~/Philosophy/zettelkasten-label-printer/launch-label-printer.sh
```

**Can't Open - Security Settings:**
- Right-click the app → `Open`
- Click `Open` in the security dialog
- Or: System Settings → Privacy & Security → Allow app

**Virtual Environment Issues:**
- Delete the `venv` folder and let the launcher recreate it
- Or run `./setup.sh` manually first
