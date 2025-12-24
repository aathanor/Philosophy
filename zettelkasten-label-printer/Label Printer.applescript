-- Label Printer macOS Application
-- Save as Application in Script Editor to create a double-clickable app

on run
	-- Path to the zettelkasten-label-printer folder
	-- Update this path if you moved the folder to a different location
	set projectFolder to (path to home folder as text) & "Philosophy:zettelkasten-label-printer"
	set projectPath to POSIX path of projectFolder

	-- Open Terminal and run the launcher
	tell application "Terminal"
		activate
		do script "cd " & quoted form of projectPath & " && ./launch-label-printer.sh"
	end tell
end run
