-- Label Printer macOS Application
-- Double-click to run, or save as Application in Script Editor

on run
	-- Get the directory containing this script
	set scriptPath to POSIX path of (path to me)
	set appFolder to do shell script "dirname " & quoted form of scriptPath

	-- Build path to launcher script
	set launcherScript to appFolder & "/launch-label-printer.sh"

	-- Open Terminal and run the launcher
	tell application "Terminal"
		activate
		do script "cd " & quoted form of appFolder & " && ./launch-label-printer.sh"
	end tell
end run
