#!/bin/bash
export PATH="/opt/homebrew/bin:$PATH"
cd "/Users/florin/Documents/--- PHILOSOPHY---/Philosophy"

DELAY=30
LOCKFILE="/tmp/git-debounce.lock"

# Function to perform git operations
do_git_commit() {
    git add .
    git commit -m "Auto-commit $(date)"
    git push
    rm -f "$LOCKFILE"
}

# Check if another instance is running
if [ -f "$LOCKFILE" ]; then
    # Update timestamp and exit
    echo "$(date +%s)" > "$LOCKFILE"
    exit 0
fi

# Create lockfile with current timestamp
echo "$(date +%s)" > "$LOCKFILE"

# Wait and check for new changes
while [ -f "$LOCKFILE" ]; do
    sleep $DELAY
    
    if [ -f "$LOCKFILE" ]; then
        LAST_UPDATE=$(cat "$LOCKFILE")
        CURRENT_TIME=$(date +%s)
        
        # If no updates in the last DELAY seconds, commit
        if [ $((CURRENT_TIME - LAST_UPDATE)) -ge $DELAY ]; then
            do_git_commit
            break
        fi
    fi
done