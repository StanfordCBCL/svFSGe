#!/bin/bash

# Script to copy latest partitioned_* folder with selective VTU/BIN file handling
# Configuration
SOURCE_BASE="${SOURCE_BASE:-bouchet:~/svFSGe}"  # Remote source on bouchet (sc3544)
DEST_DIR="${DEST_DIR:-.}"  # Current directory by default
KEEP_LAST_N_PULSATILE=400  # Keep last N pulsatile VTU files (ignore .bin files)
KEEP_LAST_N_STEADY=10      # Keep last N steady VTU files (ignore .bin files)

set -e  # Exit on error

# Optional argument: folder name to copy (defaults to latest)
FOLDER_ARG="${1:-}"

# SSH ControlMaster settings - authenticate once, reuse connection
SSH_HOST="${SOURCE_BASE%%:*}"
CONTROL_PATH="/tmp/ssh_control_${SSH_HOST}_%h_%p_%r"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$CONTROL_PATH -o ControlPersist=600"

echo "=== Partitioned Folder Copy Script ==="
echo "Source: $SOURCE_BASE"
echo "Destination: $DEST_DIR"
echo ""

# Establish master connection (one-time authentication)
if [[ $SOURCE_BASE == *:* ]]; then
    echo "Establishing SSH connection (you'll authenticate once here)..."
    ssh $SSH_OPTS -t "$SSH_HOST" "echo 'Connected'" > /dev/null 2>&1
    echo ""
fi

# Find the target partitioned_* directory (use argument if provided, else latest)
if [ -n "$FOLDER_ARG" ]; then
    FOLDER_NAME="$FOLDER_ARG"
    if [[ $SOURCE_BASE == *:* ]]; then
        REMOTE_HOST="${SOURCE_BASE%%:*}"
        LATEST_DIR="$REMOTE_HOST:~/svFSGe/$FOLDER_NAME"
    else
        LATEST_DIR="$SOURCE_BASE/$FOLDER_NAME"
    fi
elif [[ $SOURCE_BASE == *:* ]]; then
    # Remote source - get just the folder name
    REMOTE_HOST="${SOURCE_BASE%%:*}"
    FOLDER_NAME=$(ssh $SSH_OPTS "$REMOTE_HOST" "ls -td ~/svFSGe/partitioned_* 2>/dev/null | head -1 | xargs basename" 2>/dev/null | tr -d '\r')
    LATEST_DIR="$REMOTE_HOST:~/svFSGe/$FOLDER_NAME"
else
    # Local source
    LATEST_DIR=$(ls -td "$SOURCE_BASE"/partitioned_* 2>/dev/null | head -1)
    FOLDER_NAME=$(basename "$LATEST_DIR")
fi

if [ -z "$FOLDER_NAME" ]; then
    echo "Error: No partitioned_* directory found in $SOURCE_BASE"
    exit 1
fi
echo "Found latest folder: $FOLDER_NAME"

# Create destination directory
mkdir -p "$DEST_DIR/$FOLDER_NAME"

# Step 1: Copy everything EXCEPT .vtu and .bin files
echo "Step 1: Copying folder structure and other files (excluding VTU/BIN)..."
if [[ $SOURCE_BASE == *:* ]]; then
    rsync -av -e "ssh $SSH_OPTS" \
        --exclude='pulsatile/*.vtu' \
        --exclude='pulsatile/*.bin' \
        --exclude='steady/*.vtu' \
        --exclude='steady/*.bin' \
        --exclude='gr_restart/*.vtu' \
        "$LATEST_DIR/" "$DEST_DIR/$FOLDER_NAME/"
else
    rsync -av \
        --exclude='pulsatile/*.vtu' \
        --exclude='pulsatile/*.bin' \
        --exclude='steady/*.vtu' \
        --exclude='steady/*.bin' \
        --exclude='gr_restart/*.vtu' \
        "$LATEST_DIR/" "$DEST_DIR/$FOLDER_NAME/"
fi

# Step 2: Copy only last N VTU and BIN files
echo "Step 2: Copying VTU and BIN files (last N timesteps)..."
for dir in pulsatile steady; do
    # Determine how many files to keep for this directory
    if [ "$dir" = "pulsatile" ]; then
        KEEP_LAST_N=$KEEP_LAST_N_PULSATILE
    else
        KEEP_LAST_N=$KEEP_LAST_N_STEADY
    fi

    echo "  Processing $dir (keeping last $KEEP_LAST_N timesteps)..."
    mkdir -p "$DEST_DIR/$FOLDER_NAME/$dir"

    if [[ $SOURCE_BASE == *:* ]]; then
        # Remote: get base names of last N VTU files, then copy both VTU and BIN
        REMOTE_HOST="${SOURCE_BASE%%:*}"
        REMOTE_DIR="~/svFSGe/$FOLDER_NAME/$dir"

        BASENAMES=$(ssh $SSH_OPTS "$REMOTE_HOST" "ls -1v $REMOTE_DIR/*_*.vtu 2>/dev/null | sort -V | tail -$KEEP_LAST_N | xargs -I{} basename {} .vtu" | tr -d '\r')

        echo "$BASENAMES" | while read base; do
            if [ -n "$base" ]; then
                for ext in vtu bin; do
                    rsync -av -e "ssh $SSH_OPTS" "$REMOTE_HOST:$REMOTE_DIR/$base.$ext" "$DEST_DIR/$FOLDER_NAME/$dir/" 2>/dev/null || true
                done
            fi
        done
    else
        # Local: copy last N VTU and BIN files
        BASENAMES=$(ls -1v "$LATEST_DIR/$dir"/*_*.vtu 2>/dev/null | sort -V | tail -$KEEP_LAST_N | xargs -I{} basename {} .vtu)

        echo "$BASENAMES" | while read base; do
            if [ -n "$base" ]; then
                for ext in vtu bin; do
                    [ -f "$LATEST_DIR/$dir/$base.$ext" ] && cp "$LATEST_DIR/$dir/$base.$ext" "$DEST_DIR/$FOLDER_NAME/$dir/"
                done
            fi
        done
    fi
done

echo ""
echo "=== Copy completed successfully! ==="
echo "Folder location: $DEST_DIR/$FOLDER_NAME"

# Clean up SSH master connection
if [[ $SOURCE_BASE == *:* ]]; then
    ssh $SSH_OPTS -O exit "$SSH_HOST" 2>/dev/null || true
fi
