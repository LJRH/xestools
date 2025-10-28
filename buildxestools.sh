#!/bin/bash
# Clone (if needed), update, and build a Singularity container from a GitHub repo
# using /dls/tmp/<username>_i20xes_tmp if available, otherwise ./tmp

set -e  # Exit on error

# === Configuration ===
REPO_URL="git@github.com:LJRH/i20xes.git"
REPO_DIR="./i20xes"
CONTAINER_NAME="xestools.sif"
BRANCH="main"  # Change if needed

echo "🚀 Starting update and build process..."

# === Clone or update repo ===
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "📦 Cloning repository..."
    git clone "$REPO_URL" "$REPO_DIR"
else
    echo "🔄 Repository exists — pulling latest changes..."
    cd "$REPO_DIR"
    git fetch origin "$BRANCH"
    git reset --hard "origin/$BRANCH"
fi

cd "$REPO_DIR"

# === Check for Singularity recipe ===
if [ ! -f Singularity ]; then
    echo "❌ No Singularity file found in $REPO_DIR"
    exit 1
fi

# === Temporary directory setup ===
USER_TMP_DIR=""
if [ -d /dls/tmp ]; then
    USER_TMP_DIR="/dls/tmp/${USER}_i20xes_tmp"
else
    USER_TMP_DIR="./tmp"
fi

mkdir -p "$USER_TMP_DIR"
export TMPDIR="$USER_TMP_DIR"

echo "🧰 Using temporary build directory: $TMPDIR"

# === Build the container ===
echo "🏗️  Building Singularity container..."
singularity build --tmpdir "$TMPDIR" "$CONTAINER_NAME" Singularity

# === Cleanup ===
echo "🧹 Cleaning up temporary directory..."
rm -rf "$TMPDIR"

echo "✅ Build complete: $(pwd)/$CONTAINER_NAME"

