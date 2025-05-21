#!/bin/bash

module load rclone/1.63.1

INCLUDE_PATTERNS=(
  "--include" "OTHERS/**"
  "--include" "src/archived/**"
  "--include" "src/ucvme/**"
  "--include" "data/**"
  "--include" ".declare_api_key.sh"
  "--include" "analysis-and-visualisation.ipynb"
  "--include" "notes.md"
)

echo "Select copy direction:"
echo "1. Copy from local to remote"
echo "2. Copy from remote to local"
read -p "Enter your choice (1/2): " choice

case $choice in
  1)
    echo "Copying from local to remote..."
    rclone -v copy ./ pawsey1001_rakib:noisempathy/ "${INCLUDE_PATTERNS[@]}"
    ;;
  2)
    echo "Copying from remote to local..."
    rclone -v --local-no-set-modtime copy pawsey1001_rakib:noisempathy/ ./ "${INCLUDE_PATTERNS[@]}"
    ;;
  *)
    echo "Invalid choice."
    exit 1
    ;;
esac