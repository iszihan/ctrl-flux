#!/bin/bash
# Extract all tar files from laion_wds_512 to a flat folder structure

SRC_DIR="./dataset/laion2B/laion_wds_512"
TRAIN_DIR="./dataset/laion2B/extracted_train"
TEST_DIR="./dataset/laion2B/extracted_test"

# Create output directories
mkdir -p "$TRAIN_DIR"
mkdir -p "$TEST_DIR"

echo "Extracting tar files..."
echo "Train shards: 00001-00099"
echo "Test shards: 00000-00001"

# Extract test shards (00000, 00001)
for shard in 00000 00001; do
    tar_file="$SRC_DIR/${shard}.tar"
    if [ -f "$tar_file" ]; then
        echo "Extracting $tar_file to $TEST_DIR..."
        tar -xf "$tar_file" -C "$TEST_DIR"
    fi
done

# Extract train shards (00002-00099, skipping test)
for tar_file in "$SRC_DIR"/*.tar; do
    shard=$(basename "$tar_file" .tar)
    # Skip test shards
    if [ "$shard" = "00000" ] || [ "$shard" = "00001" ]; then
        continue
    fi
    echo "Extracting $tar_file to $TRAIN_DIR..."
    tar -xf "$tar_file" -C "$TRAIN_DIR"
done

echo ""
echo "Done! Extracted files:"
echo "  Train: $(ls -1 "$TRAIN_DIR"/*.jpg 2>/dev/null | wc -l) images in $TRAIN_DIR"
echo "  Test:  $(ls -1 "$TEST_DIR"/*.jpg 2>/dev/null | wc -l) images in $TEST_DIR"
