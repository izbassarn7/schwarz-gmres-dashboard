#!/bin/bash
set -euo pipefail

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

BASE_URL="https://suitesparse-collection-website.herokuapp.com/MM"

declare -A MATRICES=(
    ["bcsstk01"]="HB/bcsstk01"
    ["nos1"]="HB/nos1"
    ["thermal1"]="Schmid/thermal1"
    ["thermal2"]="Schmid/thermal2"
    ["apache2"]="GHS_psdef/apache2"
)

for name in "${!MATRICES[@]}"; do
    path="${MATRICES[$name]}"
    if [ -f "$DATA_DIR/${name}.mtx" ]; then
        echo "Already exists: ${name}.mtx"
        continue
    fi
    echo "Downloading ${name}..."
    wget -q "${BASE_URL}/${path}.tar.gz" -O "$DATA_DIR/${name}.tar.gz"
    tar -xzf "$DATA_DIR/${name}.tar.gz" -C "$DATA_DIR"
    mv "$DATA_DIR/${name}/${name}.mtx" "$DATA_DIR/${name}.mtx"
    rm -rf "$DATA_DIR/${name}" "$DATA_DIR/${name}.tar.gz"
    echo "  Done: $DATA_DIR/${name}.mtx"
done

echo "All matrices downloaded."
