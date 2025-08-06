#!/bin/bash
set -e

echo "📦 Starting dependency installation..."

for i in {1..3}; do
  echo "🔁 Attempt $i: Installing from requirements.lock.txt..."
  if pip install --no-cache-dir -r /app/requirements.lock.txt; then
    echo "✅ Dependencies installed successfully on attempt $i."
    break
  else
    echo "⚠️ Install failed on attempt $i. Retrying in 5 seconds..."
    sleep 5
  fi
done

echo "🏁 Installation script completed."
