#!/usr/bin/env bash
set -e

echo "=== Setting up Ollama local LLM server ==="

# Check if ollama is installed, install if not
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing Ollama via curl..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama is already installed."
fi

# Start Ollama service in background if not running
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "Starting Ollama background service..."
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
else
    echo "Ollama service is already running."
fi

# Pull required models
echo "Pulling primary model llama3.1:8b..."
ollama pull llama3.1:8b || echo "Failed to pull llama3.1:8b"

echo "Pulling fallback model phi3:mini..."
ollama pull phi3:mini || echo "Failed to pull phi3:mini"

echo "Pulling embedding model nomic-embed-text..."
ollama pull nomic-embed-text || echo "Failed to pull nomic-embed-text"

echo "=== Ollama setup complete ==="
