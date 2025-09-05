#!/bin/bash

# CareGuard AI - Quick Start Script
# This script sets up and runs the complete CareGuard AI system

echo "🏥 CareGuard AI - Quick Start"
echo "============================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else  
    source .venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p models logs data/raw data/processed

# Generate synthetic data
echo "Generating synthetic patient data..."
cd src
python synth_data.py

# Train the model
echo "Training risk prediction model..."
python train.py

# Test explainability
echo "Testing model explainability..."
python explain.py

# Return to root directory
cd ..

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Launch dashboard: streamlit run app/streamlit_app.py"  
echo "2. Start API server: uvicorn src.api:app --reload --port 8000"
echo "3. Open Jupyter notebook: jupyter notebook notebooks/01_explore_and_features.ipynb"
echo ""
echo "🚀 CareGuard AI is ready for use!"
