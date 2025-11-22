"""
Main runner script for MTS Data Pipeline
"""
import os
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run pipeline
from src.pipeline import main

if __name__ == "__main__":
    main()