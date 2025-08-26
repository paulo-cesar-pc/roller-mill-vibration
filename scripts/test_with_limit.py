#!/usr/bin/env python3
"""
Quick test script with data limited to 10,000 rows.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from config.settings import get_config

def main():
    """Test with limited data using testing mode."""
    print("Testing with data in testing mode...")
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Load and process data in testing mode
    df, quality_report = data_loader.load_and_process(testing_mode=True)
    
    print(f"Final data shape: {df.shape}")
    print(f"Data quality score: {quality_report.quality_score:.1f}/100")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)