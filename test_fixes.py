#!/usr/bin/env python3
"""
Test script to verify fixes for the training pipeline.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

from config.settings import get_config
from src.data.data_loader import DataLoader

def test_data_loading():
    """Test data loading with CM2_PV_VRM01_VIBRATION1 exclusion."""
    try:
        # Load configuration
        config = get_config()
        print(f"✅ Configuration loaded: {config.project.name}")
        
        # Test data loading
        data_loader = DataLoader()
        df, quality_report = data_loader.load_and_process(testing_mode=True)
        
        print(f"✅ Data loaded successfully: {df.shape}")
        print(f"✅ Data quality score: {quality_report.quality_score:.1f}/100")
        
        # Check that CM2_PV_VRM01_VIBRATION1 is not in the data
        if 'CM2_PV_VRM01_VIBRATION1' in df.columns:
            print("❌ CM2_PV_VRM01_VIBRATION1 still present in data!")
            return False
        else:
            print("✅ CM2_PV_VRM01_VIBRATION1 successfully removed from data")
        
        # Test matplotlib
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_title('Test Plot')
        plt.savefig('test_plot.png', dpi=150)
        plt.close()
        print("✅ Matplotlib backend working correctly")
        
        # Clean up
        Path('test_plot.png').unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing fixes for training pipeline...")
    success = test_data_loading()
    if success:
        print("\n🎉 All tests passed! Training pipeline fixes are working.")
    else:
        print("\n💥 Some tests failed. Need to investigate further.")