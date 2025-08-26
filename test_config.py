#!/usr/bin/env python3
"""Test configuration loading."""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

try:
    from config.settings import get_config
    
    print("Testing configuration loading...")
    config = get_config()
    
    print(f"Project: {config.project.name} v{config.project.version}")
    print(f"Target column: {config.data.target_column}")
    print(f"Testing enabled: {config.data.testing.enabled}")
    print(f"Testing max rows: {config.data.testing.max_rows}")
    
    print("✅ Configuration loaded successfully!")
    
except Exception as e:
    print(f"❌ Configuration failed: {e}")
    import traceback
    traceback.print_exc()