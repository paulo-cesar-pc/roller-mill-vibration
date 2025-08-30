#!/usr/bin/env python3
"""
Minimal test to find bottleneck.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def setup_logging():
    """Set up simple logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def main():
    """Minimal test."""
    logger = setup_logging()
    logger.info("Starting minimal test")
    
    try:
        # 1. Test basic imports
        logger.info("Testing imports...")
        sys.path.append(str(Path(__file__).parent))
        
        logger.info("Importing config...")
        from config.settings import get_config
        config = get_config()
        logger.info(f"Config loaded: {config.project.name}")
        
        logger.info("Importing DataLoader...")
        from src.data.data_loader import DataLoader
        logger.info("DataLoader imported")
        
        logger.info("Creating DataLoader instance...")
        data_loader = DataLoader()
        logger.info("DataLoader created")
        
        logger.info("✅ Minimal test completed!")
        return 0
        
    except Exception as e:
        logger.error(f"Minimal test failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)