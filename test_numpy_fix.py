#!/usr/bin/env python3
"""
Test script to verify the numpy array formatting fix.
"""

import pandas as pd
import numpy as np

# Simulate the feature importance DataFrame that was causing issues
feature_importance = pd.DataFrame({
    'feature': ['feature1', 'feature2', 'feature3'],
    'mean': [np.array([0.1234]), np.array([0.5678]), np.array([0.9012])],  # numpy arrays
    'std': [np.array([0.0123]), np.array([0.0456]), np.array([0.0789])]   # numpy arrays
})

print("Testing the numpy array formatting fix...")

# Test the original problematic code
try:
    for idx, row in feature_importance.head(10).iterrows():
        print(f"Original (broken): {row['feature']}: {row['mean']:.4f} ± {row['std']:.4f}")
except Exception as e:
    print(f"❌ Original code fails as expected: {e}")

# Test the fixed code
print("\nTesting fixed code:")
try:
    for idx, row in feature_importance.head(10).iterrows():
        mean_val = float(row['mean']) if hasattr(row['mean'], '__float__') else row['mean']
        std_val = float(row['std']) if hasattr(row['std'], '__float__') else row['std']
        print(f"✅ Fixed: {row['feature']}: {mean_val:.4f} ± {std_val:.4f}")
    print("🎉 Fix works correctly!")
except Exception as e:
    print(f"❌ Fixed code still fails: {e}")