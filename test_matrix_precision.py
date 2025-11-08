#!/usr/bin/env python3
"""
Test matrix operations with NumPy to compare with TypeScript/ml-matrix
"""

import numpy as np

# Test case 1: Simple 2x2 matrix
test_matrix1 = np.array([
    [4.0, 2.0],
    [2.0, 3.0]
])

# Test case 2: Typical Kalman innovation covariance (from actual data)
innovation_cov = np.array([[5.364]])

# Test case 3: Typical state covariance matrix (2x2 for weight + trend)
state_covariance = np.array([
    [0.364, 0.0],
    [0.0, 0.00012]
])

# Test case 4: A matrix that might have precision issues
precision_test = np.array([
    [1.23456789012345, 0.98765432109876],
    [0.98765432109876, 2.34567890123456]
])

print("=== Matrix Inverse Precision Test (Python/NumPy) ===\n")

# Test 1
print("Test 1: Simple 2x2 matrix")
print("Input:")
print(test_matrix1)
inv1 = np.linalg.inv(test_matrix1)
print("Inverse:")
print(inv1)
print("Product (should be identity):")
product1 = test_matrix1 @ inv1
print(product1)
print()

# Test 2
print("Test 2: Kalman innovation covariance (1x1)")
print("Input:")
print(innovation_cov)
inv2 = np.linalg.inv(innovation_cov)
print("Inverse:")
print(inv2)
print("Product (should be identity):")
product2 = innovation_cov @ inv2
print(product2)
print()

# Test 3
print("Test 3: State covariance matrix (2x2)")
print("Input:")
print(state_covariance)
inv3 = np.linalg.inv(state_covariance)
print("Inverse:")
print(inv3)
print("Product (should be identity):")
product3 = state_covariance @ inv3
print(product3)
print()

# Test 4
print("Test 4: High precision matrix")
print("Input:")
print(precision_test)
inv4 = np.linalg.inv(precision_test)
print("Inverse:")
print(inv4)
print("Product (should be identity):")
product4 = precision_test @ inv4
print(product4)
print()

# Test 5: Full Kalman update step
print("Test 5: Full Kalman Update Step")
H = np.array([[1, 0]])  # Observation matrix
R = np.array([[5.0]])    # Observation noise
P_pred = np.array([      # Predicted covariance
    [0.382, 0.0],
    [0.0, 0.00012]
])

print("H * P * H^T + R (innovation covariance):")
S = H @ P_pred @ H.T + R
print(S)

print("Innovation covariance inverse:")
S_inv = np.linalg.inv(S)
print(S_inv)

print("Kalman Gain = P * H^T * S^{-1}:")
K = P_pred @ H.T @ S_inv
print(K)
print()

# Output in a format that can be compared with TypeScript
print("=== Raw numerical values for comparison ===")
print("Test 2 (1x1 innovation covariance):")
print(f"  Input: {innovation_cov[0, 0]}")
print(f"  Inverse: {inv2[0, 0]}")
print(f"  Product: {product2[0, 0]}")

print("\nTest 5 (Kalman gain):")
print(f"  K[0,0] = {K[0, 0]}")
print(f"  K[1,0] = {K[1, 0]}")
