#!/usr/bin/env python3
"""
Test numerical precision differences between ml-matrix and numpy
"""

import numpy as np

# Test data: 2x2 innovation covariance matrix (typical values from Kalman filter)
test_matrices = [
    # Small values (typical for weight measurements)
    [[1.0, 0.1], [0.1, 1.0]],
    [[0.5, 0.05], [0.05, 0.5]],
    [[2.0, 0.3], [0.3, 2.0]],

    # Values from actual Kalman filtering
    [[1.234, 0.123], [0.123, 1.234]],
    [[0.789, 0.056], [0.056, 0.789]],
]

print('Testing Matrix Inversion Precision (numpy)\n')
print('=' * 60)

for matrix_data in test_matrices:
    m = np.array(matrix_data)
    print('\nOriginal Matrix:')
    print(m)

    inv = np.linalg.inv(m)
    print('Inverse:')
    print(inv)

    # Verify: A * A^-1 should equal identity matrix
    product = m @ inv
    print('Verification (A * A^-1):')
    print(product)

    # Check how close to identity
    identity = np.eye(2)
    diff = product - identity
    max_error = np.max(np.abs(diff))
    print(f'Max error from identity: {max_error:.4e}')
