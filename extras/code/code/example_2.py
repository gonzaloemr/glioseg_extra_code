import numpy as np


# Example masks
mask1 = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
mask2 = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]])

# Find the region where mask1 is 1 but mask2 is not
result = np.logical_and(mask1,np.logical_not(mask2)).astype(int)  # Bitwise AND with NOT

# Print the result
print("Mask 1:")
print(mask1)
print("Mask 2:")
print(mask2)
print("Result (Mask1 AND NOT Mask2):")
print(result)


import numpy as np


# Example mask array
mask1 = np.array([1, 2, 1, 3, 1])

# Label (scalar)
label = 1

# Element-wise comparison
result = (mask1 == label)

print(result)
