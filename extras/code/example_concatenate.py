# import numpy as np

# # Two 2D arrays
# a = np.array([[1, 2], [3, 4]])
# b = np.array([[5, 6], [7, 8]])

# print(a.shape)
# print(b.shape)

# # Concatenate along axis=0 (rows)
# concat_axis0 = np.concatenate((a, b), axis=0)
# print(concat_axis0.shape)


# # Concatenate along axis=1 (columns)
# concat_axis1 = np.concatenate((a, b), axis=1)
# print(concat_axis1.shape)

# stack_axis0 = np.stack((a, b), axis=0)
# print(stack_axis0.shape)  # (2, 2, 2)

# stack_axis1 = np.stack((a, b), axis=1)
# print(stack_axis1.shape)  # (2, 2, 2)


# import numpy as np

# # Create a list of 5 arrays with shape (1,2,2,2)
# array_list = [np.random.rand(1,2,2,2) for _ in range(5)]

# # Convert list to array for visualization
# print("Each array shape:", array_list[0].shape)  # Should be (1,2,2,2)

# # Concatenate along axis=0 (expanding first dimension)
# concatenated_array = np.concatenate(array_list, axis=0)

# # Check the shape of the concatenated array
# print("Concatenated shape:", concatenated_array.shape)

# from joblib import Parallel, delayed
# import time
# import numpy as np

# def process_item(x):
#     start_time = time.time()  # Track when the task starts
#     delay = np.random.uniform(0.5, 2.0)  # Random delay between 0.5 and 2 seconds
#     time.sleep(delay)  # Simulate processing time
#     end_time = time.time()  # Track when the task completes
#     return x**2, x, delay, start_time, end_time

# # List of inputs
# inputs = [1, 2, 3, 4, 5]

# # Run in parallel with 3 workers
# results = Parallel(n_jobs=3)(delayed(process_item)(x) for x in inputs)

# # Sort results by task completion time
# sorted_by_completion = sorted(results, key=lambda r: r[4])  # Sort by end_time

# # Print the results
# print("\n=== Task Completion Order (Based on End Time) ===")
# print("Finished Order | Input | Squared | Delay (s) | Start Time | End Time")
# for i, result in enumerate(sorted_by_completion, 1):
#     print(f"{i:<14} | {result[1]:<5} | {result[0]:<7} | {result[2]:.2f} s | {result[3]:.2f}s | {result[4]:.2f}s")

# print("\n=== Final Order in Results (Returned by Joblib) ===")
# print([r[0] for r in results])  # Extract just the squared values

from enum import Enum


class OverlapMetrics(Enum):
    DICE_SCORE = ("Dice Score", 0)
    VOLUME_SIMILARITY_ITK = ("Volume Similarity (ITK)", 1)
    VOLUME_SIMILARITY = ("Volume Similarity", 2)

    def __init__(self, label, value):
        self.label = label   # Store the human-readable label
        # Explicitly set the numerical value

    def __str__(self):
        return self.label  # Make print(metric) return the label

# Example Usage
metric = OverlapMetrics.DICE_SCORE
print(metric.label)  # "Dice Score"
print(metric._value_)  # 0 (correct numerical value, not the full tuple)
print(metric.name)   # "DICE_SCORE"
print(str(metric))   # "Dice Score" (because of __str__)


# Example usage:
# print(type(OverlapMetrics.DICE_SCORE))          # Dice Score
# print(type(OverlapMetrics.DICE_SCORE.label))   # Dice Score
# print(OverlapMetrics.DICE_SCORE.label == OverlapMetrics.DICE_SCORE) 
# print(OverlapMetrics.VOLUME_SIMILARITY.label)  # Volume Similarity
# print(OverlapMetrics.DICE_SCORE.value)    # 0
# p = OverlapMetrics("a")
# print(p)

# metric = OverlapMetrics.DICE_SCORE
# print(metric)  # Dice Score (because of __str__)
# print(metric.label)  # Dice Score (custom label)
# print(metric.value)  # 0 (numerical value)





