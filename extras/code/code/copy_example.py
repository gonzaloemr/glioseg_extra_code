# import copy
# li1 = [1, 2, [3,5], 4]
# li2 = copy.deepcopy(li1)
# print ("The original elements before deep copying")
# for i in range(0,len(li1)):
#     print (li1[i],end=" ")

# print("\r")
# li2[2][0] = 7
# print ("The new list of elements after deep copying ")
# for i in range(0,len( li1)):
#     print (li2[i],end=" ")

# print("\r")
# print ("The original elements after deep copying")
# for i in range(0,len( li1)):
#     print (li1[i],end=" ")

# import copy
# li1 = [1, 2, [3,5], 4]
# li2 = copy.copy(li1)
# print ("The original elements before shallow copying")
# for i in range(0,len(li1)):
#     print (li1[i],end=" ")

# print("\r")
# li2[2][0] = 7
# print ("The original elements after shallow copying")
# for i in range(0,len( li1)):
#     print (li1[i],end=" ")

# a= [1,2,3]
# print(a)
# a = a[::-1]
# print(a)


# def rotate_image_on_axis(image, angle, rot_axis):
#     return np.swapaxes(rotateImage(np.swapaxes(image,2,rot_axis),angle,cv2.INTER_LINEAR)
#                                  ,2,rot_axis)

# def rotate_stack(stack, angle, rot_axis):
#     images = []
#     for idx in range(stack.shape[0]):
#         images.append(rotate_image_on_axis(stack[idx], angle, rot_axis))
#     return np.stack(images, axis=0)

import numpy as np


def mirror_and_modify(arr):
    mirrored = arr[::-1]  # Creates a view, not a copy
    mirrored[0] = 999     # Modifies the view, affecting the original
    return mirrored

original = np.array([1, 2, 3])
mirrored = mirror_and_modify(original)
print("Original:", original)  # Output: [999, 2, 3]
print("Mirrored:", mirrored)  # Output: [999, 2, 1]