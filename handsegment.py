import numpy as np
import cv2
boundaries = [
    ([0, 120, 0], [140, 255, 100]),
    ([25, 0, 75], [180, 38, 255])
]


def handsegment(frame):
    lower, upper = boundaries[0]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask1 = cv2.inRange(frame, lower, upper)

    lower, upper = boundaries[1]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask2 = cv2.inRange(frame, lower, upper)

    # for i,(lower, upper) in enumerate(boundaries):
    # 	# create NumPy arrays from the boundaries
    # 	lower = np.array(lower, dtype = "uint8")
    # 	upper = np.array(upper, dtype = "uint8")

    # 	# find the colors within the specified boundaries and apply
    # 	# the mask
    # 	if(i==0):
    # 		print "Harish"
    # 		mask1 = cv2.inRange(frame, lower, upper)
    # 	else:
    # 		print "Aadi"
    # 		mask2 = cv2.inRange(frame, lower, upper)
    mask = cv2.bitwise_or(mask1, mask2)
    output = cv2.bitwise_and(frame, frame, mask=mask)
    # show the images
    # cv2.imshow("images", mask)
    # cv2.imshow("images", output)
    return output

if __name__ == '__main__':
    frame = cv2.imread("test.jpeg")
    handsegment(frame)
