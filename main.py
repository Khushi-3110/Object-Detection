import cv2
import numpy as np
img=cv2.imread('blue.jpg')

lower_range = np.array ([0,0,0]) # Example values (adjust based on the specific color range you want)
upper_range = np.array([14, 255, 255])

hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
x=np.array(hsv[:,:,0])
#print("hue",x)
mask=cv2.inRange(hsv,lower_range,upper_range)
_,mask1=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
cv2.imshow('o/p',mask)
#print("mask",mask)
cv2.imshow('image',img)

lower_white = np.array([0, 0, 00])
upper_white = np.array([180, 255, 255])
print("lw",lower_white)
# Create a binary mask for white color
white_mask = cv2.inRange(hsv, lower_white, upper_white)

# Find contours in the binary mask
contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through contours
for contour in contours:
    area = cv2.contourArea(contour)
    n=len(contours)
    print(area)

    # Print the image if the area is above a certain threshold
    if area >= 1000:  # Adjust the threshold as needed
        # Create a blank image
        print("Object detected")

    else:
        print("Object not detected")

cv2.waitKey(0)
cv2.destroyAllWindows()