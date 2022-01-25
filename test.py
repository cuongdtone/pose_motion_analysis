import cv2

image_2 = cv2.imread('samples/video1_clip1.png') #red
image_1 = cv2.imread('samples/video_1.png')
image_1 = cv2.resize(image_1, (1280, 720))
image_2 = cv2.resize(image_2, (1280, 720))
h, w = image_1.shape[:2]
for x in range(w):
    for y in range(h):
        if image_1[y, x, 0]==255 and image_1[y, x, 1] ==0:
            image_2[y, x, :] = [255,0,0]

cv2.imshow('image', image_2)
cv2.imwrite('image.png', image_2)
cv2.waitKey()