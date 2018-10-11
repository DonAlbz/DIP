import cv2
import numpy as np
import matplotlib.pyplot as plt


def salt_pepper(image, soglia):
    m=len(image)
    n=len(image[0])
    size=image.shape
    if len(size)==2:
        for i in range(0,m):
            for j in range(0,n):
                p=np.random.rand()
                if p < soglia:
                    image[i][j]=0
                elif p > (1-soglia):
                    image[i][j]=255
        return image
    else:
        for i in range(0,m):
            for j in range(0,n):
                p=np.random.rand()
                if p < soglia:
                    image[i][j][:]=0
                elif p > (1-soglia):
                    image[i][j][:]=255
        return image


img = cv2.imread('img1.jpg', 1)
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = salt_pepper(img, 0.03)
plt.imshow(img2, cmap='gray')
plt.show()
