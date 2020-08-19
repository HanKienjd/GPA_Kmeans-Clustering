import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sys

from sklearn.cluster import KMeans

#import bức ảnh
img = mpimg.imread("girl3.jpg")

#Tọa độ hóa các pixel của bức ảnh
plt.imshow(img)
imgplot = plt.imshow(img)
plt.axis("off")

#Hiển thị ảnh gốc
plt.show()

#Chồng các pixel của bức ảnh lên nhau tạo thành ma trận 
X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))

K=2 # Số lượng cluster mong muốn

kmeans = KMeans(n_clusters=K).fit(X)#Phân cụm cho các điểm ảnh

label = kmeans.predict(X)#Trả về nhãn cho từn điểm ảnh

img4 = np.zeros_like(X)#Tạo ra ma trận các pixel của một bức ảnh trắng

#Thay đổi màu sắc của các pixel trong cùng một cluster thành màu của cetroids tương ứng với cluster đó
#và gán các pixel đó cho bức ảnh img4
for k in range(K):
    img4[label == k] = kmeans.cluster_centers_[k]

#Tọa độ hóa lại các pixel của bức ảnh và in bức ảnh ra
img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2]))
plt.imshow(img5, interpolation='nearest')
plt.axis('off')
plt.show()