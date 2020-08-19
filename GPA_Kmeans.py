import csv
import numpy as np
import random
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import random

np.random.seed(18)

with open('/home/dev/Python/K-meansClusteringBasic/diem.csv') as csv_file:
    csv_reader= csv.reader(csv_file, delimiter=',')
    line_count=0
    myList=[]
    for row in csv_reader:
        if line_count==0:
            line_count+=1
        else:
            myList.append(row[1])
            line_count+=1
a=np.array(myList)
a=a.reshape(-1,1)
a=a.astype(np.float)

k=4 # cụm(Clusters)

# Hàm khởi tạo ngẫu nhiên các centroids ban đầu
def kmeans_init_centroids(X, k):
    #Chọn ngẫu nhiên k điểm trong ma trận X để tạo các centroids
    return X[np.random.choice(X.shape[0], k, replace=False)]

# Hàm tìm các label mới cho các điểm khi đã cố định các centroids
def kmeans_assign_labels(X, centroids):
    # Tính khoảng cách Euclid giữa các điểm trong X và các centroids
    D= cdist(X, centroids)
    # Trả về một ma trận ngang chứa các số 0, 1, 2 là chỉ số của các
    #centroids đầu gần với điểm đang xét nhất
    return np.argmin(D, axis= 1)

def kmeans_update_centroids(X, labels, K):
    # Tạo ma ma trận các centroids
    centroids= np.zeros((K,X.shape[1]))
    #Tinh gia trị của các centroid dựa và ma trận label
    for k in range(K):
        # Gom các điểm có cùng label lại
        Xk= X[labels== k,:]
        # Tính trung bình tọa độ giữa các điểm ta thu được tọa độ của
        #các centroids mới
        centroids[k,:]= np.mean(Xk, axis= 0)
    return centroids

# Hàm kiểm tra xem các centroids sau có giống với các centroids trước không
def has_converged(centroids, new_centroids):
    # Trả về giá trị True nếu 2 các centroids sau có giá trị giống các
    #centroids trước
    return (set([tuple(a) for a in centroids])==set([tuple(a) for a in new_centroids]))


#Hàm K- means clustering
def kmeans(X, K):
    #Khởi tạo ngẫu nhiên K centroid ban đầu
    centroids=[kmeans_init_centroids(X, K)]
    labels= []
    it= 0 # Số lần vòng lặp thực hiện hay số lần các centroids được xác định lại
    while True:
        # Thêm danh sách label mới vào danh sách dựa trên bộ các centroids 
        #được biết trước đó
        labels.append(kmeans_assign_labels(X, centroids[-1]))
        # Dựa vào bộ label mới tìm bộ các centroids mới
        new_centroids= kmeans_update_centroids(X, labels[-1], K)
        # Kiểm tra bộ các centroids mới với bộ centroid trước nó nếu giống nhau thì
        #kết thúc chương trình nến khác nhau thì cập nhật bộ các centroids mới
        if has_converged(centroids[-1], new_centroids):
            break
        centroids.append(new_centroids)
        it+=1
    return (centroids, labels, it)


# Thực hiện chương trình(centroids, labels, it) = kmeans(X, K)
(centroids, labels, it) = kmeans(a, k)

#Đếm số lượng phần tử thuộc mỗi centroid
arr= np.squeeze(np.asarray(labels[-1]))
arrLabelNumber=[]
number=0
for i in range(k):
    for label in arr:
        if label==i:
            number+=1
    arrLabelNumber.append(number)
    number=0

#Lấy giá trị của các centroid
centroids_value= np.squeeze(np.asarray(centroids[-1]))

for i in range(k-1):
    for j in range(0, k-i-1):
        if arrLabelNumber[j] > arrLabelNumber[j+1] : 
                arrLabelNumber[j], arrLabelNumber[j+1] = arrLabelNumber[j+1], arrLabelNumber[j]
                centroids_value[j], centroids_value[j+1] = centroids_value[j+1], centroids_value[j] 


print(centroids_value)
print(arrLabelNumber)
plt.plot(arrLabelNumber,centroids_value,color='green', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='blue', markersize=12)
plt.xlabel("Số lượng sinh viên")
plt.ylabel("Điểm trung bình")
plt.title("Thống kê điểm trung bình năm nhất")

#Tọa độ trục
plt.grid(True)
plt.ylim(1,4)
xstick=[]
ystick=[]
i=0
while(i<4):
    ystick.append(i)
    i+=0.25
i=0
while(i<100):
    xstick.append(i)
    i+=1
plt.yticks(ystick)
plt.show()