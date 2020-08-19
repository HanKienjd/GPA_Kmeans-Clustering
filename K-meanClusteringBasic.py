import numpy as np
import random
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

# Tạo điểm bắt đầu random
np.random.seed(18)

# Lấy ngẫu nhiên 500 điểm cho mỗi clustering với kì vọng 
# means[0], means[1], means[2] và ma trận hiệp phương sai cov

means= [[2, 2], [8, 3], [3, 6]]
cov= [[1, 0], [0, 1]]
N= 500
X0= np.random.multivariate_normal(means[0], cov, N)
X1= np.random.multivariate_normal(means[1], cov, N)
X2= np.random.multivariate_normal(means[2], cov, N)

# Gộp các điểm lại thành một vector cột duy nhất
X=np.concatenate((X0,X1,X2), axis=0)
K=3 #3 cụm(Clusters)

# Hàm khởi tạo ngẫu nhiên các centroids ban đầu
def kmeans_init_centroids(X, k):
    #Chọn ngẫu nhiên k dòng trong ma trận X để tạo các centroids
    return X[np.random.choice(X.shape[0], k, replace=False)]

# Hàm tìm các label mới cho các điểm khi đã cố định các centroids
def kmeas_assign_labels(X, centroids):
    # Tính khoảng cách giữa các điểm trong X và các centroids
    D= cdist(X, centroids)
    # Trả về một ma trận ngang chứa các số 0, 1, 2 là chỉ số của các
    #centroids đầu gần với điểm đang xét nhất
    return np.argmin(D, axis= 1)

# Hàm kiểm tra xem các centroids sau có giống với các centroids trước không
def has_converged(centroids, new_centroids):
    # Trả về giá trị True nếu 2 các centroids sau có giá trị giống các
    #centroids trước
    return (set([tuple(a) for a in centroids])==set([tuple(a) for a in new_centroids]))

# Hàm cập nhật các centroids mới - tập các centroids mới được thêm vào cuối
#danh sách các tập centroids trước nó
def kmeans_update_centroids(X, labels, K):
    # Tạo ma ma trận các centroids
    centroids= np.zeros((K,X.shape[1]))
    for k in range(K):
        # Gom các điểm có cùng label lại
        Xk= X[labels== k,:]
        # Tính trung bình tọa độ giữa các điểm ta thu được tọa độ của
        #các centroids mới
        centroids[k,:]= np.mean(Xk, axis= 0)
    return centroids

#Hàm K- means clustering
def kmeans(X, K):
    centroids=[kmeans_init_centroids(X, K)]
    labels= []
    it= 0 # Số lần vòng lặp thực hiện hay số lần các centroids được xác định lại
    while True:
        # Thêm danh sách label mới vào danh sách dựa trên bộ các centroids 
        #được biết trước đó
        labels.append(kmeas_assign_labels(X, centroids[-1]))
        # Dựa vào bộ label mới tìm bộ các centroids mới
        new_centroids= kmeans_update_centroids(X, labels[-1], K)
        # Kiểm tra bộ các centroids mới với bộ trước nó nếu giống nhau thì
        #kết thúc chương trình nến khác nhau thì cập nhật bộ các centroids mới
        if has_converged(centroids[-1], new_centroids):
            break
        centroids.append(new_centroids)
        it+=1
    return (centroids, labels, it)


#Hàm vẽ ra đồ thị
def kmeans_display(X, label):
    K = np.amax(label) + 1
    # Tách các điểm thộc cùng một clustering thành từng nhóm
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    
    # Tọa độ hóa các điểm và gán màu cho các điểm thuộc cùng cluster
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()

# Thực hiện chương trình(centroids, labels, it) = kmeans(X, K)
(centroids, labels, it) = kmeans(X, K)
print("Centers found by our algorithm:\n", centroids[-1])
print(it)
kmeans_display(X, labels[-1])

