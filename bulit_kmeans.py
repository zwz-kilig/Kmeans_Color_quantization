import numpy as np
from skimage import io
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib as mpl

image = io.imread("jacky2.png")  # 导入图片
rows = image.shape[0]            # 返回图片的行数
cols = image.shape[1]            # 返回图片的列数

image2 = image.reshape(image.shape[0] * image.shape[1], 3)                  # 把图片拉直（每条记录相当图像上一个点，有个变量代表三原色）
t = 1
for k in [2, 4, 8, 16, 32, 64, 128, 256]:
    print('正在执行第%d轮' % t)
    kmeans = KMeans(n_clusters=k, n_init=10, max_iter=200)                  # 使用kmeans方法聚类（n_clusters为簇数目）
    kmeans.fit(image2)

    clusters = np.asarray(kmeans.cluster_centers_, dtype=np.uint8)           # 获取所得簇的中心点
    labels = np.asarray(kmeans.labels_, dtype=np.uint8)                      # 获取每条记录即（图上每个点）的类标签
    labels = labels.reshape(rows, cols)                                      # 把矩阵还原成原来大小

    image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)  # 创造一个全是0的原来图片大小矩阵保存数据
    for i in range(labels.shape[0]):                                         # 用聚类后类标签所代表的数值替换原有的数值
        for j in range(labels.shape[1]):                                     # 画图展示
            image[i, j, :] = clusters[labels[i, j], :]
    #io.imsave('kmeans_jacky2_clu%d.png' % k, image)
    plt.subplot(1, 4, t)
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.axis('off')
    plt.title(u'Kmeans簇数目：%d' % k)
    io.imshow(image)
    t += 1
io.show()
