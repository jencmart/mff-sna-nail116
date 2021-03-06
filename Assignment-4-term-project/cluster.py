import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# elbow at 2 and 4
x = [[5,5,5],
[5,5,5],
[5,5,5],
[5,5,5],
[5,5,5],
[5,5,5],
[5,5,5],
[5,5,5],
[5,5,5],
[5,5,5],
[5,5,5],
[5,5,5],
[5,5,5],
[5,5,5],
[5,5,5],
[5,5,5],
[5,4.99,5],
[5,5,4.9],
[5,5,4.9],
[5,5,4.9],
[5,4.9,5],
[5,5,4.8],
[5,5,4.8],
[5,4.5,5],
[4.7,4.7,5],
[4.7,4.7,4.8],
[5,5,4],
[5,4,5],
[4,5,5],
[5,4,5],
[4.7,5,4.2],
[4.5,4.5,4.7],
[4.7,4.5,4.2],
[4.5,4,4.8],
[5,5,3.2],
[4.2,4,5],
[4,4,5],
[4,4,5],
[4,4,5],
[4,4,5],
[4,4,4.8],
[4,4.2,4.5],
[4.5,3.93,4.2],
[4,4.5,4.1],
[5,5,2.5],
[4,4.5,3.9],
[4.5,4.5,3.2],
[4,3.7,4.3],
[4,4.1,3.6],
[2.5,3.7,5],
[4,4,3.1],
[3.5,3.7,3.9],
[4,2,5],
[4,4,3],
[3.4,3.5,4],
[4.3,4,2.5],
[3.2,2.5,5],
[3,2.8,4.8],
[3,4,3],
[3.5,2.5,3.8],
[3,3.5,3.3],
[3,2,4.5],
[2,4.5,3],
[3,2.5,4],
[5,0,4.1],
[3.5,3,2.6],
[3,2,4],
[3,3,3],
[4,3.7,1.2],
[4,4,0.4],
[4.3,4,0],
[2,3,3.2],
[3,0,5],
[2,1,5],
[3,2.5,2.5],
[0.9,1.5,5],
[3,1,3.4],
[3,2.5,1.6],
[2,2,3],
[3,2.5,1.4],
[3,2,1.8],
[2.5,2,2.1],
[4,0,2.5],
[3,3,0.3],
[2,1,3],
[0,0.3,5],
[0.2,0,5],
[0,0,5],
[1,4,0],
[1,2.5,1],
[1,1,2.3],
[1,2.5,0.3],
[3.3,0,0],
[1,1.3,0],
[2,0,0],
[0.3,0.5,1],
[0.5,0.4,0.9],
[0.5,1,0.1],
[1,0.3,0],
[0,-0.2,1.1],
[0,0,0],
[0,0,0],]
if __name__ == '__main__':

    x = np.asarray(x)
    # WCSS = []
    # for i in range(1,11):
    #     model = KMeans(n_clusters = i,init = 'k-means++')
    #     model.fit(x)
    #     WCSS.append(model.inertia_)
    # fig = plt.figure(figsize = (7,7))
    # plt.plot(range(1,11),WCSS, linewidth=4, markersize=12,marker='o',color = 'green')
    # plt.xticks(np.arange(11))
    # plt.xlabel("Number of clusters")
    # plt.ylabel("WCSS")
    # plt.show()

    model = KMeans(n_clusters=5, init="k-means++", max_iter=300, n_init=10, random_state=0)
    y_clusters = model.fit_predict(x)

    # 3d scatterplot using matplotlib

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[y_clusters == 0, 0], x[y_clusters == 0, 1], x[y_clusters == 0, 2], s=140, color='blue',
               label="cluster 0")
    ax.scatter(x[y_clusters == 1, 0], x[y_clusters == 1, 1], x[y_clusters == 1, 2], s=140, color='orange',
               label="cluster 1")
    ax.scatter(x[y_clusters == 2, 0], x[y_clusters == 2, 1], x[y_clusters == 2, 2], s=140, color='green',
               label="cluster 2")
    ax.scatter(x[y_clusters == 3, 0], x[y_clusters == 3, 1], x[y_clusters == 3, 2], s=140, color='#D12B60',
               label="cluster 3")
    ax.scatter(x[y_clusters == 4, 0], x[y_clusters == 4, 1], x[y_clusters == 4, 2], s=140, color='purple',
               label="cluster 4")
    ax.set_xlabel('Burtak', fontsize=30)
    ax.set_ylabel('Magneto', fontsize=30)
    ax.set_zlabel('Sochar', fontsize=30)
    ax.legend(prop={'size': 26})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.zticks(fontsize=14)
    # plt.xlabel("Burtak")
    # plt.ylabel("Magneto")
    # # plt.zlabel("Sochar")
    plt.show()
    for i in y_clusters:
        print(i)