# fastKmeans
# Multithread Kmeans and accelarate by Ameans and AFKMC2 seeding algorithom!
# 目前最快速Kmeans算法，并由java实现！
# 1.这是一个由java实现的的，多线程Kmeans聚类算法；
# 2.在聚类的选种阶段实现了Kmeans++算法和NIPS 2016的文章“Fast and Probably Good Seedings for k-Means”中提出了AFK-MC²算法，该算法改进了k-Means算法中初始种子点的生成方式，使其聚类速度相较于目前最好的k-Means++方式提高了好几个数量级。
3.在迭代计算加速方面实现了2018年的一篇文章中描述的迭代加速算法Ameans：A-means improving the cluster assignment phase of k-means for Big Data；
上述两篇论文在项目的doc文件夹中。
4.如果你的项目中的向量是稀疏向量，并且有值的位置可以提前保存到List中，那么可以将这个List纳入聚类计算中，在进行两个向量间的距离计算时，只计算有值位置的距离。
5.实现起来不容易，如果觉得有用麻烦点个赞。
