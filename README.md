# fastKmeans
# Multithread Kmeans and accelarate by Ameans and AFKMC2 seeding algorithom!
# 目前最快速Kmeans算法，并由java实现！面对很大的K值表现依然很好。
#1.这是一个由java实现的的，多线程Kmeans聚类算法；

#2.在聚类的选种阶段实现了Kmeans++算法和NIPS 2016的文章“Fast and Probably Good Seedings for k-Means”中提出了AFK-MC²算法，该算法改进了k-Means算法中初始种子点的生成方式，使其聚类速度相较于目前最好的k-Means++方式提高了好几个数量级。

#3.在迭代计算加速方面实现了2018年的一篇文章中描述的迭代加速算法Ameans：A-means improving the cluster assignment phase of k-means for Big Data；
上述两篇论文在项目的doc文件夹中。

#4.如果你的项目中的向量是稀疏向量，并且有值的位置可以提前保存到List中，那么可以将这个List纳入聚类计算中，在进行两个向量间的距离计算时，只计算有值位置的距离。

#5.实现起来不容易，如果觉得有用麻烦点个赞呗。

# 使用方法
1.一般的使用方式： List<List> clusterRes = MultiThreadKmeans.run(kmeansData, kmeansData.length*0.7, kmeansData.length * 20,10,2);

kmeansData: 矩阵

kmeansData.length*0.7 聚类个数

kmeansData.length * 20：最大迭代次数

10：开启线程数目

2：使用哪一种选种算法；1：随机选种 2：Kmean++选种  3: AFK-MC2选种

 

2.以文本聚类为例，如果在进行文本向量化是保存了词的非零位置： List<List> nouzeroList = TfIdf.getZeroList(nodeContents); //TFIDF方式进行文本向量化,nodeContents为文本内容

List<List> clusterRes = MultiThreadKmeans.run(kmeansData, kmeansData.length*0.7, kmeansData.length * 20,10,nouzeroList，2);
