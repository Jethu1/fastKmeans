package kmeans;


import java.util.ArrayList;
import java.util.List;

/**
 *　以多线程的方式进行kmeans聚类
 * @author JetHu
 * @Date 18-4-26
 * @See
 * @Since 1.0
 */
public class MultiThreadKmeans {

    /**
     * 执行多线程Kmeans聚类
     * @param vectors 进行聚类的数组向量
     * @param clusterNum 聚类类别数
     * @param maxiteration 最大迭代次数
     * @param threadNum 线程数
     * @param initNum 根据initNum的大小确定选择哪一个初始选种算法  1：随机选种  2：Kmean++选种  3: AFK-MC2选种
     * @return  聚类结果
     */
    public static List<List<Integer>> run(
            double[][] vectors,int clusterNum,int maxiteration,int threadNum,int initNum) {

        List<List<Integer>> lists = new ArrayList<>();
        ConcurrentKMeans concurrentKMeans = new ConcurrentKMeans(
                vectors,clusterNum,maxiteration,10,threadNum,initNum);
        concurrentKMeans.run();
        Cluster[] clusters = concurrentKMeans.getClusters();
        for (Cluster cluster : clusters) {
            List<Integer> list = new ArrayList<>();
            for (Integer integer : cluster.getMemberIndexes()) {
                list.add(integer);
            }
            lists.add(list);
        }
        return lists;
    }

    /**
     * 第一次执行多线程ｋmeans聚类;考虑稀疏向量的非零值进行距离计算
     * @param vectors 进行聚类的数组向量
     * @param clusterNum 聚类类别数
     * @param maxiteration 最大迭代次数
     * @param threadNum 线程数
     * @param nouZeroList 文本向量中非0位置序号
     * @param initNum 根据initNum的大小确定选择哪一个初始选种算法  1：随机选种  2：Kmean++选种  3: AFK-MC2选种
     * @return 聚类结果
     */
    public static List<List<Integer>> run(double[][] vectors, int clusterNum,
                              int maxiteration, int threadNum,List<List<Integer>> nouZeroList,int initNum) {

        List<List<Integer>> lists = new ArrayList<>();
        ConcurrentKMeans concurrentKMeans = new ConcurrentKMeans(
                vectors,clusterNum,maxiteration,10,threadNum,nouZeroList,initNum);
        concurrentKMeans.run();
        Cluster[] clusters = concurrentKMeans.getClusters();
        for (Cluster cluster : clusters) {
            List<Integer> list = new ArrayList<>();
            for (Integer integer : cluster.getMemberIndexes()) {
                list.add(integer);
            }
            lists.add(list);
        }
        return lists;
    }

}
