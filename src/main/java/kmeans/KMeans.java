package kmeans;

/**
 * 多线程keans接口
 * @author JetHu
 * @Date 18-4-27
 * @See
 * @Since 1.0
 */
public interface KMeans extends Runnable {
    void addKMeansListener(KMeansListener var1);

    void removeKMeansListener(KMeansListener var1);

    Cluster[] getClusters();
}
