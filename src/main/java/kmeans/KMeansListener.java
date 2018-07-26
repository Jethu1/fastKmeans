package kmeans;

/**
 * 接口
 * @author JetHu
 * @Date 18-4-27
 * @See
 * @Since 1.0
 */
public interface KMeansListener {

    /**
     * A message has been received.
     *
     * @param message 接收消息
     */
    public void kmeansMessage(String message);

    /**
     * KMeans is complete.
     *
     * @param clusters the output of clustering.
     * @param executionTime the time in milliseconds taken to cluster.
     */
    public void kmeansComplete(Cluster[] clusters, long executionTime);

    /**
     * An error occurred during KMeans clustering.
     *
     * @param t 错误
     */
    public void kmeansError(Throwable t);

}