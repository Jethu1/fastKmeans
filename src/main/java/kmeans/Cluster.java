package kmeans;

/**
 * 聚类结果集合
 * @author JetHu
 * @Date 18-4-27
 * @See
 * @Since 1.0
 */
public class Cluster {
    private int[] memberIndexes;
    private double[] mcenter;

    public Cluster(int[] memberIndexes, double[] center) {
        this.memberIndexes = memberIndexes;
        this.mcenter = center;
    }

    public int[] getMemberIndexes() {
        return this.memberIndexes;
    }

    public double[] getCenter() {
        return this.mcenter;
    }
}