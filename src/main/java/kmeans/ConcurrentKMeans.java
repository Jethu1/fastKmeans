package kmeans;

import java.util.*;
import java.util.concurrent.*;

/**
 * 多线程Kmeans算法实现类
 * @author JetHu
 * @Date 18-4-27
 * @See
 * @Since 1.0
 */
public class ConcurrentKMeans implements KMeans {


    // Temporary clusters used during the clustering process.  Converted to
    // an array of the simpler class Cluster at the conclusion.
    private ProtoCluster[] mprotoClusters;

    // Cache of coordinate-to-cluster distances. Number of entries =
    // number of clusters X number of coordinates.
    private double[][] mdistanceCache;

    // Used in makeAssignments() to figure out how many moves are made
    // during each iteration -- the cluster assignment for coordinate n is
    // found in mclusterAssignments[n] where the N coordinates are numbered
    // 0 ... (N-1)
    private int[] mclusterAssignments;

    // 2D array holding the coordinates to be clustered.
    private double[][] mcoordinates;
    // The desired number of clusters and maximum number
    // of iterations.
    private int mk;
    private int mmaxIterations;
    // Seed for the random number generator used to select
    // coordinates for the initial cluster centers.
    private long mrandomSeed;
    // The number of threads used to perform the subtasks.
    private int mthreadCount;
    // Subtask manager that handles the thread pool to which
    // time-consuming tasks are delegated.
    private SubtaskManager msubtaskManager;

    // An array of Cluster objects: the output of k-means.
    private Cluster[] mclusters;

    // Listeners to be notified of significant happenings.
    private List<KMeansListener> mlisteners = new ArrayList<KMeansListener>(1);

    //max cluster number 最大簇数目不应超过该数--disable
    private int maxCluster = 0;

    //Ameans算法，已经确定属于某个簇的point
    private Set<Integer> set = new HashSet<>();

    //稀疏矩阵非零位置
    private List<List<Integer>> nouZeroList;

    //存放距离数组
    private Map<Integer,int[]> distanceMap;

    /**
     * Constructor
     *
     * @param coordinates two-dimensional array containing the coordinates to be clustered.
     * @param k  the number of desired clusters.
     * @param maxIterations the maximum number of clustering iterations.
     * @param randomSeed seed used with the random number generator.
     * @param threadCount the number of threads to be used for computing time-consuming steps.
     * @param total 最大簇数量上限 
     */
    public ConcurrentKMeans(double[][] coordinates, int k, int maxIterations,
                            long randomSeed, int threadCount,int total) {
        mcoordinates = coordinates;
        // Can't have more clusters than coordinates.
        mk = Math.min(k, mcoordinates.length);
        mmaxIterations = maxIterations;
        mrandomSeed = randomSeed;
        mthreadCount = threadCount;
        maxCluster = 20 + (int)(((double)total / 70000) * 30);
        //maxCluster = 30;
    }

    /**
     * Constructor
     *
     * @param coordinates two-dimensional array containing the coordinates to be clustered.
     * @param k  the number of desired clusters.
     * @param maxIterations the maximum number of clustering iterations.
     * @param randomSeed seed used with the random number generator.
     * @param threadCount the number of threads to be used for computing time-consuming steps.
     */
    public ConcurrentKMeans(double[][] coordinates, int k, int maxIterations,
                            long randomSeed, int threadCount) {
        mcoordinates = coordinates;
        // Can't have more clusters than coordinates.
        mk = Math.min(k, mcoordinates.length);
        mmaxIterations = maxIterations;
        mrandomSeed = randomSeed;
        mthreadCount = threadCount;

        distanceMap = new Hashtable<>(mcoordinates.length);
    }

    /**
     * Constructor that uses the return from
     * <tt>Runtime.getRuntime().availableProcessors()</tt> as the number
     * of threads for time-consuming steps.
     *  @param coordinates two-dimensional array containing the coordinates to be clustered.
     * @param k  the number of desired clusters.
     * @param maxIterations the maximum number of clustering iterations.
     * @param randomSeed seed used with the random number generator.
     * @param mThread 矩阵非零位置
     * @param nouList 矩阵非零位置
     */
    public ConcurrentKMeans(double[][] coordinates, int k, int maxIterations,
                            long randomSeed,int mThread, List<List<Integer>> nouList) {
        mcoordinates = coordinates;
        // Can't have more clusters than coordinates.
        mk = Math.min(k, mcoordinates.length);
        mmaxIterations = maxIterations;
        mrandomSeed = randomSeed;
        mthreadCount = mThread;
        nouZeroList = nouList;
        distanceMap = new Hashtable<>(mcoordinates.length);
    }

    /**
     * Adds a KMeansListener to be notified of significant happenings.
     *
     * @param l  the listener to be added.
     */
    public void addKMeansListener(KMeansListener l) {
        synchronized (mlisteners) {
            if (!mlisteners.contains(l)) {
                mlisteners.add(l);
            }
        }
    }

    /**
     * Removes a KMeansListener
     *
     * @param l the listener to be removed.
     */
    public void removeKMeansListener(KMeansListener l) {
        synchronized (mlisteners) {
            mlisteners.remove(l);
        }
    }

    /**
     * Posts a message to registered KMeansListeners.
     *
     * @param message message
     */
    private void postKMeansMessage(String message) {
        if (mlisteners.size() > 0) {
            synchronized (mlisteners) {
                int sz = mlisteners.size();
                for (int i = 0; i < sz; i++) {
                    mlisteners.get(i).kmeansMessage(message);
                }
            }
        }
    }

    /**
     * Notifies registered listeners that k-means is complete.
     *
     * @param clusters the output of clustering.
     * @param executionTime the number of milliseconds taken to cluster.
     */
    private void postKMeansComplete(Cluster[] clusters, long executionTime) {
        if (mlisteners.size() > 0) {
            synchronized (mlisteners) {
                int sz = mlisteners.size();
                for (int i = 0; i < sz; i++) {
                    mlisteners.get(i).kmeansComplete(clusters, executionTime);
                }
            }
        }
    }

    /**
     * Notifies registered listeners that k-means has failed because of
     * a Throwable caught in the run method.
     *
     * @param err error
     */
    private void postKMeansError(Throwable err) {
        if (mlisteners.size() > 0) {
            synchronized (mlisteners) {
                int sz = mlisteners.size();
                for (int i = 0; i < sz; i++) {
                    mlisteners.get(i).kmeansError(err);
                }
            }
        }
    }

    /**
     * Get the clusters computed by the algorithm.  This method should
     * not be called until clustering has completed successfully.
     *
     * @return an array of Cluster objects.
     */
    public Cluster[] getClusters() {
        return mclusters;
    }

    /**
     * Run the clustering algorithm.
     */
    public void run() {

        try {

            // Note the start time.
            final long startTime = System.currentTimeMillis();

            postKMeansMessage("K-Means clustering started");

            //AFKMC选择初始聚类算法---  最大化初始节点距离
            long t1 = System.currentTimeMillis();
            // Randomly initialize the cluster centers creating the
            // array mprotoClusters.
            try {
                // The array mclusterAssignments is used only to keep track of the cluster
                // membership for each coordinate.  The method makeAssignments() uses it
                // to keep track of the number of moves.
                if (mclusterAssignments == null) {
                    mclusterAssignments = new int[mcoordinates.length];
                    // Initialize to -1 to indicate that they haven't been assigned yet.
                    Arrays.fill(mclusterAssignments, -1);
                }
                initKMeansPlusPlusCenters();
                //initDistanceCenters();   //71
                //initCenters();  // 31
                //initAFKMCCenters();
            } catch (Exception e) {
                e.printStackTrace();
            }
            long t2 = System.currentTimeMillis();
            postKMeansMessage("... centers initialized");

            // Instantiate the subtask manager.
            msubtaskManager = new SubtaskManager(mthreadCount);

            // Post a message about the state of concurrent subprocessing.
            if (mthreadCount > 1) {
                postKMeansMessage("... concurrent processing mode with "
                        + mthreadCount + " subtask threads");
            } else {
                postKMeansMessage("... non-concurrent processing mode");
            }

            // Perform the initial computation of distances.
            //computeDistances();

            long t3 = System.currentTimeMillis();

            // Make the initial cluster assignments.
            makeAssignments();

            long t4 = System.currentTimeMillis();
            // Number of moves in the iteration and the iteration counter.
            int moves = 0;
            int it = 0;

            //System.out.println("循环iteration " + it + " initecenter "+(t2-t1) 
            // + "  computeDistances " + (t3-t2) + " makeAssignments " + (t4-t3));

            // Main Loop:
            //
            // Two stopping criteria:
            // - no moves in makeAssignments
            //   (moves == 0)
            // OR
            // - the maximum number of iterations has been reached
            //   (it == mmaxIterations)
            //
            do {
                long t5 = System.currentTimeMillis();
                // Compute the centers of the clusters that need updating.
                computeCenters();

                //进行Ameans算法
                long t6 = System.currentTimeMillis();
                aMeans();

                // Compute the stored distances between the updated clusters and the
                // coordinates.
                //computeDistances();  //距离计算目前转移到makeAssignments中
                long t7 = System.currentTimeMillis();
                // Make this iteration's assignments.
                moves = makeAssignments();

                long t8 = System.currentTimeMillis();
                it++;
                //System.out.println("循环iteration " + it + " computecenter "
                // + (t6 - t5) + "  Ameans " +
                //        (t7 - t6) + " makeAssignments " + (t8 - t7) + " moves number: " + moves);

                postKMeansMessage("... iteration " + it + " moves = " + moves);

            //} while (moves > 0 && it < mmaxIterations);
            } while (moves > 0 && it < 20);

            // Transform the array of ProtoClusters to an array
            // of the simpler class Cluster.
            mclusters = generateFinalClusters();

            long executionTime = System.currentTimeMillis() - startTime;

            postKMeansComplete(mclusters, executionTime);

        } catch (Throwable t) {

            postKMeansError(t);

        } finally {

            // Clean up temporary data structures used during the algorithm.
            cleanup();

        }
    }

    //Ameans算法相关操作
    private void aMeans() {

        int num = mcoordinates.length;
        for (int i = 0; i < num; i++) {
            if (set.contains(i)) {
                continue;
            }
            //分配完i所属类别后进行计算equidistance index-a 以及 improved threshold-u
            //int[] nearTwo = nearestClusterArray(i);
            int[] nearTwo = distanceMap.get(i);
            double a1 = distance(mcoordinates[i],mprotoClusters[nearTwo[0]].getCenter());
            double a2 = distance(mcoordinates[i],mprotoClusters[nearTwo[1]].getCenter());
            double a = Math.abs(a1 - a2);

            double u1 = distance(mprotoClusters[nearTwo[0]].getCenter(),
                    mprotoClusters[nearTwo[0]].getOldCenter());
            double u2 = distance(mprotoClusters[nearTwo[1]].getCenter(),
                    mprotoClusters[nearTwo[1]].getOldCenter());
            double u = u1 + u2;

            //a > u 则认为i在这轮迭代中将不会移动，所以将i固定分配给簇，不再参与距离计算和重新分配。
            if (a > u) {
                set.add(i);
            }
        }
    }

    /**
     * Randomly select coordinates to be the initial cluster centers.
     * O(m + k) m:number of points  k: clusters number
     */
    private void initCenters() {

        Random random = new Random(mrandomSeed);

        int coordCount = mcoordinates.length;

        // The array mclusterAssignments is used only to keep track of the cluster
        // membership for each coordinate.  The method makeAssignments() uses it
        // to keep track of the number of moves.
        if (mclusterAssignments == null) {
            mclusterAssignments = new int[coordCount];
            // Initialize to -1 to indicate that they haven't been assigned yet.
            Arrays.fill(mclusterAssignments, -1);
        }

        // Place the coordinate indices into an array and shuffle it.
        int[] indices = new int[coordCount];
        for (int i = 0; i < coordCount; i++) {
            indices[i] = i;
        }
        for (int i = 0, m = coordCount; m > 0; i++, m--) {
            int j = i + random.nextInt(m);
            if (i != j) {
                // Swap the indices.
                indices[i] ^= indices[j];
                indices[j] ^= indices[i];
                indices[i] ^= indices[j];
            }
        }

        mprotoClusters = new ProtoCluster[mk];
        for (int i = 0; i < mk; i++) {
            int coordIndex = indices[i];
            mprotoClusters[i] = new ProtoCluster(mcoordinates[coordIndex], coordIndex);
            mclusterAssignments[indices[i]] = i;
        }
    }

    /**
     *  initial KMeansPlusPlus cluster centers. implement by jetHu
     *
     */
    private void initKMeansPlusPlusCenters() {

        Random gen = new Random(mrandomSeed);
        mprotoClusters = new ProtoCluster[mk];
        int m = mcoordinates.length;  //m 数据个数
        double[] distToClosestCentroid = new double[m];
        double[] weightedDistribution  = new double[m];  // cumulative sum of squared distances
        // The array mclusterAssignments is used only to keep track of the cluster
        // membership for each coordinate.  The method makeAssignments() uses it
        // to keep track of the number of moves.
        if (mclusterAssignments == null) {
            mclusterAssignments = new int[m];
            // Initialize to -1 to indicate that they haven't been assigned yet.
            Arrays.fill(mclusterAssignments, -1);
        }

        int choose = 0;

        for (int c = 0; c < mk; c++) {

            // first centroid: choose any data point
            if (c == 0) {
                choose = gen.nextInt(m);
            } else {
                // after first centroid, use a weighted distribution
                // check if the most recently added centroid is closer
                // to any of the points than previously added ones
                for (int p = 0; p < m; p++) {
                    // gives chosen points 0 probability of being
                    // chosen again -> sampling without replacement
                    double tempDistance = nouZeroDistance(
                            mcoordinates[p], mprotoClusters[c - 1].getCenter(),
                            nouZeroList.get(p),nouZeroList.get(mprotoClusters[c - 1].coord));

                    // base case: if we have only chosen one centroid so far, nothing to compare to
                    if (c == 1) {
                        distToClosestCentroid[p] = tempDistance;
                    } else { // c != 1
                        if (tempDistance < distToClosestCentroid[p]) {
                            distToClosestCentroid[p] = tempDistance;
                        }
                    }
                    // no need to square because the distance is the square of the euclidean dist
                    if (p == 0) {
                        weightedDistribution[0] = distToClosestCentroid[0];
                    } else {
                        weightedDistribution[p] =
                                weightedDistribution[p - 1] + distToClosestCentroid[p];
                    }

                }

                // choose the next centroid
                double rand = gen.nextDouble();
                for (int j = m - 1; j > 0; j--) {
                    // TODO: review and try to optimize
                    // starts at the largest bin. EDIT: not actually the largest
                    if (rand > weightedDistribution[j - 1] / weightedDistribution[m - 1]) {
                        choose = j; // one bigger than the one above
                        break;
                    } else {
                        // Because of invalid dimension errors, we can't make
                        // the forloop go to j2 > -1 when we have (j2-1) in the loop.
                        choose = 0;
                    }
                }
            }
            // store the choosen centroid
            mprotoClusters[c] = new ProtoCluster(mcoordinates[choose],choose);
            mclusterAssignments[choose] = c;
        }
    }

    /**
     * AFKMC select coordinates to be the initial cluster centers. JetHu java实现
     *| AFKMC^2 Seeding as described by Bachem, Lucic, Hassani and Krause (2016)
     *| Runtime :code:`O(nd + mk^2d)`
     *:param X: Datapoints. Shape: (n, d)
     *:param k: Number cluster centers.
     *:param m: Length of Markov Chain. Default 200
     *
     */
    private void initAFKMCCenters() {

        int coordCount = mcoordinates.length; //数据个数
        Random random = new Random(mrandomSeed);
        mprotoClusters = new ProtoCluster[mk];

        //默认100
        //int markv = (int)(mk * Math.pow(Math.log(coordCount),2) * Math.log(mk));
        int markv = 100;
        int index = random.nextInt(coordCount); //产生一个范围内随机数
        double[] c0 = mcoordinates[index];
        mprotoClusters[0] = new ProtoCluster(c0, index);
        mclusterAssignments[index] = 0;

        //概率数组
        double[] pro = new double[coordCount];

        //每一个距离数组
        double[] dis = new double[coordCount];
        double totalDistances = 0;
        for (int i = 0; i < coordCount; i++) {
            dis[i] = distance(mcoordinates[i],c0);
            dis[i] = dis[i] * dis[i];
            totalDistances += dis[i];
        }
        double balance = 1.0 / (2 * coordCount);

        //概率总和
        double proTotal = 0;
        for (int i = 0; i < coordCount; i++) {
            pro[i] = 0.5 * dis[i] / totalDistances + balance;
            proTotal += pro[i];
        }

        for (int i = 1; i < mk; i++) {
            int x = selectOne(pro,proTotal);
            double dx2 = shortestDis(mprotoClusters,mcoordinates[x],x,i);
            for (int j = 1; j < markv; j++) {
                int y = selectOne(pro,proTotal);
                double dy2 = shortestDis(mprotoClusters,mcoordinates[y],y,i);
                if (dx2 * pro[y] == 0 || (dy2 * pro[x]) / (dx2 * pro[y]) > random.nextDouble()) {
                    x = y;
                    dx2 = dy2;
                }
            }
            mprotoClusters[i] = new ProtoCluster(mcoordinates[x],x);
            mclusterAssignments[x] = i;
        }
    }

    /**
     * 寻找和已有中心的最近距离的平方
     * @param clusters 已选择的簇
     * @param x 根据概率选出的ponit
     * @param index x在mcoordinate中的序号
     * @param i 已经选择了多少个簇心
     * @return 选中的节点与上一个已选中的簇心节点的距离
     */
    private double shortestDis(ProtoCluster[] clusters,double[] x,int index,int i) {
        /*double min = Double.MAX_VALUE;
        for (int j = 0; j < i; j++) {
            //double dis = distance(clusters[j].getCenter(),x);
            double dis = nouZeroDistance(x,clusters[j].getCenter(),
            nouZeroList.get(index),nouZeroList.get(clusters[j].coord));
            if (min > dis) {
                min = dis;
            }
        }*/
        double dis = nouZeroDistance(x,clusters[i - 1].getCenter(),
                nouZeroList.get(index),nouZeroList.get(clusters[i - 1].coord));

        return dis * dis;
    }

    /**
     * 根据指定概率数组pro产生随机数。
     * @param pro  数据对应的概率数组
     * @param totalPro 概率总和
     * @return 随机数
     */
    private static int selectOne(double[] pro,double totalPro) {

        double rnd = Math.random() * totalPro;
        for (int i = 0; i < pro.length; i++) {
            rnd -= pro[i];
            if (rnd < 0) {
                return i;
            }
        }

        return -1;
    }


    /**
     * select max distance coordinates to be the initial cluster centers. JetHu java实现
     */
    private void initDistanceCenters() {

        Random random = new Random(mrandomSeed);
        mprotoClusters = new ProtoCluster[mk];
        int coordCount = mcoordinates.length;
        int index1 = random.nextInt(coordCount);
        double[] c1 = mcoordinates[index1];

        // The array mclusterAssignments is used only to keep track of the cluster
        // membership for each coordinate.  The method makeAssignments() uses it
        // to keep track of the number of moves.
        if (mclusterAssignments == null) {
            mclusterAssignments = new int[coordCount];
            // Initialize to -1 to indicate that they haven't been assigned yet.
            Arrays.fill(mclusterAssignments, -1);
        }
        //mprotoClusters[index1] = new ProtoCluster(c1, index1);
        //mclusterAssignments[index1] = 0;

        //每一个距离数组
        double[] dis = new double[coordCount];
        int[] index = new int[coordCount];

        for (int i = 0; i < coordCount; i++) {
            index[i] = i;
            dis[i] = distance(mcoordinates[i],c1);
        }
        //QuickSort.quickSort(dis,0,coordCount - 1,index);

        //
        int num = mcoordinates.length / mk;
        for (int i = 0; i < mk; i++) {
            mprotoClusters[i] = new ProtoCluster(mcoordinates[index[i * num]], index[i * num]);
            mclusterAssignments[index[i * num]] = i;
        }
    }

    /**
     * Recompute the centers of the protoclusters with
     * update flags set to true.
     */
    private void computeCenters() {

        int numclusters = mprotoClusters.length;

        // Sets the update flags of the protoclusters that haven't been deleted and
        // whose memberships have changed in the iteration just completed.
        //
        for (int c = 0; c < numclusters; c++) {
            ProtoCluster cluster = mprotoClusters[c];
            if (cluster.getConsiderForAssignment()) {
                if (!cluster.isEmpty()) {
                    // This sets the protocluster's update flag to
                    // true only if its membership changed in last call
                    // to makeAssignments().
                    cluster.setUpdateFlag();
                    // If the update flag was set, update the center.
                    if (cluster.needsUpdate()) {
                        //更新中心前，先将center保存在oldcenter中
                        cluster.updateOldCenter();
                        cluster.updateCenter(mcoordinates);
                    }
                } else {
                    // When a cluster loses all of its members, it
                    // falls out of contention.  So it is possible for
                    // k-means to return fewer than k clusters.
                    cluster.setConsiderForAssignment(false);
                }
            }
        }
    }

    /**
     * Compute distances between coodinates and cluster centers,
     * storing them in the distance cache.  Only distances that
     * need to be computed are computed.  This is determined by
     * distance update flags in the protocluster objects.
     */
    private void computeDistances() throws InsufficientMemoryException {

        if (mdistanceCache == null) {
            int numCoords = mcoordinates.length;
            int numclusters = mprotoClusters.length;
            // Explicit garbage collection to reduce likelihood of insufficient
            // memory.
            System.gc();
            // Ensure there is enough memory available for the distances.
            // Throw an exception if not.
            long memRequired = 8L * numCoords * numclusters;
            if (Runtime.getRuntime().freeMemory() < memRequired) {
                throw new InsufficientMemoryException();
            }
            // Instantiate an array to hold the distances between coordinates
            // and cluster centers
            mdistanceCache = new double[numCoords][numclusters];
        }

        // Bulk of the work is delegated to the
        // SubtaskManager.
        msubtaskManager.computeDistances();
    }

    /**
     * Assign each coordinate to the nearest cluster.  Called once
     * per iteration.  Returns the number of coordinates that have
     * changed their cluster membership.
     */
    private int makeAssignments() {

        // Checkpoint the clusters, so we'll be able to tell
        // which one have changed after all the assignments have been
        // made.
        int numclusters = mprotoClusters.length;
        for (int c = 0; c < numclusters; c++) {
            if (mprotoClusters[c].getConsiderForAssignment()) {
                mprotoClusters[c].checkPoint();
            }
        }

        //将确定不移动的点加入到其对应的簇Ameans
        for (int i = 0; i < mclusterAssignments.length; i++) {
            int c = mclusterAssignments[i];
            if (set.contains(i)) {
                mprotoClusters[c].add(i);
            }
        }

        // Bulk of the work is delegated to the SubtaskManager.
        msubtaskManager.makeAssignments();
        // Get the number of moves from the SubtaskManager.
        return msubtaskManager.numberOfMoves();
    }

    /**
     * Find the nearest cluster to the coordinate identified by
     * the specified index.
     */
    private int nearestCluster(int ndx) {
        int nearest = -1;
        double min = Double.MAX_VALUE;
        int numclusters = mprotoClusters.length;
        for (int c = 0; c < numclusters; c++) {
            if (mprotoClusters[c].getConsiderForAssignment()) {
                double d = mdistanceCache[ndx][c];
                if (d < min) {
                    min = d;
                    nearest = c;
                }
            }
        }
        return nearest;
    }

    /**
     * Find the nearest two cluster to the coordinate identified by
     * the specified index.   JetHu---------------
     */
    private int[] nearestClusterArray(int ndx) {

        int[] result = new int[2];

        int nearest = -1;
        int second = -1;

        double min = Double.MAX_VALUE;
        int numclusters = mprotoClusters.length;
        for (int c = 0; c < numclusters; c++) {
            if (mprotoClusters[c].getConsiderForAssignment()) {
                double d;
                if (nouZeroList != null) {
                    d = nouZeroDistance(mcoordinates[ndx], 
                            mprotoClusters[c].getCenter(),
                            nouZeroList.get(ndx),mprotoClusters[c].getNouZeroList());
                } else {
                    d = distance(mcoordinates[ndx], mprotoClusters[c].getCenter());
                }
                if (d < min) {
                    min = d;
                    second = nearest;
                    nearest = c;
                }
            }
        }
        if (second == -1) {
            second = nearest;
        }

        result[0] = nearest;
        result[1] = second;

        return result;
    }

    /**
     * Compute the euclidean distance between the two arguments.
     */
    private static double distance(double[] coord, double[] center) {

        int len = coord.length;
        double sumSquared = 0.0;
        for (int i = 0; i < len; i++) {
            double v = coord[i] - center[i];
            sumSquared += v * v;
        }
        return Math.sqrt(sumSquared);
    }

    /**
     * Compute the euclidean distance between the two arguments just use the nou zero elements.
     * @param coord 数组1
     * @param center 数组2
     * @param coordList 数组1的非零位置
     * @param centerList 数组2的非零位置
     * @return 两数组的欧氏距离
     */
    private static double nouZeroDistance(
            double[] coord, double[] center,List<Integer> coordList,List<Integer> centerList) {

        Set<Integer> set = new HashSet<>();
        set.addAll(centerList);
        set.addAll(coordList);

        double sumSquared = 0.0;
        for (Integer i : set) {
            double v = coord[i] - center[i];
            sumSquared += v * v;
        }

        return Math.sqrt(sumSquared);
    }

    /**
     * Generate an array of Cluster objects from mprotoClusters.
     *
     * @return array of Cluster object references.
     */
    private Cluster[] generateFinalClusters() {

        int numclusters = mprotoClusters.length;

        // Convert the proto-clusters to the final Clusters.
        //
        // - accumulate in a list.
        List<Cluster> clusterList = new ArrayList<Cluster>(numclusters);
        for (int c = 0; c < numclusters; c++) {
            ProtoCluster pcluster = mprotoClusters[c];
            if (!pcluster.isEmpty()) {
                Cluster cluster = new Cluster(pcluster.getMembership(), pcluster.getCenter());
                clusterList.add(cluster);
            }
        }

        // - convert list to an array.
        Cluster[] clusters = new Cluster[clusterList.size()];
        clusterList.toArray(clusters);

        return clusters;
    }

    /**
     * Clean up items used by the clustering algorithm that are no longer needed.
     */
    private void cleanup() {
        mprotoClusters = null;
        mdistanceCache = null;
        mclusterAssignments = null;
        if (msubtaskManager != null) {
            msubtaskManager.shutdown();
            msubtaskManager = null;
        }
    }

    /**
     * Cluster class used temporarily during clustering.  Upon completion,
     * the array of ProtoClusters is transformed into an array of
     * Clusters.
     * @author JetHu
     */
    private static class ProtoCluster {

        // The previous iteration's cluster membership and
        // the current iteration's membership.  Compared to see if the
        // cluster has changed during the last iteration.
        private int[] mpreviousMembership;
        private int[] mcurrentMembership;
        private int mcurrentSize;

        // The last cluster center.
        private double[] oldCenter;

        // The cluster center.
        private double[] mcenter;

        // Born true, so the first call to updateDistances() will set all the
        // distances.
        private boolean mupdateFlag = true;
        // Whether or not this cluster takes part in the operations.
        private boolean mconsiderForAssignment = true;

        //record nou zero index in center: Jet Hu
        private List<Integer> nouZeroList = new ArrayList<>();

        public List<Integer> getNouZeroList() {
            return nouZeroList;
        }

        //第一个向量的位置
        public int coord;

        /**
         * Constructor
         *
         * @param center  the initial cluster center.
         * @param coordIndex  the initial member.
         */
        ProtoCluster(double[] center, int coordIndex) {
            mcenter = (double[]) center.clone();
            oldCenter = (double[]) center.clone();
            // No previous membership.
            mpreviousMembership = new int[0];
            // Provide space for 10 members to be added initially.
            mcurrentMembership = new int[10];
            mcurrentSize = 0;
            coord = coordIndex;
            add(coordIndex);
        }

        /**
         * Get the members of this protocluster.
         *
         * @return an array of coordinate indices.
         */
        synchronized int[] getMembership() {
            trimCurrentMembership();
            return mcurrentMembership;
        }

        /**
         * Get the last cluster's center.
         *
         * @return
         */
        double[] getOldCenter() {
            return oldCenter;
        }

        /**
         * update the last cluster's center.
         *
         */
        void updateOldCenter() {
            oldCenter = mcenter.clone();
        }

        /**
         * Get the protocluster's center.
         *
         * @return
         */
        double[] getCenter() {
            return mcenter;
        }

        /**
         * Reduces the length of the array of current members to
         * the number of members.
         */
        void trimCurrentMembership() {
            if (mcurrentMembership.length > mcurrentSize) {
                int[] temp = new int[mcurrentSize];
                System.arraycopy(mcurrentMembership, 0, temp, 0, mcurrentSize);
                mcurrentMembership = temp;
            }
        }

        /**
         * Add a coordinate to the protocluster. Note that this
         * method has to be synchronized, because multiple threads
         * may be adding members to the cluster.
         *
         * @param ndx index of the coordinate to be added.
         */
        synchronized void add(int ndx) {
            // Ensure there's space to add the new member.
            if (mcurrentSize == mcurrentMembership.length) {
                // If not, double the size of mcurrentMembership.
                int newCapacity = Math.max(10, 2 * mcurrentMembership.length);
                int[] temp = new int[newCapacity];
                System.arraycopy(mcurrentMembership, 0, temp, 0, mcurrentSize);
                mcurrentMembership = temp;
            }
            // Add the index.
            mcurrentMembership[mcurrentSize++] = ndx;
        }

        /**
         * Does the protocluster contain any members?
         *
         * @return true if the cluster is empty.
         */
        boolean isEmpty() {
            return mcurrentSize == 0;
        }

        /**
         * Compares the previous and the current membership.
         * Sets the update flag to true if the membership
         * changed in the previous call to makeAssignments().
         */
        void setUpdateFlag() {
            // Trim the current membership array length down to the
            // number of members.
            trimCurrentMembership();
            // Since members may have been added by multiple threads, they
            // are probably not in order.  They must be sorted in order to
            // do a valid comparison with mpreviousMembership.
            Arrays.sort(mcurrentMembership);
            mupdateFlag = false;
            if (mpreviousMembership.length == mcurrentSize) {
                for (int i = 0; i < mcurrentSize; i++) {
                    if (mpreviousMembership[i] != mcurrentMembership[i]) {
                        mupdateFlag = true;
                        break;
                    }
                }
            } else { // Number of members has changed.
                mupdateFlag = true;
            }
        }

        /**
         * Clears the current membership after copying it to the
         * previous membership.
         */
        void checkPoint() {
            mpreviousMembership = mcurrentMembership;
            mcurrentMembership = new int[10];
            mcurrentSize = 0;
        }

        /**
         * Is this protocluster currently in contention争论，争夺?
         *
         * @return true if this cluster is still in the running.
         */
        boolean getConsiderForAssignment() {
            return mconsiderForAssignment;
        }

        /**
         * Set the flag to indicate that this protocluster is
         * in or out of contention.
         *
         * @param b 是否
         */
        void setConsiderForAssignment(boolean b) {
            mconsiderForAssignment = b;
        }

        /**
         * Get the value of the update flag.  This value is
         * used to determine whether to update the cluster center and
         * whether to recompute distances to the cluster.
         *
         * @return the value of the update flag.
         */
        boolean needsUpdate() {
            return mupdateFlag;
        }

        /**
         * Update the cluster center.
         *
         * @param coordinates the array of coordinates.
         */
        void updateCenter(double[][] coordinates) {
            Arrays.fill(mcenter, 0.0);
            if (mcurrentSize > 0) {
                for (int i = 0; i < mcurrentSize; i++) {
                    double[] coord = coordinates[mcurrentMembership[i]];
                    for (int j = 0; j < coord.length; j++) {
                        mcenter[j] += coord[j];
                    }
                }
                for (int i = 0; i < mcenter.length; i++) {
                    if (mcenter[i] > 0.0) {
                        nouZeroList.add(i);
                    }
                    mcenter[i] /= mcurrentSize;
                }
            }
        }
    }

    /**
     * The class which manages the SMT-adapted subtasks.
     * @author JetHu
     */
    private class SubtaskManager {

        // Codes used to identify what step is being done.
        static final int DOING_NOTHING = 0;
        static final int COMPUTING_DISTANCES = 1;
        static final int MAKING_ASSIGNMENTS = 2;

        // What the object is currently doing. Set to one of the
        // three codes above.
        private int mdoing = DOING_NOTHING;

        // True if the at least one of the Workers is doing something.
        private boolean mworking;

        // The executor that runs the Workers.
        // When in multiple processor mode, this is a ThreadPoolExecutor
        // with a fixed number of threads. In single-processor mode, it's
        // a simple implementation that calls the single worker's run
        // method directly.
        private Executor mexecutor;

        // A Barrier to wait on multiple Workers to finish up the current task.
        // In single-processor mode, there is no need for a barrier, so it
        // is not set.
        private CyclicBarrier mbarrier;

        // The worker objects which implement Runnable.
        private Worker[] mworkers;

        /**
         * Constructor
         *
         * @param numThreads the number of worker threads to be used forthe subtasks.
         */
        SubtaskManager(int numThreads) {

            if (numThreads <= 0) {
                throw new IllegalArgumentException("number of threads <= 0: "
                        + numThreads);
            }

            int coordCount = mcoordinates.length;

            // There would be no point in having more workers than
            // coordinates, since some of the workers would have nothing
            // to do.
            if (numThreads > coordCount) {
                numThreads = coordCount;
            }

            // Create the workers.
            mworkers = new Worker[numThreads];

            // To hold the number of coordinates for each worker.
            int[] coordsPerWorker = new int[numThreads];

            // Initialize with the base amounts.
            Arrays.fill(coordsPerWorker, coordCount / numThreads);

            // There may be some leftovers, since coordCount may not be
            // evenly divisible by numworkers. Add a coordinate to each
            // until all are covered.
            int leftOvers = coordCount - numThreads * coordsPerWorker[0];
            for (int i = 0; i < leftOvers; i++) {
                coordsPerWorker[i]++;
            }

            int startCoord = 0;
            // Instantiate the workers.
            for (int i = 0; i < numThreads; i++) {
                // Each worker needs to know its starting coordinate and the number of
                // coordinates it handles.
                mworkers[i] = new Worker(startCoord, coordsPerWorker[i]);
                startCoord += coordsPerWorker[i];
            }

            if (numThreads == 1) { // Single-processor mode.

                // Create a simple executor that directly calls the single
                // worker's run method.  Do not set the barrier.
                mexecutor = new Executor() {
                    public void execute(Runnable runnable) {
                        if (!Thread.interrupted()) {
                            runnable.run();
                        } else {
                            throw new RejectedExecutionException();
                        }
                    }
                };

            } else { // Multiple-processor mode.

                // Need the barrier to notify the controlling thread when the
                // Workers are done.
                mbarrier = new CyclicBarrier(numThreads, new Runnable() {
                    public void run() {
                        // Method called after all workers haved called await() on the
                        // barrier.  The call to workersDone()
                        // unblocks the controlling thread.
                        workersDone();
                    }
                });

                // Set the executor to a fixed thread pool with
                // threads that do not time out.
                mexecutor = Executors.newFixedThreadPool(numThreads);
            }
        }

        /**
         * Make the cluster assignments.
         *
         * @return true if nothing went wrong.
         */
        boolean makeAssignments() {
            mdoing = MAKING_ASSIGNMENTS;
            return work();
        }

        /**
         * Compute the distances between the coordinates and those centers with
         * update flags.
         *
         * @return true if nothing went wrong.
         */
        boolean computeDistances() {
            mdoing = COMPUTING_DISTANCES;
            return work();
        }

        /**
         * Perform the current subtask, waiting until all the workers
         * finish their part of the current task before returning.
         *
         * @return true if the subtask succeeded.
         */
        private boolean work() {
            boolean ok = false;
            // Set the working flag to true.
            mworking = true;
            try {
                if (mbarrier != null) {
                    // Resets the barrier so it can be reused if
                    // this is not the first call to this method.
                    mbarrier.reset();
                }
                // Now execute the run methods on the Workers.
                for (int i = 0; i < mworkers.length; i++) {
                    mexecutor.execute(mworkers[i]);
                }
                if (mbarrier != null) {
                    // Block until the workers are done.  The barrier
                    // triggers the unblocking.
                    waitOnWorkers();
                    // If the isBroken() method of the barrier returns false,
                    // no problems.
                    ok = !mbarrier.isBroken();
                } else {
                    // No barrier, so the run() method of a single worker
                    // was called directly and everything must have worked
                    // if we made it here.
                    ok = true;
                }
            } catch (RejectedExecutionException ree) {
                // Possibly thrown by the executor.
            } finally {
                mworking = false;
            }
            return ok;
        }

        /**
         * Called from work() to put the controlling thread into
         * wait mode until the barrier calls workersDone().
         */
        private synchronized void waitOnWorkers() {
            // It is possible for the workers to have finished so quickly that
            // workersDone() has already been called.  Since workersDone() sets
            // mworking to false, check this flag before going into wait mode.
            // Not doing so could result in hanging the SubtaskManager.
            while (mworking) {
                try {
                    // Blocks until workersDone() is called.
                    wait();
                } catch (InterruptedException ie) {
                    // mbarrier.isBroken() will return true.
                    break;
                }
            }
        }

        /**
         * Notifies the controlling thread that it can come out of
         * wait mode.
         */
        private synchronized void workersDone() {
            // If this gets called before waitOnWorkers(), setting this
            // to false prevents waitOnWorkers() from entering a
            // permanent wait.
            mworking = false;
            notifyAll();
        }

        /**
         * Shutdown the thread pool when k-means is finished.
         */
        void shutdown() {
            if (mexecutor instanceof ThreadPoolExecutor) {
                // This terminates the threads in the thread pool.
                ((ThreadPoolExecutor) mexecutor).shutdownNow();
            }
        }

        /**
         * Returns the number of cluster assignment changes made in the
         * previous call to makeAssignments().
         */
        int numberOfMoves() {
            // Sum the contributions from the workers.
            int moves = 0;
            for (int i = 0; i < mworkers.length; i++) {
                moves += mworkers[i].numberOfMoves();
            }
            return moves;
        }

        /**
         * The class which does the hard work of the subtasks.
         * @author JetHu
         */
        private class Worker implements Runnable {

            // Defines range of coordinates to cover.
            private int mstartCoord;
            private int mnumCoords;

            // Number of moves made by this worker in the last call
            // to workerMakeAssignments().  The SubtaskManager totals up
            // this value from all the workers in numberOfMoves().
            private int mmoves;

            /**
             * Constructor
             *
             * @param startCoord index of the first coordinate covered by this Worker.
             * @param numCoords the number of coordinates covered.
             */
            Worker(int startCoord, int numCoords) {
                mstartCoord = startCoord;
                mnumCoords = numCoords;
            }

            /**
             * Returns the number of moves this worker made in the last
             * execution of workerMakeAssignments()
             */
            int numberOfMoves() {
                return mmoves;
            }

            /**
             * The run method.  It accesses the SubtaskManager field mdoing
             * to determine what subtask to perform.
             */
            public void run() {

                try {
                    switch (mdoing) {
                        case COMPUTING_DISTANCES:
                            workerComputeDistances();
                            break;
                        case MAKING_ASSIGNMENTS:
                            //实现Ameans算法
                            workerMakeAssignmentsAMeans();
                            break;
                        default:
                            return;
                    }
                } finally {
                    // If there's a barrier, call its await() method.  To ensure it
                    // gets done, it's placed in the finally clause.
                    if (mbarrier != null) {
                        try {
                            mbarrier.await();
                            // barrier.isBroken() will return true if either of these
                            // exceptions happens, so the SubtaskManager will detect
                            // the problem.
                        } catch (InterruptedException ex) {
                            System.out.println("error" + ex);
                        } catch (BrokenBarrierException ex) {
                            System.out.println("error" + ex);
                        }
                    }
                }
            }

            /**
             * Compute the distances for the covered coordinates
             * to the updated centers.
             */
            private void workerComputeDistances() {
                int lim = mstartCoord + mnumCoords;
                for (int i = mstartCoord; i < lim; i++) {
                    //set中包含该point直接跳过，不需计算距离
                    if (set.contains(i)) {
                        continue;
                    }
                    int numclusters = mprotoClusters.length;
                    for (int c = 0; c < numclusters; c++) {
                        ProtoCluster cluster = mprotoClusters[c];
                        if (cluster.getConsiderForAssignment()
                                && cluster.needsUpdate()) {
                            if (nouZeroList == null) {
                                mdistanceCache[i][c] =
                                        distance(mcoordinates[i], cluster.getCenter());
                            } else {
                                mdistanceCache[i][c] = nouZeroDistance(
                                        mcoordinates[i], cluster.getCenter(),
                                        nouZeroList.get(i),cluster.getNouZeroList());
                            }
                        }
                    }
                }
            }

            /**
             * Assign each covered coordinate to the nearest cluster. JetHu
             */
            private void workerMakeAssignmentsMaxCluster() {
                mmoves = 0;
                int lim = mstartCoord + mnumCoords;

                for (int i = mstartCoord; i < lim; i++) {
                    int c = nearestCluster(i);

                    //jethu 当一个类别数目达到maxCluster个时，不在往里面增加，防止聚类粒度不均匀
                    if (mprotoClusters[c].getMembership().length >= maxCluster) {
                        mprotoClusters[c].setConsiderForAssignment(false);
                    }

                    mprotoClusters[c].add(i);
                    if (mclusterAssignments[i] != c) {
                        mclusterAssignments[i] = c;
                        mmoves++;
                    }
                }
            }


            /**
             * Assign each covered coordinate to the nearest cluster.
             */
            private void workerMakeAssignments() {
                mmoves = 0;
                int lim = mstartCoord + mnumCoords;
                for (int i = mstartCoord; i < lim; i++) {
                    int c = nearestCluster(i);
                    mprotoClusters[c].add(i);
                    if (mclusterAssignments[i] != c) {
                        mclusterAssignments[i] = c;
                        mmoves++;
                    }
                }
            }

            /**
             * Assign each covered coordinate to the nearest cluster
             * and implement Ameans algo by JetHu
             */
            private void workerMakeAssignmentsAMeans() {
                mmoves = 0;
                int lim = mstartCoord + mnumCoords;
                for (int i = mstartCoord; i < lim; i++) {
                    //set中包含该point直接跳过，不需分配
                    if (set.contains(i)) {
                        continue;
                    }
                    int[] array = nearestClusterArray(i);

                    //保存最近距离数组，留作Ameans使用
                    distanceMap.put(i,array);
                    int c = array[0];
                    mprotoClusters[c].add(i);
                    if (mclusterAssignments[i] != c) {
                        mclusterAssignments[i] = c;
                        mmoves++;
                    }
                }
            }

        }
    }

    public static void main(String[] args) {

        double[] d = new double[]{0.2,0.5,0.1,0.1};
        double total = 0.9;
        for (int i = 0; i < 100; i++) {
            System.out.println(selectOne(d,total));
        }
    }
}