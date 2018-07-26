package kmeans;

/**
 * 内存不足异常
 * @author JetHu
 * @Date 18-4-27
 * @See
 * @Since 1.0
 */
public class InsufficientMemoryException extends Exception {

    /**
     * Constructor.
     *
     * @param message an explanatory message.
     */
    public InsufficientMemoryException(String message) {
        super(message);
    }

    /**
     * Default constructor.
     */
    public InsufficientMemoryException() {}

}
