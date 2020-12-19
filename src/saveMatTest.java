import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import utils.LinearAlgebraUtils;
import utils.OpenCVIO;

import java.util.Arrays;

public class saveMatTest {
    public static void main(String[] args){
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        save();
        read();
    }

    private static String filePath = "D:\\NeuralNetworkProject\\1.mtcv";

    private static void save(){
        double[][] arr = new double[][]{{}, {}, {}};
        OpenCVIO.saveArray2D(filePath, arr);

        Mat mat = new Mat(3, 3, CvType.CV_64FC1);
        LinearAlgebraUtils.array2DToMat(arr, mat);
        System.out.println(mat.dump());
    }
    private static void read(){
        double[][] arr = OpenCVIO.readArray2DFromFile(filePath);//= new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        Mat mat = new Mat(arr.length, arr[0].length, CvType.CV_64FC1);
        LinearAlgebraUtils.array2DToMat(arr, mat);
        //OpenCVIO.saveArray2D(filePath, arr);
        System.out.println(mat.dump());
    }
}
