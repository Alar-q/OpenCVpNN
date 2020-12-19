import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;

/**
 * Сохранение контуров на фото (сейчас в моём применении "фото" - фото буквы)
 * Каждый класс нужно в главной папке(from) разместить по разным папкам с именами класса
 */

public class RedactDataTest {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.out.println("START");

        int WIDTH = 32;
        Size size = new Size(WIDTH, WIDTH);

        String pathFrom = "D:\\digitProjects\\kazLet";
        String pathTo = "D:\\NeuralNetworkProject\\kazAlphabet\\from";

        redactImages(pathFrom, pathTo, size);
    }

    private static void redactImages(String pathFrom, String pathTo, Size size) {
        File mainDir = new File(pathFrom);
        File[] mainFiles = mainDir.listFiles();

        for(int i=0; i<mainFiles.length; i++){
            File newFile = new File(pathTo+"\\"+mainFiles[i].getName());
            if(!newFile.exists()) newFile.mkdirs();
        }

        for(int ret=0; ret<mainFiles.length; ret++) {
            File dir = new File(mainFiles[ret].getAbsolutePath());
            File[] arrFiles = dir.listFiles();

            if (arrFiles != null) {
                for (int i = 0; i < arrFiles.length; i++) {
                    Mat imgOriginal = Imgcodecs.imread(arrFiles[i].getAbsolutePath()) ;
                    Mat imgRedact = new Mat();
                    Imgproc.cvtColor(imgOriginal, imgRedact, Imgproc.COLOR_BGR2GRAY);

                    //Redact
                    Imgproc.resize(imgRedact, imgRedact, size);
                    Imgproc.Canny(imgRedact, imgRedact, 80, 200);

                    if(Imgcodecs.imwrite(pathTo + "\\" + mainFiles[ret].getName() + "\\" + i + ".png", imgRedact)){
                        System.out.println("saved");
                    }else
                        System.out.println("problems at: " + pathTo + "\\" + mainFiles[ret].getName() + "\\" + i + ".png");


                    imgOriginal.release(); imgRedact.release();
                }
            } else
                System.out.println("error! folder was not found");
        }
    }
}
