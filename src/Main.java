import NeuralNetwork.NeuralNetwork;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import utils.LinearAlgebraUtils;
import utils.OpenCVIO;

import java.io.File;
import java.util.Arrays;
import java.util.Random;

import static NeuralNetwork.Loader.loadNeuralNetwork;
import static NeuralNetwork.Loader.saveNeuralNet;

public class Main {

    private static final int WIDTH = 32, OUTPUTS = 42;
    private static final int EPOCHS = 2, EXAMPLES_SUM = 2000;
    private static final double lr = 0.05d;
    private static final String path = "D:\\NeuralNetworkProject\\kazAlphabet\\to", pathToSave = "D:\\NeuralNetworkProject\\kazAlphabet\\weights";

    private NeuralNetwork nn;

    public static void main(String[] args){
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Main main = new Main();
        main.nn = new NeuralNetwork(lr, WIDTH*WIDTH, WIDTH*WIDTH/2, WIDTH*WIDTH/4, WIDTH*WIDTH/8, OUTPUTS);
        main.train();

        main.load();
    }

    private void load(){
        NeuralNetwork nn = loadNeuralNetwork(pathToSave, 0);

        //Нахожу папки к классом
        File mainDir = new File(path);
        File[] mainFiles = mainDir.listFiles();


        //Test ********
        Mat clasEx = Imgcodecs.imread(mainFiles[0].getAbsolutePath() + "\\" + 0 + ".png");
        Imgproc.cvtColor(clasEx, clasEx, CvType.CV_64FC1, 1);
        clasEx.convertTo(clasEx, CvType.CV_64FC1, 1d/255d);

        double[] inputs = LinearAlgebraUtils.getArrayFromMat(clasEx);

        double[] predictions = nn.feedForward(inputs);

        System.out.println(Arrays.toString(predictions));
        //*************

    }

    private void train(){
        //Брать по одному Mat по очереди из каждого файла
        //и обучать сеть с индексом файла
        //Сохранить параметры нейронной сети

        //Я нахожу папки к классом
        File mainDir = new File(path);
        File[] mainFiles = mainDir.listFiles();

        int num_of_classes = mainFiles.length;

        String[] classes = new String[num_of_classes];
        for(int i=0; i<num_of_classes; i++)
            classes[i] = mainFiles[i].getName();
        System.out.println(Arrays.toString(classes));

        //Нахожу сколько в каждом классе примеров
        int[] sizes = count_examples_in_each_class(mainFiles);

        //Цикл обучения
        //Мне и класс нужен рандомный и пример нужен рандомный
        Random random = new Random(System.nanoTime());
        for(int epoch = 0; epoch<EPOCHS; epoch++){

            for(int exs=0; exs<EXAMPLES_SUM; exs++){
                if(exs%100==0)
                    System.out.println(exs);

                for(int cl=0, randClass; cl<num_of_classes; cl++){
                    //String p = mainFiles[cl].getAbsolutePath() + "\\" + random.nextInt(sizes[cl]) + ".png";
                    //System.out.println(p);
                    randClass = random.nextInt(num_of_classes);
                    Mat clasEx = Imgcodecs.imread(mainFiles[randClass].getAbsolutePath() + "\\" + random.nextInt(sizes[randClass]) + ".png");
                    Imgproc.cvtColor(clasEx, clasEx, Imgproc.COLOR_BGRA2GRAY, 1);
                    clasEx.convertTo(clasEx, CvType.CV_64FC1, 1d/255d);
                    double[] inputs = LinearAlgebraUtils.getArrayFromMat(clasEx);
                    double[] targets = new double[num_of_classes];
                    targets[cl] = 1.0d;
                    //double[] predictions =
                            nn.feedForward(inputs);
                    nn.backpropagation(targets);
                }
            }

            System.out.println(epoch);
        }
        //Test ********
        Mat clasEx = Imgcodecs.imread(mainFiles[0].getAbsolutePath() + "\\" + 0 + ".png");
        System.out.println(mainFiles[0].getAbsolutePath() + "\\" + 0 + ".png");
        Imgproc.cvtColor(clasEx, clasEx, CvType.CV_64FC1, 1);
        clasEx.convertTo(clasEx, CvType.CV_64FC1, 1d/255d);
        double[] inputs = LinearAlgebraUtils.getArrayFromMat(clasEx);
        double[] predictions = nn.feedForward(inputs);
        System.out.println(Arrays.toString(predictions));
        //*************

        saveNeuralNet(nn, pathToSave);
        System.out.println("SAVED");

    }

    private int[] count_examples_in_each_class(File[] mainFiles){
        int[] sizes = new int[mainFiles.length];
        for(int ret=0; ret<mainFiles.length; ret++) {
            File dir = new File(mainFiles[ret].getAbsolutePath());
            File[] arrFiles = dir.listFiles();
            sizes[ret] = arrFiles.length;
        }
        return sizes;
    }
}
