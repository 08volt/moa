package moa.classifiers.meta.ELM;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.NotConvergedException;
import weka.core.Instances;

import java.io.NotSerializableException;
import java.io.Serializable;
import java.util.Random;

public class Selm implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = -4579057650893908831L;
    private static DenseMatrix train_set;
    private DenseMatrix test_set;
    private static int numTrainData;
    private int numTestData;
    private static DenseMatrix InputWeight;
    private static float TrainingTime;
    private float TestingTime;
    private static double TrainingAccuracy;
    private double TestingAccuracy;
    private static int Elm_Type;
    private static int NumberofHiddenNeurons;
    private static int NumberofOutputNeurons; // also the number of classes
    private static int NumberofInputNeurons; // also the number of
    // attribution
    private static String func;
    private static int[] label;
    // this class label employ a lazy and easy method,any class must written
    // in
    // 0,1,2...so the preprocessing is required

    // the blow variables in both train() and test()
    private static DenseMatrix BiasofHiddenNeurons;
    private static DenseMatrix OutputWeight;
    private DenseMatrix testP;
    private DenseMatrix testT;
    private static DenseMatrix Y;
    private static DenseMatrix T;

    // 选择预测的属性，和随机种子
    private static int m_classAtt = 9;
    private static int m_seed = 1;

    /**
     * Construct an ELM
     *
     * @param elm_type
     *            - 0 for regression; 1 for (both binary and multi-classes)
     *            classification
     * @param numberofHiddenNeurons
     *            - Number of hidden neurons assigned to the ELM
     * @param ActivationFunction
     *            - Type of activation function: 'sig' for Sigmoidal
     *            function 'sin' for Sine function 'hardlim' for Hardlim
     *            function 'tribas' for Triangular basis function 'radbas'
     *            for Radial basis function (for additive type of SLFNs
     *            instead of RBF type of SLFNs)
     * @throws NotConvergedException
     */

    public Selm(int elm_type, int numberofHiddenNeurons,
                String ActivationFunction) {

        Elm_Type = elm_type;
        NumberofHiddenNeurons = numberofHiddenNeurons;
        func = ActivationFunction;

        TrainingTime = 0;
        TestingTime = 0;
        TrainingAccuracy = 0;
        TestingAccuracy = 0;
        NumberofOutputNeurons = 1;
    }

    // --------1
    /**
     * 构造函数
     *
     * @param elm_type
     *            - 判断是数值型预测，还是名词性分类
     * @param numberofHiddenNeurons
     *            - 隐藏神经元个数
     * @param ActivationFunction
     *            - 激活函数 sig sin 两个可使用
     * @param randomSeed
     *            - 随机种子，为了算法可重现使用
     * @param classIndex
     *            - 预测或分类 属性标号，位置
     */
    public Selm(int elm_type, int numberofHiddenNeurons,
                String ActivationFunction, int randomSeed, int classIndex) {
        Selm.m_seed = randomSeed;
        Selm.m_classAtt = classIndex;

        Elm_Type = elm_type;
        NumberofHiddenNeurons = numberofHiddenNeurons;
        func = ActivationFunction;
        NumberofOutputNeurons = 1; // 默认为一，数值型输出

        TrainingTime = 0;
        TestingTime = 0;
        TrainingAccuracy = 0;
        TestingAccuracy = 0;
    }

    public Selm() {

    }

    // by myself
    public static DenseMatrix TransMatrixAtt(DenseMatrix source, int att) {
        double temp;
        int rows = source.numRows();
        for (int i = 0; i < rows; i++) {
            temp = source.get(i, 0);
            source.set(i, 0, source.get(i, att));
            source.set(i, att, temp);
        }
        return source;
    }

    // --------3
    // by myself
    public static double[][] TransMatrixAtt(double[][] source, int rows,
                                            int att) {
        double temp;
        for (int i = 0; i < rows; i++) {
            temp = source[i][0];
            source[i][0] = source[i][att];
            source[i][att] = temp;
        }
        return source;
    }

    // --------4
    /**
     * 如果为名词性预测，提取 预测的名词 个数 ，从 0 开始 ，所以要 +1
     *
     * @param traindata
     * @throws NotConvergedException
     */
    public static void train(double[][] traindata)
            throws NotConvergedException {

        // classification require a the number of class

        train_set = new DenseMatrix(traindata);
        // train_set = TransMatrixAtt(train_set, m_classAtt);
        int m = train_set.numRows();
        if (Elm_Type == 1) {
            double maxtag = traindata[0][0];
            for (int i = 0; i < m; i++) {
                if (traindata[i][0] > maxtag)
                    maxtag = traindata[i][0];
            }
            NumberofOutputNeurons = (int) maxtag + 1;
        }

        train();
    }

    /*
     * public void WriteToFile(double[][] source,int r,int c, String
     * fileName) { try { DecimalFormat format = new
     * DecimalFormat("##0.00000000"); BufferedWriter writer = new
     * BufferedWriter(new FileWriter(new File( fileName))); writer.write(r +
     * " " + c); for (int i = 0; i < r; i++) { writer.newLine(); for (int j
     * = 0; j < c; j++) {
     * writer.write(String.valueOf(format.format(source[i][j])) + ' '); } }
     * writer.flush(); writer.close();
     *
     * } catch (IOException e) { // TODO Auto-generated catch block
     * e.printStackTrace(); } }
     */

    // --------2
    // by myself
    /**
     * 将数据归一化为双精度 二维数组
     *
     * @param instances
     * @param nomalization
     * @throws NotConvergedException
     * @throws NotSerializableException
     */
    public void train(Instances instances, double[][] nomalization)
            throws NotConvergedException, NotSerializableException {
        int rows = instances.numInstances();
        int columns = instances.numAttributes();
        double[][] traindata = new double[rows][columns];

        /**
         * 数值属性归一化 x-min/(max-min)
         */
        for (int j = 0; j < columns; j++) {
            if (instances.attribute(j).isNumeric()) {
                for (int i = 0; i < rows; i++) {
                    traindata[i][j] = instances.instance(i).value(j)
                            - nomalization[1][j];
                    traindata[i][j] /= nomalization[0][j]
                            - nomalization[1][j];
                }
            } else {
                for (int i = 0; i < rows; i++) {
                    traindata[i][j] = instances.instance(i).value(j);
                }
            }
        }

        /*
         * for(int k=0; k<rows; k++){
         * System.out.println(java.util.Arrays.toString(traindata[k])); }
         */
        TransMatrixAtt(traindata, rows, m_classAtt);// ELM 算法
        // 将第一列作为预测属性，将原来的预测属性转至第一列
        /*
         * for(int k=0; k<rows; k++){
         * System.out.println(java.util.Arrays.toString(traindata[k])); }
         */
        // WriteToFile(traindata, rows, columns, "D:\\weather.txt");

        train(traindata);
    }

    // --------5
    // by my self
    /**
     * 随机化 矩阵 因为有随机种子，随机数可控，算法结果可重现
     *
     * @param rows
     * @param columns
     * @param seed
     * @return
     */
    public static DenseMatrix randomMatrix(int rows, int columns, int seed) {
        Random x = new Random(seed);
        DenseMatrix source = new DenseMatrix(rows, columns);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                source.set(i, j, x.nextDouble());
            }
        }
        return source;
    }

    // --------4
    /**
     * 训练数据
     *
     * @throws NotConvergedException
     */
    private static void train() throws NotConvergedException {
        System.out.println("training");
        numTrainData = train_set.numRows();// 训练数据行数
        NumberofInputNeurons = train_set.numColumns() - 1;// 输入属性为全部属性减去输出属性
        /**
         * 这样随机出的矩阵不可控，结果不可重现 InputWeight = (DenseMatrix)
         * Matrices.random(NumberofHiddenNeurons, NumberofInputNeurons);
         */
        InputWeight = randomMatrix(NumberofHiddenNeurons,
                NumberofInputNeurons, m_seed);

        DenseMatrix transT = new DenseMatrix(numTrainData, 1);// transT(numTrainData,1)
        DenseMatrix transP = new DenseMatrix(numTrainData,// transP(numTrainData,NumberofInputNeurons)
                NumberofInputNeurons);
        for (int i = 0; i < numTrainData; i++) {
            transT.set(i, 0, train_set.get(i, 0));
            for (int j = 1; j <= NumberofInputNeurons; j++)
                transP.set(i, j - 1, train_set.get(i, j));
        }
        // WriteToFile(InputWeight,"InputWeight.txt",1);

        T = new DenseMatrix(1, numTrainData);// T(1,numTrainData)
        DenseMatrix P = new DenseMatrix(NumberofInputNeurons, numTrainData);// P(NumberofInputNeurons,numTrainData)
        transT.transpose(T);// T = transT 转置
        transP.transpose(P);// P = transP 转置
        System.out.println(Elm_Type);
        if (Elm_Type != 0) // CLASSIFIER
        {
            label = new int[NumberofOutputNeurons];
            for (int i = 0; i < NumberofOutputNeurons; i++) {
                label[i] = i; // class label starts form 0
            }
            DenseMatrix tempT = new DenseMatrix(NumberofOutputNeurons,// tempT(NumberofOutputNeurons,numTrainData)
                    numTrainData);
            tempT.zero();

            System.out.println(NumberofOutputNeurons);

            for (int i = 0; i < numTrainData; i++) {
                int j = 0;
                for (j = 0; j < NumberofOutputNeurons; j++) {
                    if (label[j] == T.get(0, i))
                        break;
                }
                tempT.set(j, i, 1);
            }

            T = new DenseMatrix(NumberofOutputNeurons, numTrainData); // T=temp_T*2-1;
            for (int i = 0; i < NumberofOutputNeurons; i++) {
                for (int j = 0; j < numTrainData; j++)
                    T.set(i, j, tempT.get(i, j) * 2 - 1);
            }

            transT = new DenseMatrix(numTrainData, NumberofOutputNeurons);
            T.transpose(transT);

        } // end if CLASSIFIER

        long start_time_train = System.currentTimeMillis();
        // Random generate input weights InputWeight (w_i) and biases
        // BiasofHiddenNeurons (b_i) of hidden neurons
        // InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;

        /**
         * BiasofHiddenNeurons = (DenseMatrix) Matrices.random(
         * NumberofHiddenNeurons, 1);
         */
        BiasofHiddenNeurons = randomMatrix(NumberofHiddenNeurons, 1, m_seed);

        DenseMatrix tempH = new DenseMatrix(NumberofHiddenNeurons,// tempH(NumberofHiddenNeurons,numTrainData)
                numTrainData);
        InputWeight.mult(P, tempH);

        // WriteToFile(tempH, "myOutput", 1);// write to file {the matrix}

        // tempH = InputWeight * P

        // DenseMatrix ind = new DenseMatrix(1, numTrainData);

        DenseMatrix BiasMatrix = new DenseMatrix(NumberofHiddenNeurons,// BiasMatrix(NumberofHiddenNeurons,numTrainData)
                numTrainData);

        for (int j = 0; j < numTrainData; j++) {
            for (int i = 0; i < NumberofHiddenNeurons; i++) {
                BiasMatrix.set(i, j, BiasofHiddenNeurons.get(i, 0));
            }
        }

        tempH.add(BiasMatrix);
        DenseMatrix H = new DenseMatrix(NumberofHiddenNeurons, numTrainData);

        if (func.startsWith("sig")) {
            for (int j = 0; j < NumberofHiddenNeurons; j++) {
                for (int i = 0; i < numTrainData; i++) {
                    double temp = tempH.get(j, i);
                    temp = 1.0f / (1 + Math.exp(-temp));
                    H.set(j, i, temp);
                }
            }
        } else if (func.startsWith("sin")) {
            for (int j = 0; j < NumberofHiddenNeurons; j++) {
                for (int i = 0; i < numTrainData; i++) {
                    double temp = tempH.get(j, i);
                    temp = Math.sin(temp);
                    H.set(j, i, temp);
                }
            }
        } else if (func.startsWith("hardlim")) {
            // If you need it ,you can absolutely complete it yourself
        } else if (func.startsWith("tribas")) {
            // If you need it ,you can absolutely complete it yourself
        } else if (func.startsWith("radbas")) {
            // If you need it ,you can absolutely complete it yourself
            double a = 2, b = 2, c = Math.sqrt(2);
            for (int j = 0; j < NumberofHiddenNeurons; j++) {
                for (int i = 0; i < numTrainData; i++) {
                    double temp = tempH.get(j, i);
                    temp = a * Math.exp(-(temp - b) * (temp - b) / c * c);
                    H.set(j, i, temp);
                }
            }
        }

        DenseMatrix Ht = new DenseMatrix(numTrainData,// Ht(numTrainData,NumberofHiddenNeurons)
                NumberofHiddenNeurons);
        H.transpose(Ht);

        Inverse invers = new Inverse(Ht);
        System.out.println("Ht Inverse...");
        DenseMatrix pinvHt = invers.getMPInverse(); // NumberofHiddenNeurons*numTrainData
        // DenseMatrix pinvHt = invers.getMPInverse(0.000001); //fast
        // method,
        // PLEASE CITE in your paper properly:
        // Guang-Bin Huang, Hongming Zhou, Xiaojian Ding, and Rui Zhang,
        // "Extreme Learning Machine for Regression and Multi-Class Classification,"
        // submitted to IEEE Transactions on Pattern Analysis and Machine
        // Intelligence, October 2010.

        OutputWeight = new DenseMatrix(NumberofHiddenNeurons,// outputWeight(NumberofHiddenNeurons,NumberofOutputNeurons)
                NumberofOutputNeurons);
        pinvHt.mult(transT, OutputWeight);// OutputWeight=pinv(H') * T';

        long end_time_train = System.currentTimeMillis();
        TrainingTime = (end_time_train - start_time_train) * 1.0f / 1000;

        DenseMatrix Yt = new DenseMatrix(numTrainData,// Yt(numTrainData,NumberofOutputNeurons)
                NumberofOutputNeurons);
        Ht.mult(OutputWeight, Yt);
        Y = new DenseMatrix(NumberofOutputNeurons, numTrainData);// Y(NumberofOutputNeurons,numTrainData)

        Yt.transpose(Y);
        if (Elm_Type == 0) {
            double MSE = 0;
            for (int i = 0; i < numTrainData; i++) {
                MSE += (Yt.get(i, 0) - transT.get(i, 0))
                        * (Yt.get(i, 0) - transT.get(i, 0));
            }
            TrainingAccuracy = Math.sqrt(MSE / numTrainData);
        }

        // CLASSIFIER
        else if (Elm_Type == 1) {
            float MissClassificationRate_Training = 0;

            for (int i = 0; i < numTrainData; i++) {
                double maxtag1 = Y.get(0, i);
                int tag1 = 0;
                double maxtag2 = T.get(0, i);
                int tag2 = 0;
                for (int j = 1; j < NumberofOutputNeurons; j++) {
                    if (Y.get(j, i) > maxtag1) {
                        maxtag1 = Y.get(j, i);
                        tag1 = j;
                    }
                    if (T.get(j, i) > maxtag2) {
                        maxtag2 = T.get(j, i);
                        tag2 = j;
                    }
                }
                if (tag1 != tag2)
                    MissClassificationRate_Training++;
            }
            TrainingAccuracy = 1 - MissClassificationRate_Training * 1.0f
                    / numTrainData;

        }
        System.out.println("calculate ....");// ..........................
    }

    public double[] testOut(double[][] inpt, int r, int c) {
        test_set = new DenseMatrix(inpt);
        return testOut(r, c);
    }

    /*
     * public double[] testOut(double[] inpt, int r, int c) { test_set = new
     * DenseMatrix(new DenseVector(inpt)); return testOut(r, c); }
     */

    // Output numTestData*NumberofOutputNeurons
    private double[] testOut(int rows, int columns) {
        numTestData = rows;// test_set.numRows();
        NumberofInputNeurons = columns;// test_set.numColumns();// 问题
        // System.out.println(numTestData+","+NumberofInputNeurons+","+test_set.numColumns());
        DenseMatrix ttestT = new DenseMatrix(numTestData, 1);
        DenseMatrix ttestP = new DenseMatrix(numTestData,
                NumberofInputNeurons);
        for (int i = 0; i < numTestData; i++) {
            ttestT.set(i, 0, test_set.get(i, 0));
            for (int j = 0; j < NumberofInputNeurons; j++)
                ttestP.set(i, j, test_set.get(i, j));
        }

        testT = new DenseMatrix(1, numTestData);
        testP = new DenseMatrix(NumberofInputNeurons, numTestData);
        ttestT.transpose(testT);
        ttestP.transpose(testP);
        // test_set.transpose(testP);
        // System.out.println(NumberofHiddenNeurons+" "+numTestData);
        DenseMatrix tempH_test = new DenseMatrix(NumberofHiddenNeurons,
                numTestData);
        // System.out.println(InputWeight.numRows()+" "+InputWeight.numColumns());
        // System.out.println(testP.numRows()+" "+testP.numColumns());
        InputWeight.mult(testP, tempH_test);
        DenseMatrix BiasMatrix2 = new DenseMatrix(NumberofHiddenNeurons,
                numTestData);
        for (int j = 0; j < numTestData; j++) {
            for (int i = 0; i < NumberofHiddenNeurons; i++) {
                BiasMatrix2.set(i, j, BiasofHiddenNeurons.get(i, 0));
            }
        }

        tempH_test.add(BiasMatrix2);
        DenseMatrix H_test = new DenseMatrix(NumberofHiddenNeurons,
                numTestData);

        if (func.startsWith("sig")) {
            for (int j = 0; j < NumberofHiddenNeurons; j++) {
                for (int i = 0; i < numTestData; i++) {
                    double temp = tempH_test.get(j, i);
                    temp = 1.0f / (1 + Math.exp(-temp));
                    H_test.set(j, i, temp);
                }
            }
        } else if (func.startsWith("sin")) {
            for (int j = 0; j < NumberofHiddenNeurons; j++) {
                for (int i = 0; i < numTestData; i++) {
                    double temp = tempH_test.get(j, i);
                    temp = Math.sin(temp);
                    H_test.set(j, i, temp);
                }
            }
        } else if (func.startsWith("hardlim")) {

        } else if (func.startsWith("tribas")) {

        } else if (func.startsWith("radbas")) {

        }

        DenseMatrix transH_test = new DenseMatrix(numTestData,
                NumberofHiddenNeurons);
        H_test.transpose(transH_test);
        DenseMatrix Yout = new DenseMatrix(numTestData,
                NumberofOutputNeurons);
        transH_test.mult(OutputWeight, Yout);

        // DenseMatrix testY = new
        // DenseMatrix(NumberofOutputNeurons,numTestData);
        // Yout.transpose(testY);

        double[] result = new double[numTestData];

        if (Elm_Type == 0) {
            for (int i = 0; i < numTestData; i++)
                result[i] = Yout.get(i, 0);
        }

        else if (Elm_Type == 1) {
            for (int i = 0; i < numTestData; i++) {
                int tagmax = 0;
                double tagvalue = Yout.get(i, 0);
                for (int j = 1; j < NumberofOutputNeurons; j++) {
                    if (Yout.get(i, j) > tagvalue) {
                        tagvalue = Yout.get(i, j);
                        tagmax = j;
                    }

                }
                result[i] = tagmax;
            }
        }
        return result;
    }

    public float getTrainingTime() {
        return TrainingTime;
    }

    public double getTrainingAccuracy() {
        return TrainingAccuracy;
    }

    public float getTestingTime() {
        return TestingTime;
    }

    public double getTestingAccuracy() {
        return TestingAccuracy;
    }

    public int getNumberofInputNeurons() {
        return NumberofInputNeurons;
    }

    public int getNumberofHiddenNeurons() {
        return NumberofHiddenNeurons;
    }

    public int getNumberofOutputNeurons() {
        return NumberofOutputNeurons;
    }

    public DenseMatrix getInputWeight() {
        return InputWeight;
    }

    public DenseMatrix getBiasofHiddenNeurons() {
        return BiasofHiddenNeurons;
    }

    public DenseMatrix getOutputWeight() {
        return OutputWeight;
    }

    public DenseMatrix getTrainSet() {
        return Selm.train_set;
    }
}