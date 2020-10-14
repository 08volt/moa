package moa.classifiers.meta.ELM;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Measurement;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.NotConvergedException;
import org.apache.xerces.util.SynchronizedSymbolTable;
import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class OSelm extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;

    public IntOption initialization_chunk = new IntOption("initialization_chunk", 'i',
            "The number of instances for the inizitialization chunk", 10, 1, 10000);
    public IntOption numberofHiddenNeurons = new IntOption("numberofHiddenNeurons", 'n',
            "The number of neurons in the hidden layer", 10, 1, 10000);
    public StringOption activationFunction = new StringOption("activationFunction", 'a',
            "The node's activation function", "sig");


    protected DenseMatrix H;
    protected DenseMatrix P;
    protected DenseMatrix Y;
    protected DenseMatrix outputWeights; //Bl
    private DenseMatrix InputWeight; //attributes weights
    private DenseMatrix BiasofHiddenNeurons;

    protected ArrayList<Instance> init_list_inst;


    protected int i_count = 0;

    public OSelm(){
        init_list_inst = new ArrayList<Instance>();
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if(i_count<=initialization_chunk.getValue()) {

            return new double[0];

        }

        DenseMatrix tempH = new DenseMatrix(init_list_inst.get(0).numClasses(),1);
        InputWeight.mult(P, tempH);
        DenseMatrix BiasMatrix = new DenseMatrix(numberofHiddenNeurons.getValue(),1);


        for (int i = 0; i < numberofHiddenNeurons.getValue(); i++) {
            BiasMatrix.set(i, 0, BiasofHiddenNeurons.get(i, 0));
        }

        tempH.add(BiasMatrix);

        DenseMatrix h = new DenseMatrix(1,init_list_inst.get(0).numAttributes());

        for (int j = 0; j < numberofHiddenNeurons.getValue(); j++) {
            double temp = tempH.get(j, 0);
            temp = Math.sin(temp);
            h.set(0, j, temp);
        }


        double[] output = new double[init_list_inst.get(0).numClasses()];

        for(int o = 0; o<output.length;o++){
            for(int hid = 0; hid<numberofHiddenNeurons.getValue();hid++){
                output[o] += h.get(0,hid)* outputWeights.get(o,0);
            }
        }

        return output;
    }

    private static DenseMatrix opposite(DenseMatrix m){
        DenseMatrix z = new DenseMatrix(m.numRows(),m.numColumns());
        z.zero();
        for(int r = 0; r<m.numRows(); r++){
            for(int c = 0; c<m.numColumns(); c++){
                z.set(r,c,m.get(r,c)*(-1));
            }

        }
        return z;
    }


    private static DenseMatrix transpose(DenseMatrix m){
        return (DenseMatrix) m.transpose(new DenseMatrix(m.numColumns(),m.numRows()));
    }

    private static  DenseMatrix mult(DenseMatrix A, DenseMatrix B){
        return (DenseMatrix) A.mult(B,new DenseMatrix(A.numRows(),B.numColumns()));
    }

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

    private void initialization() throws NotConvergedException {
        System.out.println("INIT START");
        //n input neurons = numAttributes
        //n output neurons = numClasses
        int seed = 1;
        int hiddenN = numberofHiddenNeurons.getValue();
        int nTrain = initialization_chunk.getValue();
        int inputN = init_list_inst.get(0).numAttributes();
        int outputN = init_list_inst.get(0).numClasses();

        H = new DenseMatrix(nTrain, hiddenN);
        Y = new DenseMatrix(nTrain,outputN); // MATRIX NxQ (i,j) == 1 if class(xi) = j
        for(int i = 0; i< nTrain; i ++){
            Y.set(i,(int)init_list_inst.get(i).classValue(),1);
        }





        InputWeight = randomMatrix(hiddenN,inputN, seed);
        outputWeights = new DenseMatrix(hiddenN,1);
        BiasofHiddenNeurons = randomMatrix(hiddenN, 1, seed);

        DenseMatrix transT = new DenseMatrix(nTrain, 1);// transT(numTrainData,1)
        DenseMatrix transP = new DenseMatrix(nTrain,inputN);

        //initialization of TransP with attributes values
        for (int i = 0; i < nTrain; i++) {
            for (int j = 0; j < inputN; j++)
                transP.set(i, j, init_list_inst.get(i).value(j));

        }

        DenseMatrix tempH = new DenseMatrix(hiddenN,nTrain);
        InputWeight.mult(transpose(transP), tempH);

        DenseMatrix BiasMatrix = new DenseMatrix(hiddenN,nTrain);

        for (int j = 0; j < nTrain; j++) {
            for (int i = 0; i < hiddenN; i++) {
                BiasMatrix.set(i, j, BiasofHiddenNeurons.get(i, 0));
            }
        }

        tempH.add(BiasMatrix);

        for (int j = 0; j < nTrain; j++) {
            for (int i = 0; i < hiddenN; i++) {
                double temp = tempH.get(i, j);
                temp = Math.sin(temp);
                H.set(j, i, temp);
            }
        }


        DenseMatrix mpH = (new Inverse(H)).getMPInverse();

        DenseMatrix transH = new DenseMatrix(H.numColumns(),H.numRows());
        H.transpose(transH);
        P = (new Inverse(mult(transH,H))).getInverse();
        System.out.println(Y.numColumns());
        //mpH.mult(Y,outputWeights);
        outputWeights = mult(mpH,Y);
        System.out.println(outputWeights.toString());
        System.out.println("INIT END");



    }


    @Override
    public void trainOnInstanceImpl(Instance inst) {
        i_count ++;
        if(i_count<initialization_chunk.getValue()) {

            init_list_inst.add(inst);
            return;

        }
        if(i_count == initialization_chunk.getValue()) {

            init_list_inst.add(inst);
            try {
                initialization();

            } catch (NotConvergedException e) {
                System.out.println("INIT NOT CONVERGENT");
            }
            return;
        }

        //H chunk k+1
        //P k+1 = P-PH'(I+HPH')ˆ(-1)HP
        //outputWeights k+1 B = B + PH'(Y-HB)

        //Sherman–Morrison formula

        int inputN = init_list_inst.get(0).numAttributes();
        int outputN = init_list_inst.get(0).numClasses();
        DenseMatrix transP = new DenseMatrix(1,numberofHiddenNeurons.getValue());
        for (int i = 0; i < numberofHiddenNeurons.getValue(); i++)
            transP.set(0, i, inst.value(i));

        InputWeight = new DenseMatrix(inputN,1);

        DenseMatrix tempH = mult(InputWeight,transP);
//        DenseMatrix BiasMatrix = new DenseMatrix(inputN,1);
//
//        BiasofHiddenNeurons = randomMatrix(inputN,1,58);
//
//        for (int i = 0; i < inputN; i++) {
//            BiasMatrix.set(i, 0, BiasofHiddenNeurons.get(i, 0));
//        }
//
//        tempH.add(BiasMatrix);

        DenseMatrix h = new DenseMatrix(1,numberofHiddenNeurons.getValue());

        for (int j = 0; j < numberofHiddenNeurons.getValue(); j++) {
            double temp = tempH.get(j, 0);
            temp = Math.sin(temp);
            h.set(0, j, temp);
        }


        DenseMatrix numerator = mult(mult(mult(P,transpose(h)),h),P);
        DenseMatrix denominator = mult(mult(h,P),transpose(h));
        double den = denominator.get(0,0) + 1;
        for(int r = 0; r<numerator.numRows(); r++){
            for(int c = 0; c<numerator.numColumns(); c++){
                numerator.set(r,c,numerator.get(r,c)/den);
            }

        }



        P.add(-1, numerator); //P update

        DenseMatrix t = new DenseMatrix(1,outputN);
        t.zero();
        t.add(0,(int)inst.classValue(),1);

        DenseMatrix adj = mult(h,outputWeights);
        adj = opposite(adj);

        System.out.println(adj.numRows() + "  " + adj.numColumns());
        System.out.println(t.numRows() + "  " + t.numColumns());

        adj.add(t);

        outputWeights.add(mult(P,mult(transpose(h),adj)));

    }



    @Override
    public void resetLearningImpl() {

    }



    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[0];
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {

    }

    @Override
    public boolean isRandomizable() {
        return false;
    }
}
