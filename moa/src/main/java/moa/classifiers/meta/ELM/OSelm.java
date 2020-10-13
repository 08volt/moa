package moa.classifiers.meta.ELM;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.NotConvergedException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class OSelm extends AbstractClassifier {

    private static final long serialVersionUID = 1L;

    public IntOption initialization_chunk = new IntOption("initialization_chunk", 'i',
            "The number of instances for the inizitialization chunk", 10, 1, 10000);
    public IntOption numberofHiddenNeurons = new IntOption("numberofHiddenNeurons", 'n',
            "The number of neurons in the hidden layer", 10, 1, 10000);
    public StringOption activationFunction = new StringOption("activationFunction", 'n',
            "The node's activation function", "sig");


    protected DenseMatrix H;
    protected DenseMatrix P;
    protected DenseMatrix Y;
    protected DenseMatrix outputWeights; //Bl
    private DenseMatrix InputWeight; //attributes weights
    private DenseMatrix BiasofHiddenNeurons;

    protected final Instances init_inst = new Instances();


    protected int i_count = 0;



    @Override
    public double[] getVotesForInstance(Instance inst) {
        if(i_count<initialization_chunk.getValue()) {

            init_inst.add(inst);
            return new double[0];
        }
        if(i_count == initialization_chunk.getValue()) {

            init_inst.add(inst);
            try {
                initialization();
                return new double[0];
            } catch (NotConvergedException e) {
                System.out.println("INIT NOT CONVERGENT");
                return new double[0];
            }
        }



        return new double[0];
    }

    private static DenseMatrix opposite(DenseMatrix m){
        DenseMatrix z = new DenseMatrix(m.numRows(),m.numColumns());
        z.zero();
        for(int i = 0; i<z.numRows(); i++)
            for(int j = 0; j<z.numColumns(); j++)
                z.add(i,j,-1);
        return mult(m,z);
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
        //n input neurons = numAttributes
        //n output neurons = numClasses
        int seed = randomSeedOption.getValue();
        int hiddenN = numberofHiddenNeurons.getValue();
        int nTrain = initialization_chunk.getValue();
        int inputN = init_inst.numAttributes();
        int outputN = init_inst.numClasses();

        H = new DenseMatrix(nTrain, hiddenN);
        Y = new DenseMatrix(nTrain,outputN); // MATRIX NxQ (i,j) == 1 if class(xi) = j
        for(int i = 0; i< nTrain; i ++){
            Y.set(i,(int)init_inst.instance(i).classValue(),1);
        }





        InputWeight = randomMatrix(hiddenN,inputN, seed);
        outputWeights = new DenseMatrix(hiddenN,1);
        BiasofHiddenNeurons = randomMatrix(hiddenN, 1, seed);

        DenseMatrix transT = new DenseMatrix(nTrain, 1);// transT(numTrainData,1)
        DenseMatrix transP = new DenseMatrix(nTrain,inputN);

        //initialization of TransP with attributes values
        for (int i = 0; i < nTrain; i++) {
            for (int j = 0; j < inputN; j++)
                transP.set(i, j - 1, init_inst.get(i).value(j));

        }

        DenseMatrix tempH = new DenseMatrix(outputN,nTrain);
        InputWeight.mult(transpose(transP), tempH);

        DenseMatrix BiasMatrix = new DenseMatrix(hiddenN,nTrain);

        for (int j = 0; j < nTrain; j++) {
            for (int i = 0; i < hiddenN; i++) {
                BiasMatrix.set(i, j, BiasofHiddenNeurons.get(i, 0));
            }
        }

        tempH.add(BiasMatrix);

        for (int j = 0; j < hiddenN; j++) {
            for (int i = 0; i < nTrain; i++) {
                double temp = tempH.get(j, i);
                temp = Math.sin(temp);
                H.set(j, i, temp);
            }
        }


        DenseMatrix mpH = (new Inverse(H)).getMPInverse();

        DenseMatrix transH = new DenseMatrix(H.numColumns(),H.numRows());
        H.transpose(transH);
        P = (new Inverse((DenseMatrix) H.mult(transH,P))).getInverse();
        mpH.mult(Y,outputWeights);

    }


    @Override
    public void trainOnInstanceImpl(Instance inst) {
        //H chunk k+1
        //P k+1 = P-PH'(I+HPH')ˆ(-1)HP
        //outputWeights k+1 B = B + PH'(Y-HB)

        //Sherman–Morrison formula
        DenseMatrix transP = new DenseMatrix(1,init_inst.numAttributes());
        for (int i = 0; i < init_inst.numAttributes(); i++)
            transP.set(0, i, inst.value(i));

        DenseMatrix tempH = new DenseMatrix(init_inst.numClasses(),1);
        InputWeight.mult(transpose(transP), tempH);

        DenseMatrix BiasMatrix = new DenseMatrix(numberofHiddenNeurons.getValue(),1);


        for (int i = 0; i < numberofHiddenNeurons.getValue(); i++) {
            BiasMatrix.set(i, 0, BiasofHiddenNeurons.get(i, 0));
        }

        tempH.add(BiasMatrix);

        DenseMatrix h = new DenseMatrix(1,init_inst.numAttributes());

        for (int j = 0; j < numberofHiddenNeurons.getValue(); j++) {
            double temp = tempH.get(j, 0);
            temp = Math.sin(temp);
            h.set(0, j, temp);
        }

        DenseMatrix numerator = mult(mult(mult(P,h),transpose(h)),P);
        DenseMatrix denominator = mult(mult(transpose(h),P),h);
        for(int i = 0; i<denominator.numRows(); i++)
            for(int j = 0; j<denominator.numColumns(); j++)
                denominator.add(i,j,1);

        P.add(-1, mult(numerator,(new Inverse(denominator)).getInverse())); //P update

        DenseMatrix t = new DenseMatrix(1,init_inst.numClasses());
        t.zero();
        t.add(0,(int)inst.classValue(),1);

        DenseMatrix adj = mult(transpose(h),outputWeights);
        adj = opposite(adj);
        adj.add(transpose(t));

        outputWeights.add(mult(P,mult(h,adj)));

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
