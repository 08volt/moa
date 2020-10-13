package moa.classifiers.meta.ELM;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import no.uib.cipr.matrix.DenseMatrix;
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
    protected DenseMatrix outputWeights;

    private ArrayList<Integer> y_values = new ArrayList<>();
    private Instances init_inst;


    protected int i_count = 0;



    @Override
    public double[] getVotesForInstance(Instance inst) {
        if(i_count<initialization_chunk.getValue()) {
            y_values.add((int)inst.classValue());
            init_inst.add(inst);
            return new double[0];
        }
        if(i_count == initialization_chunk.getValue()) {
            y_values.add((int)inst.classValue());
            init_inst.add(inst);
            try {
                return initialization();
            } catch (NotConvergedException e) {
                System.out.println("INIT NOT CONVERGENT");
                return new double[0];
            }
        }


        return new double[0];
    }

    private double[] initialization() throws NotConvergedException {
        H = new DenseMatrix(initialization_chunk.getValue(), numberofHiddenNeurons.getValue());
        Y = new DenseMatrix(initialization_chunk.getValue(),init_inst.numClasses());
        for(int i = 0; i< initialization_chunk.getValue(); i ++){
            Y.set(i,y_values.get(i),1);
        }
        y_values = null;

        int[] label = new int[init_inst.numClasses()];
        for (int i = 0; i < init_inst.numClasses(); i++) {
            label[i] = i; // class label starts form 0
        }


        for(int j = 0; j < numberofHiddenNeurons.getValue();  j ++) {
            //activation function
            if (activationFunction.getValue().startsWith("sig")) {
                for (int j = 0; j < numberofHiddenNeurons.getValue(); j++) {
                    for (int i = 0; i < initialization_chunk.getValue(); i++) {
                        double temp = tempH.get(j, i);
                        temp = 1.0f / (1 + Math.exp(-temp));
                        H.set(j, i, temp);
                    }
                }
            } else if (activationFunction.getValue().startsWith("sin")) {
                for (int j = 0; j < numberofHiddenNeurons.getValue(); j++) {
                    for (int i = 0; i < initialization_chunk.getValue(); i++) {
                        double temp = tempH.get(j, i);
                        temp = Math.sin(temp);
                        H.set(j, i, temp);
                    }
                }
            }
        }

        DenseMatrix mpH =  (new Inverse(H)).getMPInverse();
        outputWeights = new DenseMatrix(mpH.mult(Y,outputWeights));


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

    @Override
    public void resetLearningImpl() {

    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {

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
