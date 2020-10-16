package moa.classifiers.meta.ELM;

import com.github.javacliparser.IntOption;
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Measurement;
import no.uib.cipr.matrix.DenseMatrix;
import org.kramerlab.bmad.algorithms.GreedySelector;
import scala.NotImplementedError;
import weka.core.Utils;

import java.util.Random;

public class OSELM extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;

    public IntOption n_initialization_chunk = new IntOption("initialization_chunk", 'I',
            "The number of instances for the inizitialization chunk", 10, 1, 10000);
    public IntOption numberofHiddenNeurons = new IntOption("numberofHiddenNeurons", 'N',
            "The number of neurons in the hidden layer", 20, 1, 10000);

    public IntOption numberofattributes = new IntOption("numberofattributes", 'A',
            "The number of attributes for each instance", 10, 1, 10000);

    public IntOption numberofclasses = new IntOption("numberofclasses", 'C',
            "The number of classes", 2, 1, 10000);
    public StringOption activationFunction = new StringOption("activationFunction", 'F',
            "The node's activation function", "sig");


    private static DenseMatrix transpose(DenseMatrix m){
        return (DenseMatrix) m.transpose(new DenseMatrix(m.numColumns(),m.numRows()));
    }
    private static  DenseMatrix mult(DenseMatrix A, DenseMatrix B){
        return (DenseMatrix) A.mult(B,new DenseMatrix(A.numRows(),B.numColumns()));
    }
    private static  DenseMatrix eye(int dim){
        DenseMatrix result = new DenseMatrix(dim,dim);
        for(int d = 0; d<dim; d++){
            result.set(d,d,1);
        }
        return result;
    }
    private static  DenseMatrix addallcols(DenseMatrix A, DenseMatrix B){
        DenseMatrix result = A.copy();
        for(int i = 0; i<A.numRows(); i++)
            for(int j = 0; j<A.numColumns(); j++)
                result.set(i,j, A.get(i,j) + B.get(j,0));
        return result;
    }
    private static  DenseMatrix sub(DenseMatrix A, DenseMatrix B){
        DenseMatrix result = B.copy();
        result.set(-1,B);
        return (DenseMatrix) result.add(A);
    }
    private static  DenseMatrix add(DenseMatrix A, DenseMatrix B){
        DenseMatrix result = A.copy();
        return (DenseMatrix) result.add(B);
    }
    public static DenseMatrix randomMatrix(int rows, int columns, int seed, boolean zeroMean) {
        Random x = new Random(seed);
        DenseMatrix source = new DenseMatrix(rows, columns);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                if(zeroMean)
                    source.set(i, j, x.nextDouble() * 2 -1);
                else
                    source.set(i, j, x.nextDouble());
            }
        }
        return source;
    }

    /**
     *
     * @param features Samples X attributes
     * @param weights Hidden x attributes
     * @param bias Hidden X 1
     * @return new H Semples X hidden
     */
    public static DenseMatrix sigmoidActFunc(DenseMatrix features, DenseMatrix weights, DenseMatrix bias) {
        assert(features.numColumns() == weights.numColumns());
        int numSamples = features.numRows();

        DenseMatrix V = addallcols(mult(features, transpose(weights)),bias);
        DenseMatrix Htemp = new DenseMatrix(numSamples, weights.numRows());
        for (int j = 0; j < weights.numRows(); j++) {
            for (int i = 0; i < numSamples; i++) {
                double temp = V.get(i, j);
                temp = 1.0f / (1 + Math.exp(-temp));
                Htemp.set(i, j, temp);
            }
        }
        return Htemp;

    }

    /**
     *
     * @param features Samples X attributes
     * @return new H Semples X hidden
     */
    public DenseMatrix calculateHiddenLayerActivation(DenseMatrix features){
        if(activationFunction.getValue().equals("sig")) {
            return sigmoidActFunc(features, this.inputWeights, this.bias);
        }

        System.out.println("Unknown activation function type");
        throw new NotImplementedError();


    }

    /**
     *
     * @param features (numSamples, numInputs)
     * @param targets (numSamples, numOutputs)
     */
    public void initializePhase( DenseMatrix features,DenseMatrix targets){
        int hiddenN = this.numberofHiddenNeurons.getValue();
        int inputN = this.numberofattributes.getValue();
        int outputN = this.numberofclasses.getValue();

        assert(features.numRows() == targets.numRows());
        assert(features.numColumns() == inputN);
        assert(features.numColumns() == outputN);


        //randomly initialize the input->hidden connections
        this.inputWeights = randomMatrix(hiddenN, inputN,seed, true);


        this.bias = randomMatrix(numberofHiddenNeurons.getValue(), 1, seed, true);


        DenseMatrix H0 = calculateHiddenLayerActivation(features);
        this.M = new Inverse(mult(transpose(H0),H0)).getInverse();
        try {
            this.beta = mult(new Inverse(H0).getMPInverse(), targets);
        }catch (Exception e){
            e.printStackTrace();
            System.out.println("NOT INVERTIBLE H0");
        }

    }

    /***
     *
     * @param features (numSamples, numInputs)
     * @param targets (numSamples, numOutputs)
     */
    public void trainChunk(DenseMatrix features,DenseMatrix targets ){
        assert(features.numRows() == targets.numRows());

        int numSamples = features.numRows();
        DenseMatrix H = this.calculateHiddenLayerActivation(features);
        DenseMatrix Ht = transpose(H);

        DenseMatrix I = eye(numSamples);



        DenseMatrix temp0 = new Inverse(add(I, mult(H, mult(M, Ht)))).getInverse(); // N x N

        DenseMatrix temp1 = mult(M, mult(Ht, mult(temp0, mult(H, M)))); // H x H

        this.M = sub(M, temp1); // Woodbury formula
        this.beta = add(beta, mult(M,mult(Ht,sub(targets,mult(H,beta))))); // H x O


    }

    /***
     *
     * @param features (numSamples, numInputs)
     * @return (numSamples, numOutputs)
     */
    public DenseMatrix predict(DenseMatrix features){
        DenseMatrix H = this.calculateHiddenLayerActivation(features);
        return mult(H,beta);

    }



    // input to hidden weights
    DenseMatrix inputWeights;
    // bias of hidden units
    DenseMatrix bias;
    // hidden to output layer connection
    DenseMatrix beta;
    // P (k+1)
    DenseMatrix M;

    //chunk for the initialization Train
    DenseMatrix initializationChunk;
    DenseMatrix initializationTargets;

    //instances seen so far
    int count_inst = 0;

    int seed = 1;



    @Override
    public double[] getVotesForInstance(Instance inst) {
        if(inputWeights == null){
            System.out.println(this.activationFunction.getValue());
            this.inputWeights = randomMatrix(numberofHiddenNeurons.getValue(), numberofattributes.getValue(), seed, false);

            this.bias = randomMatrix(numberofHiddenNeurons.getValue(), 1, seed, true);

            this.beta = randomMatrix(numberofHiddenNeurons.getValue(), numberofclasses.getValue(), seed, false);

            this.initializationChunk = new DenseMatrix(n_initialization_chunk.getValue(),numberofattributes.getValue());
            this.initializationTargets = new DenseMatrix(n_initialization_chunk.getValue(),inst.numClasses());


        }
        if(count_inst < n_initialization_chunk.getValue()){
            return new double[0];
        }

        double[][] features = new double[1][inst.numAttributes()-1];
        for(int a = 0; a<inst.numAttributes()-1; a++){
            features[0][a] = inst.value(a);
        }

        return predict(new DenseMatrix(features)).getData();
    }

    @Override
    public void resetLearningImpl() {
        this.inputWeights = randomMatrix(numberofHiddenNeurons.getValue(), numberofattributes.getValue(), seed, false);
        this.bias = randomMatrix(numberofHiddenNeurons.getValue(), 1, seed, true);
        this.beta = randomMatrix(numberofHiddenNeurons.getValue(), numberofclasses.getValue(), seed, false);
        this.initializationChunk = new DenseMatrix(n_initialization_chunk.getValue(),numberofattributes.getValue());
        this.initializationTargets = new DenseMatrix(n_initialization_chunk.getValue(),numberofclasses.getValue());

    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if(count_inst < n_initialization_chunk.getValue() -1 ){
            if(initializationChunk == null){
                this.initializationChunk = new DenseMatrix(n_initialization_chunk.getValue(),numberofattributes.getValue());
                this.initializationTargets = new DenseMatrix(n_initialization_chunk.getValue(),inst.numClasses());
            }

            for(int a = 0; a<inst.numAttributes()-1; a++){
                initializationChunk.set(count_inst,a,inst.value(a));
                initializationTargets.set(count_inst,(int)inst.classValue(),1);
            }
            count_inst ++;
            return;


        }
        if (count_inst == n_initialization_chunk.getValue() -1){
            for(int a = 0; a<inst.numAttributes()-1; a++){
                initializationChunk.set(count_inst,a,inst.value(a));
                initializationTargets.set(count_inst,(int)inst.classValue(),1);
            }
            this.initializePhase(initializationChunk,initializationTargets);
        }

        double[][] features = new double[1][inst.numAttributes()-1];
        for(int a = 0; a<inst.numAttributes()-1; a++){
            features[0][a] = inst.value(a);
        }
        DenseMatrix target = new DenseMatrix(1,numberofclasses.getValue());
        target.zero();
        target.set(0,(int)inst.classValue(),1);
        trainChunk(new DenseMatrix(features),target);

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
        return true;
    }
}
