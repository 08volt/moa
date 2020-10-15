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

public class newOSELM extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;

    public IntOption initialization_chunk = new IntOption("initialization_chunk", 'I',
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
    private static  DenseMatrix addallrows(DenseMatrix A, DenseMatrix B){
        DenseMatrix result = A.copy();
        for(int i = 0; i<A.numRows(); i++)
            for(int j = 0; j<A.numRows(); j++)
                result.set(i,j, A.get(i,j) + B.get(0,j));
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
     * @param bias 1 x Hidden
     * @return new H Semples X hidden
     */
    public static DenseMatrix sigmoidActFunc(DenseMatrix features, DenseMatrix weights, DenseMatrix bias) {
        assert(features.numColumns() == weights.numColumns());
        int numSamples = features.numRows();

        DenseMatrix V = addallrows(mult(features, transpose(weights)),bias);
        DenseMatrix Htemp = new DenseMatrix(numSamples, bias.numColumns());
        for (int j = 0; j < bias.numColumns(); j++) {
            for (int i = 0; i < numSamples; i++) {
                double temp = V.get(j, i);
                temp = 1.0f / (1 + Math.exp(-temp));
                Htemp.set(j, i, temp);
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
        if(this.activationFunction.getValue() == "sig") {
            this.bias = randomMatrix(numberofHiddenNeurons.getValue(), 1, seed, true);
        }else{
            System.out.println("Unknown activation function type");
            throw new NotImplementedError();
        }

        DenseMatrix H0 = calculateHiddenLayerActivation(features);
        this.M = new Inverse(mult(transpose(H0),H0)).getInverse();
        this.beta = mult(new Inverse(H0).getInverse(),targets);

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
        try {
            DenseMatrix I = eye(numSamples);

            DenseMatrix temp0 = new Inverse(add(I, mult(H, mult(M, Ht)))).getInverse(); // N x N

            DenseMatrix temp1 = mult(M, mult(Ht, mult(temp0, mult(H, M)))); // H x H

            this.M = sub(M, temp1); // Woodbury formula
            this.beta = add(beta, mult(M,mult(Ht,sub(targets,mult(H,beta))))); // H x O

        } catch (Exception e){
            e.printStackTrace();
            System.out.println("SVD not converge, ignore the current training cycle");
        }
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
    DenseMatrix M;
    int seed = 1;



    @Override
    public double[] getVotesForInstance(Instance inst) {
        if(inputWeights == null){

            this.inputWeights = randomMatrix(numberofHiddenNeurons.getValue(), numberofattributes.getValue(), seed, false);

            this.bias = randomMatrix(numberofHiddenNeurons.getValue(), 1, seed, true);

            this.beta = randomMatrix(numberofHiddenNeurons.getValue(), numberofclasses.getValue(), seed, false);
            return new double[0];

        }

        double[][] features = new double[1][inst.numAttributes()];
        for(int a = 0; a<inst.numAttributes(); a++){
            features[0][a] = inst.value(a);
        }

        return predict(new DenseMatrix(features)).getData();
    }

    @Override
    public void resetLearningImpl() {
        this.inputWeights = randomMatrix(numberofHiddenNeurons.getValue(), numberofattributes.getValue(), seed, false);

        this.bias = randomMatrix(numberofHiddenNeurons.getValue(), 1, seed, true);

        this.beta = randomMatrix(numberofHiddenNeurons.getValue(), numberofclasses.getValue(), seed, false);

    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        double[][] features = new double[1][inst.numAttributes()];
        for(int a = 0; a<inst.numAttributes(); a++){
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
