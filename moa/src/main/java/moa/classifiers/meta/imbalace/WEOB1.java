package moa.classifiers.meta.imbalace;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.capabilities.CapabilitiesHandler;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Measurement;
import moa.options.ClassOption;

public class WEOB1 extends AbstractClassifier implements MultiClassClassifier,
        CapabilitiesHandler {

    private static final long serialVersionUID = 1L;

    public FloatOption oobtheta = new FloatOption("oobtheta", 'o',
            "The time decay factor for oob classifier", 0.9, 0, 1);
    public FloatOption uobtheta = new FloatOption("uobtheta", 'u',
            "The time decay factor for uob classifier", 0.9, 0, 1);
    public FloatOption recalltheta = new FloatOption("recalltheta", 'r',
            "The time decay factor for class recall", 0.9, 0, 1);

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of models in the bag.", 10, 1, Integer.MAX_VALUE);

    protected ImprovedOOB oob;
    protected ImprovedUOB uob;
    protected double[] classRecallOOB;
    protected double[] classRecallUOB;

    public WEOB1(){
        super();
        oob = new ImprovedOOB();
        uob = new ImprovedUOB();
        oob.theta.setValue(this.oobtheta.getValue());
        oob.ensembleSizeOption.setValue(ensembleSizeOption.getValue());
        oob.baseLearnerOption = this.baseLearnerOption;
        uob.theta.setValue(this.uobtheta.getValue());
        uob.ensembleSizeOption.setValue(ensembleSizeOption.getValue());
        uob.baseLearnerOption = this.baseLearnerOption;
        oob.prepareForUse();
        uob.prepareForUse();

    }

    public double calcGini(double[] recalls){
        if(recalls == null)
            return 1d;
        double gini = 1;
        for (double recall : recalls) {
            gini *= recall;
        }

        return Math.pow(gini,1d/(double) recalls.length);
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        double[] oobVotes = oob.getVotesForInstance(inst);
        double[] uobVotes = uob.getVotesForInstance(inst);
        double[] finalVotes = new double[oobVotes.length];

        double uobGini = calcGini(classRecallUOB);
        double oobGini = calcGini(classRecallOOB);

        double alphaO = oobGini / (oobGini + uobGini);
        double alphaU = uobGini / (oobGini + uobGini);

        for(int i = 0; i < oobVotes.length; i++){
            finalVotes[i] = alphaO * oobVotes[i] + alphaU*uobVotes[i];
        }
        return finalVotes;


    }

    @Override
    public void resetLearningImpl() {
        oob.resetLearningImpl();
        uob.resetLearningImpl();
    }

    public void recallUpdate(double[] recalls,int instanceClass, boolean pred){
        recalls[instanceClass] = this.recalltheta.getValue() * recalls[instanceClass] + (1d - this.recalltheta.getValue()) * (pred ? 1d:0d);
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (this.classRecallOOB == null) {
            classRecallOOB = new double[inst.numClasses()];
            classRecallUOB = new double[inst.numClasses()];
            // <---le19/01/18 modification to start class size as equal for all classes
            for (int i=0; i<classRecallOOB.length; ++i) {
                classRecallOOB[i] = 1d;
                classRecallUOB[i] = 1d;
            }
        }
        recallUpdate(classRecallOOB,(int)inst.classValue(),oob.correctlyClassifies(inst));
        recallUpdate(classRecallUOB,(int)inst.classValue(),uob.correctlyClassifies(inst));

        oob.trainOnInstanceImpl(inst);
        uob.trainOnInstanceImpl(inst);


    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return oob.getModelMeasurementsImpl();
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {

    }

    @Override
    public boolean isRandomizable() {
        return true;
    }
}
