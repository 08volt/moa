package moa.classifiers.meta.imbalace;

import com.github.javacliparser.FloatOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.meta.AdaptiveRandomForest;
import moa.core.DoubleVector;
import moa.core.InstanceExample;
import moa.core.MiscUtils;
import weka.classifiers.functions.neural.NeuralConnection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.concurrent.ExecutorService;

public class ARFwithClassImbalance extends AdaptiveRandomForest {

    public FloatOption theta = new FloatOption("theta", 't',
            "The time decay factor for class size.", 0.9, 0, 1);

    protected double[] classSize;



    @Override
    public void trainOnInstanceImpl(Instance instance) {
        super.disableWeightedVote.setValue(false);
        super.disableDriftDetectionOption.setValue(true);
        updateClassSize(instance);
        super.lambdaOption.setValue(calculatePoissonLambda(instance));
        super.trainOnInstanceImpl(instance);
    }

    public double calculatePoissonLambda(Instance inst) {



        int majClass = getMajorityClass();
        if ((int) inst.classValue() == majClass)
            return 1d;
        //if((int) inst.classValue() == majClass)
            //return 1d;
        return 1d/classSize[(int) inst.classValue()];
        //return classSize[majClass] / (classSize[(int) inst.classValue()] * classSize[(int) inst.classValue()]);
        //return classSize[majClass] / classSize[(int) inst.classValue()];
        //return (classSize[majClass] * classSize[majClass]) / (classSize[(int) inst.classValue()] * classSize[(int) inst.classValue()]);

    }

    protected void updateClassSize(Instance inst) {
        if (this.classSize == null) {
            classSize = new double[inst.numClasses()];

            // <---le19/01/18 modification to start class size as equal for all classes
            Arrays.fill(classSize, 1d / classSize.length);
        }

        for (int i=0; i<classSize.length; ++i) {
            classSize[i] = theta.getValue() * classSize[i] + (1d - theta.getValue()) * ((int) inst.classValue() == i ? 1d:0d);
        }
    }

    public int getMinorityClass() {
        int indexMin = 0;

        for (int i=1; i<classSize.length; ++i) {
            if (classSize[i] <= classSize[indexMin]) {
                indexMin = i;
            }
        }
        return indexMin;
    }



    public int getMajorityClass() {
        int indexMaj = 0;

        for (int i=1; i<classSize.length; ++i) {
            if (classSize[i] > classSize[indexMaj]) {
                indexMaj = i;
            }
        }
        return indexMaj;
    }
}
