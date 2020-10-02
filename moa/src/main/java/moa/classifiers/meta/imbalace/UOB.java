package moa.classifiers.meta.imbalace;

import com.yahoo.labs.samoa.instances.Instance;

public class UOB extends OOB {
    public double calculatePoissonLambda(Instance inst) {

        if (majorities.contains((int)inst.classValue()))
            return 1 - classSize[(int)inst.classValue()];
        return 1d;

    }
}
