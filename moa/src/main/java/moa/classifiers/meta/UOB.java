package moa.classifiers.meta;

import com.yahoo.labs.samoa.instances.Instance;

/**
 * Undersampling Online Bagging
 *
 *
 * <p>A learning framework for online class imbalance learning
 * Shuo Wang and Leandro L. Minku and Xin Yao
 * 2013</p>
 *
 *
 */

public class UOB extends OOB {

    //UOB lambda
    @Override
    public double calculatePoissonLambda(Instance inst) {
        // decrease the lambda if the class is in the majorities group
        if (majorities.contains((int)inst.classValue()))
            return 1 - classSize[(int)inst.classValue()];
        return 1d;

    }


}
