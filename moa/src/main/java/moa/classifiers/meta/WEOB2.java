package moa.classifiers.meta;

import com.yahoo.labs.samoa.instances.Instance;

/**
 * Weighted Ensamble Online Bagging 2
 *
 *
 * <p>Resampling-Based Ensemble Methods for Online Class Imbalance Learning
 * Shuo Wang and Leandro L. Minku and Xin Yao
 * 2015</p>
 *
 *
 */

public class WEOB2 extends WEOB1{

    @Override
    public double[] getVotesForInstance(Instance inst) {
        double[] oobVotes = oob.getVotesForInstance(inst);
        double[] uobVotes = uob.getVotesForInstance(inst);

        if(classRecallUOB == null){
            oob.randomSeedOption.setValue(this.randomSeedOption.getValue());
            uob.randomSeedOption.setValue(this.randomSeedOption.getValue());
            classRecallOOB = new SmoothedRecall(inst.numClasses(),recalltheta.getValue(),SmoothedRecallWindowSizeOption.getValue());
            classRecallUOB = new SmoothedRecall(inst.numClasses(),recalltheta.getValue(),SmoothedRecallWindowSizeOption.getValue());
        }

        double uobGmean = classRecallUOB.getGmean();
        double oobGmean = classRecallOOB.getGmean();

        if (oobGmean>uobGmean){
            return oobVotes;
        }
        return uobVotes;


    }
}
