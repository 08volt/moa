package moa.classifiers.meta.imbalace;

import com.yahoo.labs.samoa.instances.Instance;

public class WEOB2 extends WEOB1{

    @Override
    public double[] getVotesForInstance(Instance inst) {
        double[] oobVotes = oob.getVotesForInstance(inst);
        double[] uobVotes = uob.getVotesForInstance(inst);

        double uobGini = calcGini(classRecallUOB);
        double oobGini = calcGini(classRecallOOB);

        if (oobGini>uobGini){
            return oobVotes;
        }
        return uobVotes;


    }
}
