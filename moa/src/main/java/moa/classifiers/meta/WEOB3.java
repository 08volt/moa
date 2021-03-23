package moa.classifiers.meta;

import com.yahoo.labs.samoa.instances.Instance;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;
import java.util.Collections;

public class WEOB3 extends WEOB1{



    @Override
    public double[] getVotesForInstance(Instance inst) {
        double[] oobVotes = oob.getVotesForInstance(inst);
        double[] uobVotes = uob.getVotesForInstance(inst);
        double max = -1;

        if(oobVotes == null) {
            oob.randomSeedOption.setValue(this.randomSeedOption.getValue());
            uob.randomSeedOption.setValue(this.randomSeedOption.getValue());
        }

            for(int c = 0; c<oob.classSize.length; c++)
            max = Math.max(max, oob.classSize[c]);
        double min = max;
        for(int c = 0; c<oob.classSize.length; c++)
            min = Math.min(min, oob.classSize[c]);

        if (oob.classSize[(int)inst.classValue()]/min>4){
            return uobVotes;
        }
        return oobVotes;


    }
}
