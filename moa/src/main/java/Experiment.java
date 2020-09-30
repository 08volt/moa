import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.Classifier;
import moa.core.TimingUtils;
import moa.streams.ConceptDriftStream;

import java.io.IOException;

public class Experiment {

    public Experiment(){
    }

    public void run(int numInstances, boolean isTesting){
        Classifier learner = new MyKNN();
        ConceptDriftStream stream = new ConceptDriftStream();
        stream.positionOption.setValue(2500);
        stream.widthOption.setValue(1);
        stream.streamOption.setValueViaCLIString("generators.AgrawalGenerator");
        stream.driftstreamOption.setValueViaCLIString("ConceptDriftStream -s (generators.AgrawalGenerator -f 2) -d (ConceptDriftStream -s generators.AgrawalGenerator -d (generators.AgrawalGenerator -f 4) -p 2500 -w 10000) -p 2500 -w 1");

        stream.prepareForUse();

        learner.setModelContext(stream.getHeader());
        learner.prepareForUse();

        int numberSamplesCorrect = 0;
        int numberSamples = 0;
        boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();
        long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        while (stream.hasMoreInstances() && numberSamples < numInstances) {


            Instance trainInst = (Instance) stream.nextInstance().getData();
            if (isTesting) {
                if (learner.correctlyClassifies(trainInst)){
                    numberSamplesCorrect++;
                }
            }

            numberSamples++;
            learner.trainOnInstance(trainInst);
        }
        double accuracy = 100.0 * (double) numberSamplesCorrect/ (double) numberSamples;
        double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread()- evaluateStartTime);
        System.out.println(numberSamples + " instances processed with " + accuracy + "% accuracy in "+time+" seconds.");
    }

    public static void main(String[] args) throws IOException {
        Experiment exp = new Experiment();
        exp.run(10000, true);
    }
}