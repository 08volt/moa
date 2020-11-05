package moa;

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.Classifier;
import moa.classifiers.lazy.MyKNN;
import moa.core.TimingUtils;
import moa.dabrze.streams.generators.ImbalancedDriftGenerator;
import moa.streams.ArffFileStream;
import moa.streams.ConceptDriftStream;
import moa.streams.ImbalancedStream;
import moa.tasks.StandardTaskMonitor;
import moa.tasks.TaskMonitor;
import moa.tasks.WriteStreamToARFFFile;
import org.kramerlab.bmad.general.Tuple;
//import moa.classifiers.lazy.MyKNN;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class Experiment {

    public Experiment(){
    }

    public void run(){
        ArrayList<String> driftsByName = new ArrayList<>();
//        driftsByName.add("appearing-minority");
//        driftsByName.add("disappearing-minority");
//        driftsByName.add("jitter");
//        driftsByName.add("clusters-movement");
//        driftsByName.add("appearing-clusters");
//        driftsByName.add("splitting-clusters");
          driftsByName.add("borderline");
//        driftsByName.add("shapeshift");
//        driftsByName.add("minority-share");

        String[] speeds = {"periodic"};//"sudden","incremental",
        int[][] startend = new int[3][2];
        startend[1][0] = 50000;
        startend[1][1] = 50000;
        startend[0][0] = 45000;
        startend[0][1] = 55000;
        startend[2][0] = 45000;
        startend[2][1] = 55000;

        int[] minority_shares = {2,3,4};

        for(String name: driftsByName){
            for (int minority: minority_shares){
                try {

                    Path path = Paths.get("/Users/08volt/Desktop/Drifts/" + name + "/" + minority);

                    //java.nio.file.Files;
                    Files.createDirectories(path);

                    System.out.println("Directory is created!");

                } catch (IOException e) {

                    System.err.println("Failed to create directory!" + e.getMessage());

                }

                for(int s = 0; s < speeds.length; s++) {

                    String cli = "moa.dabrze.streams.generators.ImbalancedDriftGenerator -d " +
                        name +
                        "/" + speeds[s] +
                        ",start=" + startend[s][0] +
                        ",end=" + startend[s][1] +
                        ",value-start=0.0,value-end=1.0" +
                        " -n 2 -m 0." + minority +
                        " -s 0.5 -b 0.5";

                WriteStreamToARFFFile file = new WriteStreamToARFFFile();
                file.arffFileOption.setValue("/Users/08volt/Desktop/Drifts/" + name +
                        "/" + minority + "/" + name + speeds[s] + ".arff");
                file.streamOption.setValueViaCLIString(cli);
                file.maxInstancesOption.setValue(100000);
                file.prepareForUse();
                System.out.println(cli);
                file.doTask();
                System.out.println(minority + " - " + name + " - " + speeds[s] + " arff");
            }


            }


        }
//
//        ConceptDriftStream stream = new ConceptDriftStream();
//        stream.positionOption.setValue(2500);
//        stream.widthOption.setValue(1);
//        stream.streamOption.setValueViaCLIString("generators.AgrawalGenerator");
//        stream.driftstreamOption.setValueViaCLIString("ConceptDriftStream -s (generators.AgrawalGenerator -f 2) -d " +
//                "(ConceptDriftStream -s generators.AgrawalGenerator -d " +
//                "(generators.AgrawalGenerator -f 4) -p 2500 -w 10000) -p 2500 -w 1");
//
//        stream.prepareForUse();
//
//        learner.setModelContext(stream.getHeader());
//        learner.prepareForUse();
//
//        int numberSamplesCorrect = 0;
//        int numberSamples = 0;
//        boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();
//        long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
//        while (stream.hasMoreInstances() && numberSamples < numInstances) {
//
//
//            Instance trainInst = (Instance) stream.nextInstance().getData();
//            if (isTesting) {
//                if (learner.correctlyClassifies(trainInst)){
//                    numberSamplesCorrect++;
//                }
//            }
//
//            numberSamples++;
//            learner.trainOnInstance(trainInst);
//        }
//        double accuracy = 100.0 * (double) numberSamplesCorrect/ (double) numberSamples;
//        double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread()- evaluateStartTime);
//        System.out.println(numberSamples + " instances processed with " + accuracy + "% accuracy in "+time+" seconds.");
    }

    public static void main(String[] args) throws IOException {
        Experiment exp = new Experiment();
        exp.run();
        //exp.run(100000, true);
    }
}