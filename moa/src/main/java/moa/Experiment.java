package moa;

import com.yahoo.labs.samoa.instances.*;
import moa.classifiers.Classifier;
import moa.classifiers.lazy.MyKNN;
import moa.classifiers.lazy.neighboursearch.HVDMDistance;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.core.TimingUtils;
import moa.dabrze.streams.generators.ImbalancedDriftGenerator;
import moa.streams.ArffFileStream;
import moa.streams.ConceptDriftStream;
import moa.streams.ImbalancedStream;
import moa.tasks.StandardTaskMonitor;
import moa.tasks.TaskMonitor;
import moa.tasks.WriteStreamToARFFFile;
import org.kramerlab.bmad.general.Tuple;
import scala.Int;
//import moa.classifiers.lazy.MyKNN;

import java.io.*;
import java.lang.reflect.Array;
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
//        driftsByName.add("borderline");
//        driftsByName.add("shapeshift");
//        driftsByName.add("minority-share");

        String[] speeds = {"incremental","sudden","periodic"};
        int[][] startend = new int[3][2];
        startend[1][0] = 50000;
        startend[1][1] = 50000;
        startend[0][0] = 45000;
        startend[0][1] = 55000;
        startend[2][0] = 45000;
        startend[2][1] = 55000;

        int[] minority_shares = {4,2,3,1};

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
                        ",start=0,end=1000000,value-start=0.0,value-end=1.0" +
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
        //exp.run();
        System.out.println(args[0]+ "preprocessing_KDDCup.csv");
        exp.kneighbours(args[0]);

        //exp.run(100000, true);
    }

    public void kneighbours(String path){
        String dir = path;//""/Users/08volt/Desktop/StremingML/MyMoa/moa/moa/src/main/java/moa/";
        String csvFile = dir + "preprocessing_KDDCup.csv";
        Integer[] cat_idx = {2,3,4,7,12};
        ArrayList<Integer> cat_idxx = new ArrayList<>();
        Collections.addAll(cat_idxx, cat_idx);
        BufferedReader br = null;
        String line = "";
        String cvsSplitBy = ",";

        Attribute[] attributes = new Attribute[40];
        for(int a = 0; a<40; a++)
        {
            attributes[a] = new Attribute();

        }


        Instances df = new Instances("KDDCup",attributes,500000);
        int cnt = 0;
        try {

            br = new BufferedReader(new FileReader(csvFile));

            while ((line = br.readLine()) != null) {


                String[] l = line.split(cvsSplitBy);
                InstanceImpl inst = new InstanceImpl(l.length -1);

                for (int i=0; i< l.length-1; i++)
                    inst.setValue(i, Double.parseDouble(l[i]));


                inst.setClassValue(Double.parseDouble(l[l.length-1]));
                //if(inst.classValue() == 0.0)
                   //System.out.println(inst.value(0));
                df.add(inst);
                cnt++;
                if(cnt%10000 == 0)
                    System.out.println(cnt);

                //if(cnt >= 400000)
                  //  break;

//                    break;

                //System.out.println(cnt);
            }


            System.out.println("start NN search");
            NearestNeighbourSearch search = new LinearNNSearch(df);

            search.setDistanceFunction(new HVDMDistance(df, cat_idxx));
            Instances kn;
            for(int i=0; i<df.numInstances(); i++) {
                kn = search.kNearestNeighbours(df.get(i), 6);
                int[] res = new int[5];
                for( int k = 0; k<5; k++){
                    res[k] = (int)kn.get(k).value(0);

                }

                System.out.println(df.get(i).classValue());
                System.out.println(Arrays.toString(res));

                File f1 = new File("KDDCup_results.txt");
                if(!f1.exists()) {
                    f1.createNewFile();
                }
                FileWriter fileWritter = new FileWriter(f1.getName(),true);
                BufferedWriter bw = new BufferedWriter(fileWritter);
                bw.write(df.get(i).value(0) + " : "+ Arrays.toString(res) + "\n");
                bw.close();


            }





        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}