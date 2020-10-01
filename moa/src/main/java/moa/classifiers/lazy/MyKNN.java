package moa.classifiers.lazy;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.capabilities.Capabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.lazy.neighboursearch.*;
import moa.core.Measurement;
import org.kramerlab.autoencoder.math.matrix.Mat;

import java.util.ArrayList;



public class MyKNN extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;

    public IntOption kOption = new IntOption( "n_neighbours", 'n', "numero di vicini", 15, 1, Integer.MAX_VALUE);

    public IntOption limitOption = new IntOption( "limite", 'w', "numero massimo di sample da ricordare", 301, 1, Integer.MAX_VALUE);

    protected int C = 0;

    protected Instances window;

    @Override
    public String getPurposeString() {
        return "MY KNN.";
    }


    @Override
    public void setModelContext(InstancesHeader context) {
        try {
            this.window = new Instances(context,0);
            this.window.setClassIndex(context.classIndex());
        } catch(Exception e) {
            System.err.println("Error: no Model Context available.");
            e.printStackTrace();
            System.exit(1);
        }
    }


    @Override
    public double[] getVotesForInstance(Instance inst) {

        double[] result = new double[this.C + 1];
        try {
            if(window == null || window.numInstances() == 0)
                return new double[inst.numClasses()];
            ArrayList<Instance> vicini = new ArrayList<>();
            double distanza_minima = 0;
            int i = 0;
            DistanceFunction distance = new EuclideanDistance(window);


            while(i< window.numInstances()){ //O(limit)
                if(vicini.size() < kOption.getValue()){
                    vicini.add(window.get(i));
                    double d = distance.distance(inst,window.get(i));
                    distanza_minima = Math.max(d, distanza_minima);
                }
                else{
                    double d = distance.distance(inst,window.get(i));
                    if (d<distanza_minima){
                        //SOSTITUISCI VICINO
                        int max = 0;
                        double dist = distance.distance(inst,vicini.get(0));
                        for(int v = 1; v<vicini.size(); v++){ // O(knn)
                            double newd = distance.distance(inst,vicini.get(v));
                            if(newd>dist){
                                dist = newd;
                                max = v;
                            }
                        }
                        vicini.set(max, window.get(i));
                        distanza_minima = dist;
                    }
                }
                i++;
            }
            for(i = 0; i < vicini.size(); i++)
                result[(int)vicini.get(i).classValue()]++;

        } catch(Exception e) {
            System.out.println("EXCEPT");
            return new double[inst.numClasses()];
        }
        return result;

    }

    @Override
    public void resetLearningImpl() {
        this.window = null;

    }

    // aggiunge sample alla window
    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (inst.classValue() > this.C)
            this.C = (int)inst.classValue();
        if (this.window == null) {
            this.window = new Instances(inst.dataset());
        }
        if (this.limitOption.getValue() <= this.window.numInstances()) {
            this.window.delete(0);
        }
        this.window.add(inst);

    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {

    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

}
