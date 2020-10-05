package moa.classifiers.lazy;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.lazy.neighboursearch.KDTree;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;

import java.util.ArrayList;

public class KNNwithClassImbalance extends kNN {

    protected ArrayList<Integer> sizes;
    InstancesHeader context;


    @Override
    public String getPurposeString() {
        return "kNN: class imbalance keeper.";
    }

    @Override
    public void resetLearningImpl() {
        this.sizes = null;
    }

    @Override
    public void setModelContext(InstancesHeader context) {
        super.setModelContext(context);

        this.sizes = new ArrayList<>();
        //this.window = new Instances(context,0); //new StringReader(context.toString())
        //this.window.setClassIndex(context.classIndex());

    }

    private double getClassWeight(int c){
        if(this.sizes.get(c) == 0)
            return 100;
        return (double) getTotalInstances()/this.sizes.get(c);
    }


    private int getTotalInstances(){
        int ris = 0;
        for (int i : sizes) ris += i;
        return ris;
    }

    private int majorClass() {
        int major = 0;
        for (int i = 1; i < sizes.size(); i++)
            if (sizes.get(i) > sizes.get(major))
                major = i;
        return major;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        int c = (int)inst.classValue();
        if (c > C)
            C = c;
        if (this.sizes == null) {
            this.sizes = new ArrayList<>();
        }
        while (this.sizes.size() <= C) {
            this.sizes.add(0);
        }

        if (this.limitOption.getValue() <= this.getTotalInstances()) {
            int major = majorClass();

            for(int i= window.numInstances()-1;i>=0;i--){
                if(window.get(i).classValue() == major) {
                    window.delete(i);
                    this.sizes.set(major, sizes.get(major) - 1);
                    break;
                }
            }


        }
        this.sizes.set(c,sizes.get(c) + 1);
        this.window.add(inst);
    }


    @Override
    public double[] getVotesForInstance(Instance inst) {
        double v[] = new double[C+1];
        try {
            NearestNeighbourSearch search;
            if (this.nearestNeighbourSearchOption.getChosenIndex()== 0) {
                search = new LinearNNSearch(this.window);
            } else {
                search = new KDTree();
                search.setInstances(this.window);
            }
            if (this.window.numInstances()>0) {
                Instances neighbours = search.kNearestNeighbours(inst,Math.min(kOption.getValue(),this.window.numInstances()));
                for(int i = 0; i < neighbours.numInstances(); i++) {
                    int c = (int)neighbours.instance(i).classValue();
                    v[c] += getClassWeight(c);
                }
            }
        } catch(Exception e) {
            //System.err.println("Error: kNN search failed.");
            //e.printStackTrace();
            //System.exit(1);
            return new double[inst.numClasses()];
        }
        return v;
    }

}
