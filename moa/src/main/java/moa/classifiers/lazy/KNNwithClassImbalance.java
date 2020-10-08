package moa.classifiers.lazy;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.lazy.neighboursearch.KDTree;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import org.apache.commons.math3.optim.SimplePointChecker;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class KNNwithClassImbalance extends kNN {

    protected ArrayList<Integer> sizes;

    public FloatOption purifyFreq = new FloatOption( "purify", 'p', "purify the window with this frequence", 0.2, 0, 1);
    public IntOption Kpurity = new IntOption( "KNNforpurity", 'v', "minimum number of neighbours of minority to purify", 3, 1, Integer.MAX_VALUE);
    protected Random r;


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
        r = new Random();

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

    //DA RIVEDERE
    private void purify(){
        int major = majorClass();
        ArrayList<Integer> impurity = new ArrayList<>();
        for(int i = 0; i< window.numInstances();i++){
            if(window.get(i).classValue() == major){
                Instances neighbours = findKNNeighbours(window.get(i),Kpurity.getValue());
                if(neighbours == null)
                    return;
                int cntMajor = 0;
                for(int n = 0; n < neighbours.numInstances(); n++) {
                    if( (int)neighbours.get(n).classValue() == major)
                        cntMajor++;

                }
                if(cntMajor < (Kpurity.getValue()+1)/sizes.size())
                    impurity.add(i);

            }else{

                Instances neighbours = findKNNeighbours(window.get(i),Kpurity.getValue());
                if(neighbours == null)
                    return;
                int cntMajor = 0;
                for(int n = 0; n < neighbours.numInstances(); n++) {
                    if( (int)neighbours.get(n).classValue() == major)
                        cntMajor++;

                }
                if(cntMajor < (Kpurity.getValue()+1)/sizes.size()){
                    for(int n = 0; n < neighbours.numInstances(); n++)
                        if(neighbours.get(n).classValue() == major)
                            impurity.add(window.indexOf(neighbours.instance(n)));
                }

            }
        }

        if(impurity.size() > 0) {
            System.out.println("number of classes: " + sizes.size());
            System.out.println("major: " + major);
            System.out.println(impurity.size());
            for (int s : sizes)
                System.out.println("size " + s);
        }
        Collections.sort(impurity);
        for(int m = impurity.size()-1;m>=0;m--)
            window.delete(impurity.get(m));
        sizes.set(major, sizes.get(major) - impurity.size());



    }

    public double imbalance(){
        if(sizes.size() < 2 || getTotalInstances() < Kpurity.getValue())
            return 0;

        double balance = 1d;



        for(int s = 0; s<sizes.size();s++)
            balance *= ((double) sizes.get(s))/((double) window.size());

        return 1d - (balance/(double)(sizes.size() * sizes.size()));
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
//        double imb = imbalance();
//        if( this.purifyFreq.getValue() > r.nextDouble() && imb > 0.0) {
//            System.out.println("size0: " + sizes.get(0) + " size1: " + sizes.get(1) + " imbalance= " + imb);
//            purify();
//        }


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
            //this.sizes.set((int)window.get(window.numInstances()-1).classValue(), sizes.get((int)window.get(window.numInstances()-1).classValue()) - 1);
            //window.delete(window.numInstances()-1);
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

    private Instances findKNNeighbours(Instance inst,int kvalue){
        try {
            NearestNeighbourSearch search;
            if (this.nearestNeighbourSearchOption.getChosenIndex()== 0) {
                search = new LinearNNSearch(this.window);
            } else {
                search = new KDTree();
                search.setInstances(this.window);
            }
            if (this.window.numInstances()>0) {
                return search.kNearestNeighbours(inst,Math.min(kvalue,this.window.numInstances()));
            }
        } catch(Exception e) {
            //System.err.println("Error: kNN search failed.");
            //e.printStackTrace();
            //System.exit(1);
            return null;
        }
        return null;
    }


    @Override
    public double[] getVotesForInstance(Instance inst) {
        double v[] = new double[C+1];
        Instances neighbours = findKNNeighbours(inst, kOption.getValue());
        if(neighbours == null)
            return new double[inst.numClasses()];

        for(int i = 0; i < neighbours.numInstances(); i++) {
            int c = (int)neighbours.instance(i).classValue();
            v[c] += getClassWeight(c);
        }

        return v;


    }

}
