package moa.classifiers.meta;

import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;

import java.util.Arrays;

public class SmoothedRecall {

    private final double[][] recalls;
    private final int windowSize;
    private final double[] smoothedRecalls;
    private final double theta;
    private int windowPos;

    public SmoothedRecall(int numClasses, double theta, int windowSize){

        this.recalls = new double[numClasses][windowSize];
        this.windowPos = -1;
        this.windowSize = windowSize;
        this.smoothedRecalls = new double[numClasses];
        this.theta = theta;
        for(int i = 0; i<numClasses; i++){
            Arrays.fill(this.recalls[i],-1d);
        }
    }

    public double getGmean(){
        if(recalls == null)
            return 1d;
        double gini = 1d;
        for (double recall : smoothedRecalls) {
            gini *= recall;
        }

        return Math.pow(gini,1d/(double) smoothedRecalls.length);
    }

    public void insertPrediction(int classValue, boolean pred){
        double r;

        //array initialization
        if( this.windowPos == -1){
            r = pred ? 1d:0d;
            for(int c = 0; c< recalls.length; c++)
                if( c == classValue) {
                    recalls[c][0] = r;
                    smoothedRecalls[c] = r;
                }
                else {
                    recalls[c][0] = 0;
                    smoothedRecalls[c] = 0;
                }
            windowPos = 0;
            return;

        }


        r = this.theta * recalls[classValue][windowPos] + (1d - this.theta) * (pred ? 1d:0d);

        //newWP -> position of the new instance on the window
        //timesteps -> timesteps on the window until this instance
        int newWP = windowPos + 1;
        int timesteps = newWP;
        if(newWP >= windowSize) {
            newWP = 0;
            timesteps = windowSize;
        }else if(recalls[classValue][newWP] >= 0){
            timesteps = windowSize;
        }




        for(int c = 0; c< smoothedRecalls.length; c++) {

            //smoothedRecall become the sum of the saved recall
            smoothedRecalls[c] = smoothedRecalls[c] * timesteps;
            if (timesteps == windowSize)
                //if window is full
                smoothedRecalls[c] = smoothedRecalls[c] - recalls[c][newWP];
            if(c == classValue) {
                recalls[c][newWP] = r;
                smoothedRecalls[c] = (smoothedRecalls[c] + r) / Math.min(timesteps + 1, windowSize);
            }
            else {
                recalls[c][newWP] = recalls[c][windowPos];
                smoothedRecalls[c] = (smoothedRecalls[c] + recalls[c][windowPos]) / Math.min(timesteps + 1, windowSize);
            }
        }
        windowPos = newWP;


    }


}
