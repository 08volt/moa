package moa.classifiers.meta;

import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;

import java.util.Arrays;

public class SmoothedRecall {

    private final double[][] recalls;
    private int windowSize;
    private final double[] smoothedRecalls;
    private final double theta;
    private int windowPos;

    public SmoothedRecall(int numClasses, double theta, int windowSize){

        this.recalls = new double[numClasses][windowSize];
        this.windowPos = -1;
        this.windowSize = 0;
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
        //newWP -> cursor for the position of the new instance on the window
        int newWP = windowPos + 1;
        if(newWP >= recalls[classValue].length)
            newWP = 0;
        // remove the oldest recalls if the window is full
        for(int c = 0; c< smoothedRecalls.length; c++) {
            //smoothedRecall become the sum of the saved recall
            smoothedRecalls[c] = smoothedRecalls[c] * windowSize;
            smoothedRecalls[c] = smoothedRecalls[c] - recalls[c][newWP];
        }
        // increase the size of the window if its not at the maximum
        this.windowSize = Math.min(windowSize + 1, recalls[classValue].length);
        //add the new recalls to the window
        for(int c = 0; c < smoothedRecalls.length; c++) {
            if (c == classValue) {
                recalls[c][newWP] = r;
            } else {
                recalls[c][newWP] = recalls[c][windowPos];
            }
            smoothedRecalls[c] = (smoothedRecalls[c] + recalls[c][newWP]) / windowSize;
        }
        //update the cursor window position
        windowPos = newWP;
    }


}
