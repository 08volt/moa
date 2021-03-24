/**
 * Improved Undersampling Online Bagging
 *
 *
 * <p>Resampling-Based Ensemble Methods for Online Class Imbalance Learning
 * Shuo Wang and Leandro L. Minku and Xin Yao
 * 2015</p>
 *
 *
 */

package moa.classifiers.meta;

import com.yahoo.labs.samoa.instances.Instance;

public class ImprovedUOB extends ImprovedOOB {

	@Override
	public String getPurposeString() {
		return "Undersampling on-line bagging of Wang et al IJCAI 2016.";
	}
	
	public ImprovedUOB() {
		super();
	}

	@Override
	public double calculatePoissonLambda(Instance inst) {
		int minClass = getMinorityClass();
		
		return classSize[minClass] / classSize[(int) inst.classValue()];
		
	}

	// find the index of the class with the smaller size
	protected int getMinorityClass() {
		int indexMin = 0;

		for (int i=1; i<classSize.length; ++i) {
			if (classSize[i] <= classSize[indexMin]) {
				indexMin = i;
			}
		}
		return indexMin;
	}

}