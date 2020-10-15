package moa.classifiers.meta.ELM;

import java.io.NotSerializableException;
import java.io.Serializable;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrices;
import no.uib.cipr.matrix.NotConvergedException;
import no.uib.cipr.matrix.SVD;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

public class ELM extends AbstractClassifier implements
		WeightedInstancesHandler, TechnicalInformationHandler {

	/**
	 * author ye_qiangsheng on 2014/10/03
	 */
	private static final long serialVersionUID = -7834549585915326436L;

	/* 随机种子 用于初始化矩阵 */
	private int m_randomSeed = 1;
	/* 激活函数 */
	private String m_activeFunction = "sig";
	/* 隐藏层 神经元个数 */
	private int m_numberofHiddenNeurons = 20;
	/**/
	private Selm m_elm = null;
	/**
	 * 归一化数组，标准化 数值型属性，取最大值 Max 减去最小值 Min 为除数，原值 x 减最小值 Min 为被除数
	 * 数组取两行，第一行为最大值，第二行为最小值
	 */
	private double[][] m_nomalization;

	@Override
	public void buildClassifier(Instances instances) throws Exception {
		// TODO Auto-generated method stub

		// remove instances with missing class
		Instances data = new Instances(instances);
		data.deleteWithMissingClass();

		int length = instances.numAttributes();
		m_nomalization = new double[2][length];
		for (int i = 0; i < length; i++) {
			if (instances.attribute(i).isNumeric()) {
				m_nomalization[0][i] = instances.attributeStats(i).numericStats.max;
				m_nomalization[1][i] = instances.attributeStats(i).numericStats.min;
			}
		}
		// System.out.println(java.util.Arrays.toString(m_nomalization[0]));
		// System.out.println(java.util.Arrays.toString(m_nomalization[1]) +
		// "\n");

		int elm_type = 0;
		if (instances.classAttribute().isNominal()) {
			elm_type = 1;
		}
		int classIndex = instances.classIndex();
		m_elm = new Selm(elm_type, m_numberofHiddenNeurons, m_activeFunction,
				m_randomSeed, classIndex);
		m_elm.train(instances, m_nomalization);
		// System.out.println("training over" +
		// elm_type+m_numberofHiddenNeurons+m_function+m_randomSeed+classIndex);
	}

	/**
	 * Classifies a given instance.
	 * 
	 * @param instance
	 *            the instance to be classified
	 * @return index of the predicted class
	 */
	@Override
	public double classifyInstance(Instance instance) {
		for (int j = 0; j < instance.numAttributes(); j++) {
			if (instance.attribute(j).isNumeric()) {
				instance.setValue(j, (instance.value(j) - m_nomalization[1][j])
						/ (m_nomalization[0][j] - m_nomalization[1][j]));
			}
		}
		instance.setValue(instance.classIndex(), instance.value(0));
		int columns = instance.numAttributes() - 1;
		double[][] predict = new double[1][columns];
		for (int i = 1; i < instance.numAttributes(); i++) {
			predict[0][i - 1] = instance.value(i);
		}
		// System.out.println(instance.numAttributes() +" -- "
		// +instance.classIndex());
		// System.out.println(java.util.Arrays.toString(predict[0]));
		double[] result = m_elm.testOut(predict, 1, columns);
		//System.out.println(java.util.Arrays.toString(result));
		if (instance.attribute(instance.classIndex()).isNominal()) {
			return result[0];
		}
		return result[0]* (m_nomalization[0][instance.classIndex()] - 
				m_nomalization[1][instance.classIndex()])+ m_nomalization[1][instance.classIndex()];
	}

	/**
	 * Returns a description of the classifier.
	 * 
	 * @return a description of the classifier as a string.
	 */
	@Override
	public String toString() {
		return "Extreme Learning Mechine";
	}

	/**
	 * Returns a string describing classifier
	 * 
	 * @return a description suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {

		return "Class for building and using a ELM classifier;"
				+ " this code modified by Ye Qiangsheng, the source code author is DongLi (follow website) \n\n"
				+ getTechnicalInformation().toString();
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		TechnicalInformation result;

		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "Author: Guanbinhuang");
	    result.setValue(Field.YEAR, "Year: 2004");
	    result.setValue(Field.TITLE,
	        " http://www.ntu.edu.sg/home/egbhuang/elm_codes.html ");
		return result;
	}

	 /**
	   * Returns default capabilities of the classifier.
	   * 
	   * @return the capabilities of this classifier
	   */
	  @Override
	  public Capabilities getCapabilities() {
	    Capabilities result = super.getCapabilities();
	    result.disableAll();

	    // attributes
	    result.enable(Capability.NOMINAL_ATTRIBUTES);
	    result.enable(Capability.NUMERIC_ATTRIBUTES);
	   // result.enable(Capability.DATE_ATTRIBUTES);
	    result.enable(Capability.STRING_ATTRIBUTES);
	    result.enable(Capability.RELATIONAL_ATTRIBUTES);
	    //result.enable(Capability.MISSING_VALUES);

	    // class
	    result.enable(Capability.NOMINAL_CLASS);
	    result.enable(Capability.NUMERIC_CLASS);
	   // result.enable(Capability.DATE_CLASS);
	    //result.enable(Capability.MISSING_CLASS_VALUES);

	    // instances
	    result.setMinimumNumberInstances(0);

	    return result;
	  }
	/**
	 * Returns an enumeration describing the available options..
	 * 
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {

		Vector<Option> newVector = new Vector<Option>(14);

		newVector.addElement(new Option(
				"\t number of HiddenNeurons , default 20. \n"
				+ "\tbigger than 0, more bigger more better but slowly.",
				"N", 1, "-N number of HiddenNeurons"));
		newVector.addElement(new Option(
				"\t random seed ,default 1\n",
				"R", 1, "-R random seed "));
		newVector.addElement(new Option(
				"\t active function \n", "F", 1, "-N active function"));

		newVector.addAll(Collections.list(super.listOptions()));

		return newVector.elements();
	}

	/**
	 * 
	 */
	@Override
	public void setOptions(String[] options) throws Exception {

		String numberofHiddenNeuronsString = Utils.getOption('N', options);
		if (numberofHiddenNeuronsString.length() != 0) {
			m_numberofHiddenNeurons = Integer
					.parseInt(numberofHiddenNeuronsString);
		} else {
			m_numberofHiddenNeurons = 20;
		}

		String randomSeedString = Utils.getOption('R', options);
		if (randomSeedString.length() != 0) {
			m_randomSeed = Integer.parseInt(randomSeedString);
		} else {
			m_randomSeed = 1;
		}

		String activeFunctionString = Utils.getOption('F', options);
		if (activeFunctionString.length() != 0) {
			m_activeFunction = activeFunctionString;
		} else {
			m_activeFunction = "sig";
		}

		super.setOptions(options);
	}

	/**
	 * 
	 */
	@Override
	public String[] getOptions() {

		Vector<String> options = new Vector<String>(3);

		options.add("-N");
		options.add("" + m_numberofHiddenNeurons);
		options.add("-R");
		options.add("" + m_randomSeed);
		options.add("-F");
		options.add("" + m_activeFunction);

		Collections.addAll(options, super.getOptions());

		return options.toArray(new String[0]);
	}
	  
	/**
	 * Main method for testing this class.
	 * 
	 * @param argv
	 *            the options
	 */
	public static void main(String[] argv) {
		runClassifier(new ELM(), argv);
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String randomSeedTipText() {
		return "the random seed to initial the matrix";
	}
	
	public int getRandomSeed() {
		return m_randomSeed;
	}

	public void setRandomSeed(int m_randomSeed) {
		this.m_randomSeed = m_randomSeed;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String activeFunctionTipText() {
		return "the active function";
	}
	
	public String getActiveFunction() {
		return m_activeFunction;
	}

	public void setActiveFunction(String m_function) {
		this.m_activeFunction = m_function;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String numberofHiddenNeuronsTipText() {
		return "number of hidden neurons ";
	}
	
	public int getNumberofHiddenNeurons() {
		return m_numberofHiddenNeurons;
	}

	public void setNumberofHiddenNeurons(int m_numberofHiddenNeurons) {
		this.m_numberofHiddenNeurons = m_numberofHiddenNeurons;
	}

}
