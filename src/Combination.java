import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Scanner;

import Jama.Matrix;

public class Combination extends Classifier {

	private String[] output;
	private ArrayList<FeatureHeader> features;
	private ArrayList<ArrayList<DataSet>> data;
	private double[] theta;
	private double[][] featureValues;
	private double[][] featureValues0;
	private double[][] featureValues1;
	private double[][] truth;
	private double output0P;

	public Combination(String namesFilepath) {
		super(namesFilepath);
		features = new ArrayList<FeatureHeader>();
		data = new ArrayList<ArrayList<DataSet>>();

		Scanner file = readFile(namesFilepath);
		output = file.nextLine().split("\\s+");
		for (String s : output)
			data.add(new ArrayList<DataSet>());

		file.nextLine();
		while (file.hasNextLine()) {
			String[] data = file.nextLine().split("\\s+");
			features.add(new FeatureHeader(data));
		}
	}

	@Override
	public void train(String trainingDataFilpath) {
		Scanner file = readFile(trainingDataFilpath);

		while (file.hasNextLine()) {
			String[] data = file.nextLine().split("\\s+");
			String result = data[data.length - 1];
			for (int i = 0; i < output.length; i++)
				if (result.equals(output[i])) {
					this.data.get(i).add(new DataSet(data, features, output));
				}
		}

		int rows0 = this.data.get(0).size();
		int rows1 = this.data.get(1).size();
		int cols = this.features.size();

		featureValues0 = new double[rows0][cols + 1];
		featureValues1 = new double[rows1][cols + 1];

		featureValues = new double[rows0 + rows1][cols + 1];
		truth = new double[rows0 + rows1][1];

		for (int i = 0; i < rows0; i++) {
			featureValues[i] = this.data.get(0).get(i).matrix;
			truth[i] = this.data.get(0).get(i).outputIndex;
		}

		for (int i = 0; i < rows1; i++) {
			featureValues[i + rows0] = this.data.get(1).get(i).matrix;
			truth[i + rows0] = this.data.get(1).get(i).outputIndex;
		}
		
		
		for (double[] row : featureValues) {
			for (int i = 0; i < cols; i++) {
				if (!features.get(i).isNumeric()) {
					features.get(i).probabilities[0][(int) row[i + 1] - 1] = findProbability(
							i, row[i + 1], 0);
					features.get(i).probabilities[1][(int) row[i + 1] - 1] = findProbability(
							i, row[i + 1], 1);
				}
			}
		}

		Matrix X = new Matrix(featureValues);
		Matrix Y = new Matrix(truth);

		Matrix thetaMatrix = ((X.transpose().times(X)).inverse().times(X
				.transpose().times(Y)));

		theta = new double[thetaMatrix.getArray().length];
		int index = 0;
		for (double[] a : thetaMatrix.getArray()) {
			for (double b : a) {
				theta[index] = b;
				index++;
			}
		}

		int iterations = 1;
		do {
			double[] newTheta = new double[theta.length];
			for (int i = 0; i < theta.length; i++) {
				double change = getChange(i);
				newTheta[i] = theta[i] - (0.1) * change;
			}
			theta = newTheta;
			iterations++;
		} while (iterations < 2000);

	}
	
	public double[] getMeanVariance(int index, int output) {
		double[] data = new double[2];
		data[0] = 0;
		data[1] = 0;
		int total = 0;

		double[][] values = output == 0 ? featureValues0 : featureValues1;

		for (double[] row : values) {
			data[0] += row[index];
			total++;
		}

		data[0] /= total;

		for (double[] row : values)
			data[1] += Math.pow(row[index] - data[0], 2);

		data[1] /= total;
		return data;
	}

	public double findProbability(int index, double value, int output) {

		int total = 0;
		// Adding 1 as smoothing factor so Jl is just j
		int j = features.get(index).values.length;
		double[][] values = output == 0 ? featureValues0 : featureValues1;

		for (double[] row : values) {
			if (row[index + 1] == value)
				total++;
		}

		return (double) (total + 1)
				/ (double) (this.data.get(output).size() + j);

	}
	
	public double calculateProbability(int i, double value, int output) {
		if (features.get(i).isNumeric()) {
			double[] mv = getMeanVariance(i + 1, output);
			// System.out.print("mean: " + mv[0] + " variance: " + mv[1]);
			double pow = -(Math.pow(value - mv[0], 2)) / (2 * mv[1]);
			double prob = (Math.pow(1, pow) / Math.sqrt((2 * Math.PI * mv[1])));
			// System.out.println(" " + prob);
			return prob;
		}

		return features.get(i).probabilities[output][(int) value - 1];
		// return findProbability(i, value, output);

	}

	public double getChange(int xi) {
		// number of training values
		double m = featureValues.length;
		double sum = 0;

		if (xi == 0) {
			for (int i = 0; i < m; i++) {
				double hx = getHx(featureValues[i]);
				double y = truth[i][0];
				sum += (hx - y);
			}
		}

		else {
			for (int i = 0; i < m; i++) {
				double hx = getHx(featureValues[i]);
				double y = truth[i][0];
				sum += (hx - y) * featureValues[i][xi];
			}
		}
		return sum / m;
	}

	public double getHx(double[] x) {
		return 1 / (1 + Math.pow(Math.E, -1 * (dotProduct(theta, x))));
	}

	public double dotProduct(double[] v1, double[] v2) {
		double sum = 0;
		for (int i = 0; i < v2.length; i++)
			sum += (v1[i] * v2[i]);
		return sum;
	}

	@Override
	public void makePredictions(String testDataFilepath) {

		Scanner file = readFile(testDataFilepath);

		int count = 1;
		int correct = 0;

		while (file.hasNextLine()) {
			String[] line = file.nextLine().split("\\s+");

			DataSet data = new DataSet(line, features, null);
			double[] values = data.matrix;
			
			double p0 = output0P;
			double p1 = 1 - output0P;

			for (int i = 0; i < features.size(); i++) {
				p0 *= calculateProbability(i, values[i + 1], 0);
				p1 *= calculateProbability(i, values[i + 1], 1);
			}


			double hx = getHx(values);

			String out = (hx < 0.5) ? output[0] : output[1];
			// System.out.println(count + " " + out);
			if (out.equals(data.output))
				correct++;
			count++;
		}

		System.out.println((double) correct / (count - 1));

	}

	public Scanner readFile(String filename) {
		Scanner s;
		try {
			s = new Scanner(new File(filename));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			s = null;
		}
		return s;
	}

}
