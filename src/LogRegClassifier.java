import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

import Jama.Matrix;

public class LogRegClassifier extends Classifier {

	private String[] output;
	private ArrayList<FeatureHeader> features;
	private ArrayList<ArrayList<DataSet>> data;
	private double[][] theta;
	private double[][] featureValues;
	private double[][] truth;

	public LogRegClassifier(String namesFilepath) {
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
		// for(FeatureHeader feature : features)
		// System.out.println(feature);
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

		// for(double[] row : featureValues){
		// for(double v : row)
		// System.out.print(v + " ");
		// System.out.println();
		// }

		Matrix X = new Matrix(featureValues);
		Matrix Y = new Matrix(truth);

		Matrix thetaMatrix = ((X.transpose().times(X)).inverse().times(X
				.transpose().times(Y)));


		theta = new double[thetaMatrix.getArray().length][];
		theta[0] = new double[]{thetaMatrix.getArray()[0][0]};
		//System.out.println(theta[0][0]);
		for (int index = 1; index < thetaMatrix.getArray().length; index++) {
			for (double b : thetaMatrix.getArray()[index]) {
				if(features.get(index-1).isNumeric())
					theta[index] = new double[]{b};
				else{
					//System.out.println(features.get(index).values.length);
					theta[index] = new double[features.get(index-1).values.length];
					Arrays.fill(theta[index], b);
				}
				// System.out.print(b + " ");
			}
			// System.out.println();
		}

		double highest = 0;

		// for (double f : featureValues[1])
		// System.out.print(f + ", ");
		
		int iterations = 1;
		do {
			double[][] newTheta = new double[theta.length][];
			for (int i = 0; i < theta.length; i++) {
				newTheta[i] = new double[theta[i].length];
				for(int j = 0; j < theta[i].length; j++){
					double change = getChange(i, j);
					//System.out.println(change);
					//System.out.println(i + " " + j);
					newTheta[i][j] = theta[i][j] - (0.1) * change;
				// System.out.println(i + " " + change);
				// if (Math.abs(change) > highest)
				// highest = Math.abs(change);
				//System.out.print(getCost(i) + " ");
				}
			}
			//System.out.println();
			// System.out.println("\n" + getHx(test[0]));
			theta = newTheta;
			//System.out.println("theta2: " + theta[2][0] + " ");
			// System.out.println("\n" + getHx(test[0]));
			//System.out.println(theta[10][0]);
			highest++;
			iterations++;
		} while (highest < 1200);



	}

	public double getCost() {
		double sum = 0;
		double m = featureValues.length;;

		for (int i = 0; i < m; i++) {
			double hx = getHx(featureValues[i]);
			double y = truth[i][0];
			// if (xi == 0 && i == 0) System.out.println(hx + " " + y);
			// System.out.println(hx+ " " + y + " ");
			if (hx != 0 && hx != 1){
				sum += ((-y * Math.log(hx)) - ((1 - y)
						* Math.log(1 - hx)));
			}
		}

		return sum / m;

	}

	// Give feature array and feature output
	public double getChange(int xi, int xj) {
		// number of training values
		double m = 0;
		double sum = 0;

		if (xi == 0) {
			for (int i = 0; i < featureValues.length; i++) {
				double hx = getHx(featureValues[i]);
				double y = truth[i][0];
				//System.out.println("hx: " + hx + " " + y);
				sum += (hx - y);
				m++;
			}
		}

		else {
			//xi -= 1;
			for (int i = 0; i < featureValues.length; i++) {			
				//System.out.println(featureValues[i][0]);
				double hx = getHx(featureValues[i]);
				double y = truth[i][0];
				//if (xi == 1 && i == 0) System.out.println(hx + " " + y);
				if(features.get(xi-1).isNumeric() || featureValues[i][xi] == xj + 1){
					sum += (hx - y) * featureValues[i][xi];
					m++;
				}
				
			}
		}

		// if (xi == 0) System.out.println(sum);

		return sum / m;
	}

	public double getHx(double[] x) {
		//System.out.println("dot: " + (dotProduct(getTheta(x), x)));
		return 1 / (1 + Math.pow(Math.E, -1*(dotProduct(getTheta(x), x))));
	}
	
	public double[] getTheta(double[] x){
		double[] ret = new double[theta.length];
		ret[0] = theta[0][0];
		//System.out.print("ret: " + theta[0][0] + " ");
		for(int i = 1; i < theta.length; i++){
			if(features.get(i-1).isNumeric())
				ret[i] = theta[i][0];
			else
				ret[i] = theta[i][(int)x[i]-1];	
		}
		//System.out.println(ret[1] + " " + ret[2]);
		return ret;
	}

	public double dotProduct(double[] v1, double[] v2) {
		// if (v1.length - 1 != v2.length)
		// return 0;
		double sum = 0;
		for (int i = 0; i < v2.length; i++){
			//System.out.println("summing: " + v1[i] + " " + v2[i]);
			sum += (v1[i] * v2[i]);
		}
		return sum;
	}

	@Override
	public void makePredictions(String testDataFilepath) {

		Scanner file = readFile(testDataFilepath);

		// for (double[] b : truth)
		// System.out.print(b[0] + " ");
		// System.out.println();

		int count = 1;
		int correct = 0;

		while (file.hasNextLine()) {
			String[] line = file.nextLine().split("\\s+");

			DataSet data = new DataSet(line, features, null);
			double[] values = data.matrix;

			double hx = getHx(values);
			
			//System.out.println(hx);

			// System.out.print(p0 + " " + p1 + " ");
			String out = hx < 0.5 ? output[0] : output[1];
			//System.out.println(count + " " + out);
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

	public void generateOutput() {
		PrintWriter writer;
		try {
			writer = new PrintWriter("output.txt");
			for (int i = 0; i < output.length; i++) {
				writer.println(output[i]);
				for (DataSet d : data.get(i))
					writer.println(d);
				writer.println("\n");
			}
			writer.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

}
