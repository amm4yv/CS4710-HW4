import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Scanner;

import Jama.Matrix;


public class CustomClassifier extends Classifier {
	
	private String[] output;
	private ArrayList<FeatureHeader> features;
	private ArrayList<ArrayList<DataSet>> data;
	private double[] theta;
	private double[][] featureValues;
	private double[][] truth;

	public CustomClassifier(String namesFilepath) {
		super(namesFilepath);
		features = new ArrayList<FeatureHeader>();
		data = new ArrayList<ArrayList<DataSet>>();
		
		Scanner file = readFile(namesFilepath);
		output = file.nextLine().split("\\s+");
		for(String s : output)
			data.add(new ArrayList<DataSet>());
		
		file.nextLine();
		while(file.hasNextLine()) {
			String[] data = file.nextLine().split("\\s+");
			features.add(new FeatureHeader(data));
		}
//		for(FeatureHeader feature : features)
//			System.out.println(feature);
	}

	@Override
	public void train(String trainingDataFilpath) {
		Scanner file = readFile(trainingDataFilpath);
		
			
		while(file.hasNextLine()) {
			String[] data = file.nextLine().split("\\s+");
			String result = data[data.length-1];
			for(int i = 0; i < output.length; i++)
				if(result.equals(output[i]))
					this.data.get(i).add(new DataSet(data, features, output));
		}
		
		int rows0 = this.data.get(0).size();
		int rows1 = this.data.get(1).size();
		int cols = this.features.size();
		
		featureValues = new double[rows0 + rows1][cols];
		truth = new double[rows0 + rows1][1];
		
		for(int i = 0; i < rows0; i++){
			featureValues[i] = this.data.get(0).get(i).matrix;
			truth[i] = this.data.get(0).get(i).outputIndex;
		}
		
		
		for(int i = 0; i < rows1; i++){
			featureValues[i+rows0] = this.data.get(1).get(i).matrix;
			truth[i+rows0] = this.data.get(1).get(i).outputIndex;
		}
		
		Matrix X = new Matrix(featureValues);
		Matrix Y = new Matrix(truth);
		

		
		Matrix thetaMatrix = ((X.transpose().times(X)).inverse().times(X.transpose().times(Y)));
		
		//System.out.println(theta.getArray().toString());
		
		theta = new double[thetaMatrix.getArray().length+1];
		theta[0] = 1;
		int index = 1;
		for(double[] a : thetaMatrix.getArray()){
			for(double b : a){
				theta[index] = b;
				//System.out.print(b + " ");
				index++;
			}
			//System.out.println();
		}
		
		double[][] test = {{60.0, 2.0, 4.0, 9.0, 1.0, 5.0, 3.0, 1.0, 2.0, 0.0, 0.0, 50.0, 1.0},
						   {28.0, 0.0, 1.0, 10.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 0.0},
						   {46.0, 4.0, 1.0, 10.0, 0.0, 12.0, 2.0, 0.0, 1.0, 0.0, 0.0, 40.0, 0.0}, 
							   {52.0, 0.0, 0.0, 13.0, 0.0, 4.0, 2.0, 0.0, 1.0, 15024.0, 0.0, 45.0, 0.0}, 
							   {52.0, 0.0, 0.0, 13.0, 0.0, 4.0, 2.0, 0.0, 1.0, 0.0, 0.0, 40.0, 0.0}, 
							   {28.0, 0.0, 3.0, 9.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 45.0, 0.0} };
		
		//System.out.println(theta.getArray().length);
		
//		int a = 0;
//		
//		System.out.println(theta[a]);
//		
//		double result = getCost(a);
//		
		double result = 1/(1+Math.pow(Math.E, -(dotProduct(theta, test[0]))));
		System.out.println(result);
		System.out.println(getHx(test[0]));
//		
//		theta[a] -= result;
//		
//		result = getCost(a);
//		System.out.println(result);
		
		double highest = 0;
		
		for(double f : featureValues[0])
			System.out.print(f + ", ");
		int iterations = 1;
		do {
			double[] newTheta = new double[theta.length];
			for(int i = 1; i < theta.length; i++){
				double change = getCost(i-1);
				newTheta[i] = theta[i] - (1.0/4)*change;
				//System.out.println(i + " " + change);
//				if (Math.abs(change) > highest)
//					highest = Math.abs(change);
			}
			theta = newTheta;
			//System.out.println(theta[8] + " " + getCost(8));
			//System.out.println("\n" + getHx(test[0]));
			highest++;
			iterations++;
		} while (highest < 15);
		
		result = 1/(1+Math.pow(Math.E, -(dotProduct(theta, test[3]))));
		System.out.println("\n" + getHx(test[1]));
				
				
//		for(double[] a : matrix){
//			for(double b : a)
//				System.out.print(b + ", ");
//			System.out.println();
//		}
//		
//		for(double[] b : truth)
//			System.out.print(b[0] + " ");
//		System.out.println();
		
		//generateOutput();
			
	}
	
	//Give feature array and feature output
	public double getCost(int xi){
		//number of training values
		double m = featureValues.length;
		
		double sum = 0;
		for(int i = 0; i < m; i++){
			double hx = getHx(featureValues[i]);
			double y = truth[i][0];
			//System.out.println(hx + " " + y);
			sum+= (y - hx)*featureValues[i][xi];
		}
		 
		return sum/m;
	}
	
	public double getHx(double[] x){
		//System.out.println((crossProduct(theta, x)));
		return 1.0/(1+Math.pow(Math.E, -(dotProduct(theta, x))));
	}
	
	public double dotProduct(double[] v1, double[] v2) {
		if(v1.length-1 != v2.length) return 0;
		double sum = v1[0];
		for(int i = 0; i < v2.length; i++)
			sum+= (v1[i+1]*v2[i]);
		return sum;
	}
	

	@Override
	public void makePredictions(String testDataFilepath) {
		
		
	}
	
	
	public Scanner readFile(String filename){
		Scanner s;
		try {
			s = new Scanner(new File(filename));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			s = null;
		}
		return s;
	}
	
	
	public void generateOutput(){
		PrintWriter writer;
		try {
			writer = new PrintWriter("output.txt");
			for(int i = 0; i < output.length; i++){
				writer.println(output[i]);
				for(DataSet d : data.get(i))
					writer.println(d);
				writer.println("\n");
			}
			writer.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}	
	}

}
