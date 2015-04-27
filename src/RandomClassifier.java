import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;


public class RandomClassifier extends Classifier {
	
	private String[] output;
	private ArrayList<FeatureHeader> features;
	private ArrayList<ArrayList<DataSet>> data;
	private double[] theta;
	private double[][] featureValues;
	private double[][] truth;

	public RandomClassifier(String namesFilepath) {
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
		// TODO Auto-generated method stub
		
	}

	@Override
	public void makePredictions(String testDataFilepath) {
		Scanner file = readFile(testDataFilepath);
		
		int count = 1;
		int correct = 0;
		
		while (file.hasNextLine()) {
			String[] line = file.nextLine().split("\\s+");

			DataSet data = new DataSet(line, features, null);

			double hx = Math.random();
			
			String out = hx < 0.5 ? output[0] : output[1];
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
