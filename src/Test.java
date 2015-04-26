import java.io.FileNotFoundException;


public class Test {

	public static void main(String[] args) throws FileNotFoundException {
		CustomClassifier tester = new CustomClassifier("trainingData/census.names");
		//BayesClassifier tester = new BayesClassifier("trainingData/census.names");
		tester.train("trainingData/census.train");
		tester.makePredictions("trainingData/censusShort.train");
	}
	
}
