import java.io.FileNotFoundException;


public class Test {

	public static void main(String[] args) throws FileNotFoundException {
		//RandomClassifier tester = new RandomClassifier("trainingData/census.names");
		//LogRegClassifier tester = new LogRegClassifier("trainingData/census.names");
		BayesClassifier tester = new BayesClassifier("trainingData/census.names");
		tester.train("trainingData/censusShort.train");
		//for(int i = 0; i < 100; i++)
			tester.makePredictions("trainingData/census.train");
	}
	
}
