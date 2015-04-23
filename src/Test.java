import java.io.FileNotFoundException;


public class Test {

	public static void main(String[] args) throws FileNotFoundException {
		CustomClassifier tester = new CustomClassifier("trainingData/census.names");
		System.out.println();
		tester.train("trainingData/censusShort.train");
	}
	
}
