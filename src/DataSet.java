import java.util.ArrayList;
import java.util.HashMap;

public class DataSet {

	String output;
	double[] outputIndex;
	private HashMap<FeatureHeader, String> data;
	double[] matrix;

	public DataSet(String[] input, ArrayList<FeatureHeader> features,
			String[] output) {
		this.data = new HashMap<FeatureHeader, String>();
		this.matrix = new double[input.length];
		this.outputIndex = new double[1];
		
		//bias
		this.matrix[0] = 1;

		for (int i = 0; i < input.length - 1; i++) {
			this.data.put(features.get(i), input[i]);
			matrix[i+1] = features.get(i).find(input[i]);
		}

		this.output = input[input.length - 1];

		if (output != null) {
			for (int i = 0; i < output.length; i++)
				if (this.output.equals(output[i]))
					this.outputIndex[0] = i;
		}

	}

	public String toString() {
		String s = "";
		for (FeatureHeader k : this.data.keySet())
			s += k.name + ":" + this.data.get(k) + " ";
		return s;
	}

}
