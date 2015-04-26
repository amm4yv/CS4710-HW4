import java.util.ArrayList;


public class FeatureHeader {

	String name;
	private boolean numeric;
	String[] values;
	double[] probabilities;
	
	public FeatureHeader(String[] data) {
		this.name = data[0];
		if (data[1].equals("numeric")) {
			this.numeric = true;
			this.values = null;
		} else {
			this.numeric = false;
			this.values = new String[data.length-1];
			for(int i = 1; i < data.length; i++){
				this.values[i-1] = data[i];
			}
		}
	}
	
	public int find(String value) {
		if(this.numeric) return Integer.parseInt(value);
		for(int i = 0; i < values.length; i++){
			if(values[i].equals(value))
				return i+1;
		}
		return -1;
	}
	
	public boolean isNumeric(){
		return this.numeric;
	}
	
	@Override
	public String toString(){
		String temp = "Name: " + this.name + "\n";
		if(!this.numeric){
			for(String s : this.values){
				temp += s + " ";
			}
		} else {
			temp += "numeric";
		}
		return temp;
	}
	
}
