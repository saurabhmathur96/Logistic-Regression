package logistic_regression;

import java.util.Arrays;

/**
 * Stores a single training example for binary classification.
 * Created by saurabh on 10/2/16.
 */
public class Instance {
    private double[] values;
    private int label;


    public Instance(double[] X, int y) {
        values = X;
        label = y;
    }


    public int getLabel() {
        return label;
    }

    public double[] getAllInputs() {
        return values;
    }

    public double getInput(int i) {
        return values[i];
    }

    @Override
    public String toString() {
        return String.format("Instance [ values=%s label=%s ]", Arrays.toString(values), label);
    }
}
