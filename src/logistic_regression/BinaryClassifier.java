package logistic_regression;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * An implementation of logistic regression using gradient descent for 2 class classification.
 * Created by saurabh on 10/2/16.
 */
public class BinaryClassifier {
    private double biasWeight;
    private double[] weights;

    public BinaryClassifier(int n) {
        biasWeight = 0.0;
        weights = new double[n];
    }

    protected double sigmoid(double z) {
        return 1.0 / (1 + Math.exp(-z));
    }

    /**
     * computes the probability that the given set of inputs have label 1.
     *
     * @param inputs input values
     * @return probability that label = 1
     */
    public double computeHypothesis(double[] inputs) {
        double summation = 0.0;
        summation += biasWeight;

        for (int i = 0; i < inputs.length; i++) {
            summation += inputs[i] * weights[i];
        }
        return sigmoid(summation);
    }

    /**
     * computes the cost function which is to be minimized.
     *
     * @param examples training data
     * @return cost
     */
    public double computeCost(ArrayList<Instance> examples) {
        int m = examples.size();

        double cost = 0.0;
        for (Instance example : examples) {
            double prediction = computeHypothesis(example.getAllInputs());
            int label = example.getLabel();
            cost += label * Math.log(prediction)
                    + (1.0 - label) * Math.log(1.0 - prediction);
        }
        return -cost / m;
    }


    /**
     * Optimises the weights using given training data with gradient descent.
     * calls _trainIteration nIterations number of times.
     *
     * @param examples    training data
     * @param rate        learning rate
     * @param nIterations number of iterations
     * @param verbose     prints log after each iteration if true
     */
    public void train(ArrayList<Instance> examples, double rate, int nIterations, boolean verbose) {
        for (int i = 0; i < nIterations; i++) {
            _trainIteration(examples, rate);

            if (verbose) {
                System.out.printf("Iteration %d, cost = %f\n", i, computeCost(examples));
            }
        }

    }

    /**
     * classifies given inputs as 1 if hypothesis value >= threshold else 0
     *
     * @param inputs -
     * @param threshold -
     * @return class 0 or 1
     */
    public int classify(double[] inputs, double threshold) {
        double p = computeHypothesis(inputs);
        if (p >= threshold) {
            return 1;
        } else {
            return 0;
        }
    }


    protected void _trainIteration(ArrayList<Instance> examples, double rate) {
        double biasGradient = 0.0;
        double[] gradients = new double[weights.length];

        for (Instance example : examples) {

            double error = example.getLabel() - computeHypothesis(example.getAllInputs());
            biasGradient += (error);
            for (int i = 0; i < gradients.length; i++) {
                gradients[i] += (error) * example.getInput(i);
            }

        }
        for (int i = 0; i < weights.length; i++) {
            gradients[i] = gradients[i] / examples.size();
        }

        biasWeight += rate * biasGradient / examples.size();
        for (int i = 0; i < weights.length; i++) {
            weights[i] += rate * gradients[i] / examples.size();
        }

    }

    @Override
    public String toString() {
        return String.format("BinaryClassifier [ biasWeight=%s weights=%s ]", biasWeight, Arrays.toString(weights));
    }
}
