package logistic_regression;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;


public class Main {

    public static void main(String[] args) {
        final String DATA_FILE_NAME = "test-scores-data.csv";
        final String REGEX = "\\s*,\\s*";
        final int N_INPUT_ATTRIBUTES = 2;
        final double LEARNING_RATE = 0.05;
        final int N_ITERATIONS = 40000;
        final boolean VERBOSE = false;

        ArrayList<ArrayList<Double>> X = new ArrayList<>();
        ArrayList<Integer> y = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(DATA_FILE_NAME))) {

            String line;
            while ((line = br.readLine()) != null) {
                String[] cells = line.split(REGEX);

                ArrayList<Double> values = new ArrayList<>();
                for (int i = 0; i < cells.length - 1; i++) {
                    values.add(Double.parseDouble(cells[i]));
                }
                X.add(values);

                Integer label = Integer.parseInt(cells[cells.length - 1]);
                y.add(label);
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Error reading data from file");
        }

        DataPreProcessor preProcessor = new DataPreProcessor(X);
        ArrayList<ArrayList<Double>> normalizedX = new ArrayList<>();

        for (ArrayList<Double> x : X) {
            normalizedX.add(preProcessor.normalizeRow(x));
        }

        ArrayList<Instance> examples = preProcessor.createTrainingExamples(normalizedX, y);

        BinaryClassifier classifier = new BinaryClassifier(N_INPUT_ATTRIBUTES);

        System.out.printf("Initial cost = %f\n", classifier.computeCost(examples));
        System.out.println("== Starting Training ==");
        System.out.printf("Learning Rate = %f, No. of iterations = %d\n", LEARNING_RATE, N_ITERATIONS);
        classifier.train(examples, LEARNING_RATE, N_ITERATIONS, VERBOSE);

        System.out.println("== Training Complete ==");
        System.out.println(classifier);

        System.out.printf("Final cost = %f\n", classifier.computeCost(examples));

        double[] test = {45.0, 85.0};
        System.out.printf("Test data = %s\n", Arrays.toString(test));
        double h = classifier.computeHypothesis(preProcessor.normalizeRow(test));
        System.out.printf("h(test) = %f\n", h);
    }
}
