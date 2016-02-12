package logistic_regression;

import java.util.ArrayList;

/**
 *
 * Created by saurabh on 12/2/16.
 */
public class DataPreProcessor {

    private double[] means;
    private double[] standardDeviations;

    /**
     * computes the column-wise mean and standard deviation
     *
     * @param data -
     */
    public DataPreProcessor(ArrayList<ArrayList<Double>> data) {
        int nColumns = data.get(0).size();
        int nRows = data.size();

        means = new double[nColumns];
        double[] squareMeans = new double[nColumns];
        standardDeviations = new double[nColumns];

        for (ArrayList<Double> x : data) {
            for (int i = 0; i < x.size(); i++) {
                means[i] += x.get(i) / nRows;
                squareMeans[i] += x.get(i) * x.get(i) / nRows;
            }
        }


        for (int i = 0; i < standardDeviations.length; i++) {
            standardDeviations[i] = Math.sqrt(squareMeans[i] - means[i] * means[i]);
        }

    }

    /**
     * Normalizes the row using mean and standard deviation.
     * newX = x - mean/standardDeviation
     *
     * @param row -
     * @return normalizedRow (ArrayList)
     */
    public ArrayList<Double> normalizeRow(ArrayList<Double> row) {
        ArrayList<Double> normalizedRow = new ArrayList<>();

        for (int i = 0; i < row.size(); i++) {
            normalizedRow.add((row.get(i) - means[i]) / standardDeviations[i]);
        }
        return normalizedRow;
    }

    /**
     * Normalizes the row using mean and standard deviation.
     * newX = x - mean/standardDeviation
     *
     * @param row -
     * @return normalizedRow (array of doubles)
     */
    public double[] normalizeRow(double[] row) {
        double[] normalizedRow = new double[row.length];

        for (int i = 0; i < row.length; i++) {
            normalizedRow[i] = (row[i] - means[i]) / standardDeviations[i];
        }
        return normalizedRow;
    }

    /**
     * Converts data to ArrayList of Instance(row, label) for training BinaryClassifier
     *
     * @param rows   -
     * @param labels -
     * @return trainingExamples (ArrayList of Instance objects)
     */
    public ArrayList<Instance> createTrainingExamples(ArrayList<ArrayList<Double>> rows, ArrayList<Integer> labels) {
        ArrayList<Instance> trainingData = new ArrayList<>();

        for (int i = 0; i < rows.size(); i++) {
            Instance inst = new Instance(rows.get(i).stream().mapToDouble(Double::doubleValue).toArray(), labels.get(i));
            trainingData.add(inst);
        }
        return trainingData;
    }

}
