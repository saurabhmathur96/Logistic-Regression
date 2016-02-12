Logistic-Regression
===================

An implementation of Binary Logistic Regression Classifier using Gradient Descent in pure Java 8.

Classes :
- Instance
- DataPreProcessor
- BinaryClassifier

Test-Data Files:
- test-scores-data.csv : From Stanford's ml-class on Coursera, assignment2 `ex2data1.txt` 
- xor.csv : The classifier fails on this as the data is not linearly separable.

Output from test run:
```
Initial cost = 0.693147
== Starting Training ==
Learning Rate = 0.050000, No. of iterations = 40000
== Training Complete ==
BinaryClassifier [ biasWeight=0.8150521625489258 weights=[1.8105877596418987, 1.6405966520919257] ]
Final cost = 0.258550
Test data = [45.0, 85.0]
h(test) = 0.634277

```


## Documentation

### `public class Instance`

Stores a single training example for binary classification. Created by saurabh on 10/2/16.



 

### `public class DataPreProcessor`

Created by saurabh on 12/2/16.

##### `public DataPreProcessor(ArrayList<ArrayList<Double>> data)`

computes the column-wise mean and standard deviation

 * **Parameters:** `data` — 

##### `public ArrayList<Double> normalizeRow(ArrayList<Double> row)`

Normalizes the row using mean and standard deviation. newX = x - mean/standardDeviation

 * **Parameters:** `row` — 
 * **Returns:** normalizedRow (ArrayList)

##### `public double[] normalizeRow(double[] row)`

Normalizes the row using mean and standard deviation. newX = x - mean/standardDeviation

 * **Parameters:** `row` — 
 * **Returns:** normalizedRow (array of doubles)

##### `public ArrayList<Instance> createTrainingExamples(ArrayList<ArrayList<Double>> rows, ArrayList<Integer> labels)`

Converts data to ArrayList of Instance(row, label) for training BinaryClassifier

 * **Parameters:**
   * `rows` — 
   * `labels` — 
 * **Returns:** trainingExamples (ArrayList of Instance objects)
 




### `public class BinaryClassifier`

An implementation of logistic regression using gradient descent for 2 class classification. Created by saurabh on 10/2/16.

##### `public double computeHypothesis(double[] inputs)`

computes the probability that the given set of inputs have label 1.

 * **Parameters:** `inputs` — 
 * **Returns:** probability that label = 1

##### `public double computeCost(ArrayList<Instance> examples)`

computes the cost function which is to be minimized.

 * **Parameters:** `examples` — 
 * **Returns:** cost

##### `public void train(ArrayList<Instance> examples, double rate, int nIterations, boolean verbose)`

Optimises the weights using given training data with gradient descent. calls _trainIteration nIterations number of times.

 * **Parameters:**
   * `examples` — 
   * `rate` — 
   * `nIterations` — 
   * `verbose` — 

##### `public int classify(double[] inputs, double threshold)`

classifies given inputs as 1 if hypothesis value >= threshold else 0

 * **Parameters:**
   * `inputs` — 
   * `threshold` — 
 * **Returns:** class 0 or 1
 