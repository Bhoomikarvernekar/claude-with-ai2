import java.util.Random;

public class NeuralNetworkMovie {
    private static final int INPUT_SIZE = 2;
    private static final int HIDDEN_SIZE = 2;
    private static final double LEARNING_RATE = 0.1;
    private static final int EPOCHS = 10_000;

    private final double[][] inputToHidden = new double[INPUT_SIZE][HIDDEN_SIZE];
    private final double[] hiddenBias = new double[HIDDEN_SIZE];
    private final double[] hiddenToOutput = new double[HIDDEN_SIZE];
    private double outputBias;

    private final Random random = new Random(42);

    public NeuralNetworkMovie() {
        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                inputToHidden[i][j] = randomWeight();
            }
        }

        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hiddenBias[j] = randomWeight();
            hiddenToOutput[j] = randomWeight();
        }

        outputBias = randomWeight();
    }

    private double randomWeight() {
        return random.nextDouble() * 2 - 1; // Range: -1 to 1
    }

    private double sigmoid(double value) {
        return 1.0 / (1.0 + Math.exp(-value));
    }

    public double predict(double[] input) {
        if (input.length != INPUT_SIZE) {
            throw new IllegalArgumentException("Input must contain exactly 2 values.");
        }

        double[] hidden = new double[HIDDEN_SIZE];

        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double sum = hiddenBias[j];

            for (int i = 0; i < INPUT_SIZE; i++) {
                sum += input[i] * inputToHidden[i][j];
            }

            hidden[j] = sigmoid(sum);
        }

        double outputSum = outputBias;

        for (int j = 0; j < HIDDEN_SIZE; j++) {
            outputSum += hidden[j] * hiddenToOutput[j];
        }

        return sigmoid(outputSum);
    }

    public void train(double[][] inputs, double[] targets) {
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            for (int sample = 0; sample < inputs.length; sample++) {
                double[] input = inputs[sample];
                double target = targets[sample];

                double[] hidden = new double[HIDDEN_SIZE];

                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    double sum = hiddenBias[j];

                    for (int i = 0; i < INPUT_SIZE; i++) {
                        sum += input[i] * inputToHidden[i][j];
                    }

                    hidden[j] = sigmoid(sum);
                }

                double outputSum = outputBias;

                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    outputSum += hidden[j] * hiddenToOutput[j];
                }

                double output = sigmoid(outputSum);
                double outputError = target - output;
                double outputDelta = outputError * output * (1 - output);

                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    double hiddenError = outputDelta * hiddenToOutput[j];
                    double hiddenDelta = hiddenError * hidden[j] * (1 - hidden[j]);

                    hiddenToOutput[j] += LEARNING_RATE * outputDelta * hidden[j];
                    hiddenBias[j] += LEARNING_RATE * hiddenDelta;

                    for (int i = 0; i < INPUT_SIZE; i++) {
                        inputToHidden[i][j] += LEARNING_RATE * hiddenDelta * input[i];
                    }
                }

                outputBias += LEARNING_RATE * outputDelta;
            }
        }
    }

    public static void main(String[] args) {
        // Input: [likes action movies, likes romance movies]
        // Target: 1 = likes recommended movie, 0 = does not like it
        double[][] inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };

        double[] targets = {0, 1, 1, 1};

        NeuralNetworkMovie network = new NeuralNetworkMovie();
        network.train(inputs, targets);

        System.out.println("Action fan: " + network.predict(new double[]{1, 0}));
        System.out.println("Romance fan: " + network.predict(new double[]{0, 1}));
        System.out.println("Likes both: " + network.predict(new double[]{1, 1}));
        System.out.println("Likes neither: " + network.predict(new double[]{0, 0}));
    }
}
