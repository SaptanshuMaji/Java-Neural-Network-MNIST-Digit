import java.io.*;
import java.util.Random;

public class NeuralNetwork {
    private final int inputSize = 784, hiddenSize = 256, outputSize = 10;
    private double[][] w1 = new double[inputSize][hiddenSize];
    private double[][] w2 = new double[hiddenSize][outputSize];
    private double[] b1 = new double[hiddenSize];
    private double[] b2 = new double[outputSize];

    private double[][] g1 = new double[inputSize][hiddenSize];
    private double[][] g2 = new double[hiddenSize][outputSize];
    private double[] gb1 = new double[hiddenSize];
    private double[] gb2 = new double[outputSize];

    private double[][] m1 = new double[inputSize][hiddenSize];
    private double[][] v1 = new double[inputSize][hiddenSize];
    private double[][] m2 = new double[hiddenSize][outputSize];
    private double[][] v2 = new double[hiddenSize][outputSize];
    private double[] mb1 = new double[hiddenSize];
    private double[] vb1 = new double[hiddenSize];
    private double[] mb2 = new double[outputSize];
    private double[] vb2 = new double[outputSize];

    private double[] hRaw = new double[hiddenSize];
    private double[] h = new double[hiddenSize];
    private double[] logits = new double[outputSize];
    private double[] eo = new double[outputSize];
    private double[] eh = new double[hiddenSize];

    private double learningRate = 0.001;
    private final double beta1 = 0.9;
    private final double beta2 = 0.999;
    private final double epsilon = 1e-8;
    private int t = 0;

    private final double temperature = 1.0;
    private final double smoothing = 0.1;
    private final Random rand = new Random();

    public NeuralNetwork() {
        double std1 = Math.sqrt(2.0 / inputSize);
        for (int i = 0; i < inputSize; i++)
            for (int j = 0; j < hiddenSize; j++) w1[i][j] = rand.nextGaussian() * std1;

        double std2 = Math.sqrt(2.0 / hiddenSize);
        for (int i = 0; i < hiddenSize; i++)
            for (int j = 0; j < outputSize; j++) w2[i][j] = rand.nextGaussian() * std2;
    }

    public void saveWeights(String path) throws Exception {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(path))) {
            out.writeObject(w1);
            out.writeObject(w2);
            out.writeObject(b1);
            out.writeObject(b2);
        }
    }

    public void loadWeights(String path) throws Exception {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(path))) {
            w1 = (double[][]) in.readObject();
            w2 = (double[][]) in.readObject();
            b1 = (double[]) in.readObject();
            b2 = (double[]) in.readObject();
        }
    }

    private double relu(double x) { return x > 0 ? x : x * 0.01; }
    private double reluDeriv(double x) { return x > 0 ? 1 : 0.01; }

    private double[] softmax(double[] x) {
        double max = x[0], sum = 0;
        for (double v : x) if (v > max) max = v;
        double[] r = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            r[i] = Math.exp((x[i] - max) / temperature);
            sum += r[i];
        }
        for (int i = 0; i < x.length; i++) r[i] /= (sum + 1e-10);
        return r;
    }

    public double[] predict(double[] input) {
        for (int i = 0; i < hiddenSize; i++) {
            double s = b1[i];
            for (int j = 0; j < inputSize; j++) s += input[j] * w1[j][i];
            h[i] = relu(s);
        }
        double[] o = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            double s = b2[i];
            for (int j = 0; j < hiddenSize; j++) s += h[j] * w2[j][i];
            o[i] = s;
        }
        return softmax(o);
    }

    public void accumulateGradients(double[] input, int label) {
        for (int i = 0; i < hiddenSize; i++) {
            double s = b1[i];
            for (int j = 0; j < inputSize; j++) s += input[j] * w1[j][i];
            hRaw[i] = s; h[i] = relu(s);
        }
        for (int i = 0; i < outputSize; i++) {
            double s = b2[i];
            for (int j = 0; j < hiddenSize; j++) s += h[j] * w2[j][i];
            logits[i] = s;
        }
        double[] o = softmax(logits);
        double lowConf = smoothing / outputSize;
        double highConf = 1.0 - smoothing + lowConf;
        for (int i = 0; i < outputSize; i++) {
            double target = (i == label) ? highConf : lowConf;
            eo[i] = o[i] - target;
        }
        for (int i = 0; i < hiddenSize; i++) {
            double s = 0;
            for (int j = 0; j < outputSize; j++) s += eo[j] * w2[i][j];
            eh[i] = s * reluDeriv(hRaw[i]);
        }
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                g2[i][j] += eo[j] * h[i];
            }
        }
        for (int i = 0; i < outputSize; i++) {
            gb2[i] += eo[i];
        }
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                g1[i][j] += eh[j] * input[i];
            }
        }
        for (int i = 0; i < hiddenSize; i++) {
            gb1[i] += eh[i];
        }
    }

    public void applyGradients(int actualBatchSize) {
        t++;
        double factor = 1.0 / actualBatchSize;
        double lr_t = learningRate * Math.sqrt(1 - Math.pow(beta2, t)) / (1 - Math.pow(beta1, t));

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                double grad = g2[i][j] * factor;
                m2[i][j] = beta1 * m2[i][j] + (1 - beta1) * grad;
                v2[i][j] = beta2 * v2[i][j] + (1 - beta2) * grad * grad;
                w2[i][j] -= lr_t * m2[i][j] / (Math.sqrt(v2[i][j]) + epsilon);
                g2[i][j] = 0;
            }
        }
        for (int i = 0; i < outputSize; i++) {
            double grad = gb2[i] * factor;
            mb2[i] = beta1 * mb2[i] + (1 - beta1) * grad;
            vb2[i] = beta2 * vb2[i] + (1 - beta2) * grad * grad;
            b2[i] -= lr_t * mb2[i] / (Math.sqrt(vb2[i]) + epsilon);
            gb2[i] = 0;
        }
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                double grad = g1[i][j] * factor;
                m1[i][j] = beta1 * m1[i][j] + (1 - beta1) * grad;
                v1[i][j] = beta2 * v1[i][j] + (1 - beta2) * grad * grad;
                w1[i][j] -= lr_t * m1[i][j] / (Math.sqrt(v1[i][j]) + epsilon);
                g1[i][j] = 0;
            }
        }
        for (int i = 0; i < hiddenSize; i++) {
            double grad = gb1[i] * factor;
            mb1[i] = beta1 * mb1[i] + (1 - beta1) * grad;
            vb1[i] = beta2 * vb1[i] + (1 - beta2) * grad * grad;
            b1[i] -= lr_t * mb1[i] / (Math.sqrt(vb1[i]) + epsilon);
            gb1[i] = 0;
        }
    }

    public void train(double[] input, int label) {
        accumulateGradients(input, label);
        applyGradients(1);
    }

    public void setLearningRate(double lr) { this.learningRate = lr; }
    public double getLearningRate() { return this.learningRate; }
}