import javax.swing.*;
import java.awt.*;
import java.util.*;
import java.util.List;

public class Main {
    private static volatile boolean keepTraining = true;

    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork();

        try {
            nn.loadWeights("model_weights.dat");
        } catch (Exception ignored) {
        }

        JFrame frame = new JFrame("MNIST Recognizer");
        DrawingPanel panel = new DrawingPanel();
        JButton submitButton = new JButton("Predict");
        JButton clearButton = new JButton("Clear");
        JButton stopButton = new JButton("Stop Training");
        JLabel statusLabel = new JLabel("Status: Waiting...");

        stopButton.addActionListener(e -> {
            keepTraining = false;
            stopButton.setEnabled(false);
            statusLabel.setText("Status: Stopping...");
        });

        submitButton.addActionListener(e -> {
            double[] input = panel.getNormalizedInput();
            double[] output = nn.predict(input);
            Integer[] indices = new Integer[10];
            for (int i = 0; i < 10; i++) indices[i] = i;
            Arrays.sort(indices, (a, b) -> Double.compare(output[b], output[a]));

            StringBuilder msg = new StringBuilder("Top Predictions:\n");
            for (int i = 0; i < 3; i++) {
                msg.append(indices[i]).append(": ").append(String.format("%.2f", output[indices[i]] * 100)).append("%\n");
            }
            JOptionPane.showMessageDialog(frame, msg.toString());
        });

        clearButton.addActionListener(e -> panel.clearGrid());

        JPanel buttonPanel = new JPanel();
        buttonPanel.add(submitButton);
        buttonPanel.add(clearButton);
        buttonPanel.add(stopButton);

        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new BorderLayout());
        frame.add(panel, BorderLayout.CENTER);
        frame.add(buttonPanel, BorderLayout.SOUTH);
        frame.add(statusLabel, BorderLayout.NORTH);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        new Thread(() -> {
            try {
                double[][] allImages = MNISTLoader.loadImages("data_mnist/train-images.idx3-ubyte", 60000);
                int[] allLabels = MNISTLoader.loadLabels("data_mnist/train-labels.idx1-ubyte", 60000);

                double[][] testImages = MNISTLoader.loadImages("data_mnist/t10k-images.idx3-ubyte", 10000);
                int[] testLabels = MNISTLoader.loadLabels("data_mnist/t10k-labels.idx1-ubyte", 10000);

                int trainSize = 50000;
                double[][] trainImages = Arrays.copyOfRange(allImages, 0, trainSize);
                int[] trainLabels = Arrays.copyOfRange(allLabels, 0, trainSize);
                double[][] valImages = Arrays.copyOfRange(allImages, trainSize, 60000);
                int[] valLabels = Arrays.copyOfRange(allLabels, trainSize, 60000);

                List<Integer> indices = new ArrayList<>();
                for (int i = 0; i < trainSize; i++) indices.add(i);

                double bestAcc = 0;
                int batchSize = 32;

                for (int epoch = 0; epoch < 30 && keepTraining; epoch++) {
                    Collections.shuffle(indices);
                    final int currentEpoch = epoch + 1;
                    SwingUtilities.invokeLater(() -> statusLabel.setText("Status: Training Epoch " + currentEpoch));

                    int actualBatchSize = 0;
                    for (int i = 0; i < trainSize && keepTraining; i++) {
                        int idx = indices.get(i);
                        nn.accumulateGradients(trainImages[idx], trainLabels[idx]);
                        actualBatchSize++;
                        if (actualBatchSize == batchSize || i == trainSize - 1) {
                            nn.applyGradients(actualBatchSize);
                            actualBatchSize = 0;
                        }
                    }

                    double valAcc = runEvaluation(nn, valImages, valLabels);
                    System.out.println("Epoch " + currentEpoch + " | Val Accuracy: " + String.format("%.4f", valAcc));

                    if (valAcc >= 0.975) {
                        System.out.println("Target accuracy reached. Running unbiased final test...");
                        double testAcc = runEvaluation(nn, testImages, testLabels);
                        System.out.println("Final Test Accuracy: " + String.format("%.4f", testAcc));

                        SwingUtilities.invokeLater(() -> statusLabel.setText("Status: Final Accuracy: " + String.format("%.4f", testAcc)));
                        nn.saveWeights("model_weights.dat");
                        keepTraining = false;
                    } else {
                        if (valAcc < bestAcc) {
                            nn.setLearningRate(nn.getLearningRate() * 0.9);
                        } else {
                            bestAcc = valAcc;
                        }
                    }
                }
                SwingUtilities.invokeLater(() -> stopButton.setEnabled(false));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }).start();
    }

    private static double runEvaluation(NeuralNetwork nn, double[][] images, int[] labels) {
        int correct = 0;
        for (int i = 0; i < images.length; i++) {
            double[] out = nn.predict(images[i]);
            int guess = 0;
            for (int j = 1; j < 10; j++) if (out[j] > out[guess]) guess = j;
            if (guess == labels[i]) correct++;
        }
        return (double) correct / images.length;
    }
}