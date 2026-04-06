import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.util.Arrays;

public class DrawingPanel extends JPanel {
    private double[][] grid = new double[28][28];
    private final double brushSigma = 0.6;

    public DrawingPanel() {
        setPreferredSize(new Dimension(420, 420));
        setBackground(Color.BLACK);
        MouseAdapter ma = new MouseAdapter() {
            public void mouseDragged(MouseEvent e) { draw(e.getX(), e.getY()); }
            public void mousePressed(MouseEvent e) { draw(e.getX(), e.getY()); }
        };
        addMouseMotionListener(ma);
        addMouseListener(ma);
    }

    private void draw(int mx, int my) {
        double x = (double) mx / getWidth() * 28;
        double y = (double) my / getHeight() * 28;
        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                double distSq = Math.pow(col - x, 2) + Math.pow(row - y, 2);
                double intensity = 255 * Math.exp(-distSq / (2 * brushSigma * brushSigma));
                grid[row][col] = Math.min(255, grid[row][col] + intensity);
            }
        }
        repaint();
    }

    public double[] getNormalizedInput() {
        double totalMass = 0, mX = 0, mY = 0;
        int minR = 28, maxR = 0, minC = 28, maxC = 0;
        boolean empty = true;

        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                if (grid[r][c] > 5) {
                    empty = false;
                    totalMass += grid[r][c];
                    mX += c * grid[r][c];
                    mY += r * grid[r][c];
                    minR = Math.min(minR, r); maxR = Math.max(maxR, r);
                    minC = Math.min(minC, c); maxC = Math.max(maxC, c);
                }
            }
        }

        if (empty) return new double[784];

        int w = maxC - minC + 1;
        int h = maxR - minR + 1;
        double scale = 20.0 / Math.max(w, h);
        int nw = (int) (w * scale);
        int nh = (int) (h * scale);

        BufferedImage rawImg = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                rawImg.getRaster().setSample(c, r, 0, (int) grid[r][c]);
            }
        }

        Image scaled = rawImg.getSubimage(minC, minR, w, h).getScaledInstance(nw, nh, Image.SCALE_SMOOTH);
        BufferedImage finalImg = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = finalImg.createGraphics();
        g2d.setBackground(Color.BLACK);
        g2d.clearRect(0, 0, 28, 28);

        double cogX = (mX / totalMass - minC) * scale;
        double cogY = (mY / totalMass - minR) * scale;
        int offsetX = (int) (14.0 - cogX);
        int offsetY = (int) (14.0 - cogY);

        g2d.drawImage(scaled, offsetX, offsetY, null);
        g2d.dispose();

        double[] finalInput = new double[784];
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                finalInput[r * 28 + c] = finalImg.getRaster().getSample(c, r, 0) / 255.0;
            }
        }
        printDebugImage(finalInput);
        return finalInput;
    }

    private void printDebugImage(double[] input) {
        System.out.println("--- AI INPUT VIEW ---");
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                double val = input[r * 28 + c];
                if (val > 0.5) System.out.print("#");
                else if (val > 0.1) System.out.print(".");
                else System.out.print(" ");
            }
            System.out.println();
        }
    }

    public void clearGrid() {
        for (int r = 0; r < 28; r++) Arrays.fill(grid[r], 0);
        repaint();
    }

    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        int cellW = getWidth() / 28;
        int cellH = getHeight() / 28;
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                int val = (int) Math.min(255, grid[r][c]);
                g.setColor(new Color(val, val, val));
                g.fillRect(c * cellW, r * cellH, cellW, cellH);
            }
        }
    }
}