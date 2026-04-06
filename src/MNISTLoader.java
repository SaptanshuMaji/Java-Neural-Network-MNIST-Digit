import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class MNISTLoader {
    public static double[][] loadImages(String file, int max) throws IOException {
        DataInputStream dis = new DataInputStream(new FileInputStream(file));
        dis.readInt();
        int num = dis.readInt();
        int rows = dis.readInt();
        int cols = dis.readInt();
        int count = Math.min(num, max);
        double[][] images = new double[count][rows * cols];
        for (int i = 0; i < count; i++)
            for (int j = 0; j < rows * cols; j++)
                images[i][j] = dis.readUnsignedByte() / 255.0;
        dis.close();
        return images;
    }

    public static int[] loadLabels(String file, int max) throws IOException {
        DataInputStream dis = new DataInputStream(new FileInputStream(file));
        dis.readInt();
        int num = dis.readInt();
        int count = Math.min(num, max);
        int[] labels = new int[count];
        for (int i = 0; i < count; i++)
            labels[i] = dis.readUnsignedByte();
        dis.close();
        return labels;
    }
}