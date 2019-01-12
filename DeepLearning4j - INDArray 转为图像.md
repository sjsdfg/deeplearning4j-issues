# DeepLearning4j - INDArray 转为图像

# 三通通道彩色图

代码引自：https://github.com/sjsdfg/dl4j-tutorials/blob/master/src/main/java/styletransfer/NeuralStyleTransfer.java

```Java
 /**
 * Takes an INDArray containing an image loaded using the native image loader
 * libraries associated with DL4J, and converts it into a BufferedImage.
 * The INDArray contains the color values split up across three channels (RGB)
 * and in the integer range 0-255.
 *
 * @param array INDArray containing an image
 * @return BufferedImage
 */
private BufferedImage imageFromINDArray(INDArray array) {
    long[] shape = array.shape();

    long height = shape[2];
    long width = shape[3];
    BufferedImage image = new BufferedImage((int)width, (int)height, BufferedImage.TYPE_INT_RGB);
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int red = array.getInt(0, 2, y, x);
            int green = array.getInt(0, 1, y, x);
            int blue = array.getInt(0, 0, y, x);

            //handle out of bounds pixel values
            red = Math.min(red, 255);
            green = Math.min(green, 255);
            blue = Math.min(blue, 255);

            red = Math.max(red, 0);
            green = Math.max(green, 0);
            blue = Math.max(blue, 0);
            image.setRGB(x, y, new Color(red, green, blue).getRGB());
        }
    }
    return image;
}
```

# 单通道灰度图

代码引自：https://github.com/sjsdfg/dl4j-tutorials/blob/master/src/main/java/lesson6/UsingModelToPredict.java

```Java
/**
 * 将单通道的 INDArray 保存为灰度图
 *
 * There's also NativeImageLoader.asMat(INDArray) and we can then use OpenCV to save it as an image file.
 *
 * @param array 输入
 * @return 灰度图转化
 */
private static BufferedImage imageFromINDArray(INDArray array) {
    long[] shape = array.shape();

    int height = (int)shape[2];
    int width = (int)shape[3];
    BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int gray = array.getInt(0, 0, y, x);

            // handle out of bounds pixel values
            gray = Math.min(gray, 255);
            gray = Math.max(gray, 0);

            image.getRaster().setSample(x, y, 0, gray);
        }
    }
    return image;
}
```

# 保存图片到本地

```Java
private void saveImage(INDArray combination, int iteration) throws IOException {
    IMAGE_PRE_PROCESSOR.revertFeatures(combination);

    BufferedImage output = imageFromINDArray(combination);
    URL resource = getClass().getResource(OUTPUT_PATH);
    File file = new File(resource.getPath() + "/iteration" + iteration + ".jpg");
    ImageIO.write(output, "jpg", file);
}
```



