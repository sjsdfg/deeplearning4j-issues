# ND4J读取图片

## 一、构建加载器

```
NativaImageLoader loader = new NativeImageLoader(height, width, channels);
```
构建图片加载器，顺便完成了对图片的缩放功能，可以直接用于fit网络模型。

 1. height: 图片的高
 2. width: 图片的宽
 3. channels: 图片通道数，3为彩色，1为黑白

## 二、图片读取

```
INDArray image = loader.asMatrix(new File("/picture/path"))
```
根据图片路径读取对应图片成为`INDArray`

## 三，图片矩阵规范化

```
DataNormalization scaler = new ImagePreProcessScaler(0, 1);
scaler.transform(image);
```

`ImagePreProcessScaler`构造器参数为需要规范化的区间。

---
更多文档可以查看 https://github.com/sjsdfg/deeplearning4j-issues。
你的star是我持续分享的动力