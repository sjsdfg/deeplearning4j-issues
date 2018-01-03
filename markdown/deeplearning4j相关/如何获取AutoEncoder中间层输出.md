# 获取AutoEncoder中间层输出

dl4j对于自动编码机提供了多种实现形式

 - [深度自动编码器][1]
 - [降噪自动编码器][2]
 - [堆叠式降噪自动编码器][3]

---

在dl4j中，对于`MultiLayerNetwork`提供了两个方法

```
net.activate(layer, input)
net.feedForwardToLayer(layNum,input)
```
可以用于获取中间输出，
通过在github上面gitter的作者提供，应当使用`net.feedForwardToLayer(layNum,input)`方法


>   
**liuqiang @liuq4360 10:31**
hi , i use deep autoencoder to compress data, how can i get the compressed feature vector? which one is correct of the flowning two?
net.activate(layer, input)
net.feedForwardToLayer(layNum,input)
**Alex Black @AlexDBlack 10:52**
@liuq4360 you want feedForwardToLayer usually

-----
更多文档可以查看 https://github.com/sjsdfg/deeplearning4j-issues。
欢迎star

  [1]: https://deeplearning4j.org/cn/deepautoencoder
  [2]: https://deeplearning4j.org/cn/denoisingautoencoder
  [3]: https://deeplearning4j.org/cn/stackeddenoisingautoencoder