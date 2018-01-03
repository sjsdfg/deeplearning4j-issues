# SparkDl4jMultiLayer模型存储
---


使用Spark分布式训练需要使用如下的类，那么如何对模型进行保存
```
SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);
```

首先调用`getNetwork()`方法获取对应的实体类
```
MultiLayerNetWork network = sparkNet.getNetwork();
```
然后调用调用
```
ModelSerializer.restoreMultiLayerNetwork()
```
对模型进行保存