# Deeplearning4j - 使用SparkComputationGraph构建网络UI不能正常监控

---

# 一、问题描述

使用`SparkComputationGraph`构建跳跃连接（skip connection）的神经网络结构，在模型训练阶段，使用`RemoteUIStatsStorageRouter`用于监控模型的运行状态，但是无法获取模型的训练信息。

 - Deeplearning4j 版本为 1.0

```scala
/** spark graph use **/ 
val sparkGraph = new SparkComputationGraph(sparkConf, networkConfig, trainingMaster) 
val seesionId = "test" 
val remoteUIRouter: StatsStorageRouter = new RemoteUIStatsStorageRouter("http://localhost:9000") 
val statsListener = new StatsListener(null, null, null, seesionId, null) 
sparkGraph.setListeners(remoteUIRouter, Collections.singletonList(statsListener))

UIServer uiServer = UIServer.getInstance();
uiServer.enableRemoteListener();
```

但是界面展现图均为空：
<center>![此处输入图片的描述][1]</center>
<center>![此处输入图片的描述][2]</center>


# 二、解决方法
在构造监听器时，使用如下代码段：
```
val statsConfig = new DefaultStatsUpdateConfiguration.Builder().reportingFrequency(1).build() 
val statsListener = new StatsListener(null, null, statsConfig, seesionId, null)
```
--- 
原问题链接： https://github.com/deeplearning4j/deeplearning4j/issues/5080#issue-320828533

更多文档可以查看 https://github.com/sjsdfg/deeplearning4j-issues。
你的star是我持续分享的动力

  [1]: https://user-images.githubusercontent.com/18146142/39741881-10ce9422-52ce-11e8-91b1-6e33c2942ce8.png
  [2]: https://user-images.githubusercontent.com/18146142/39741882-1118a29c-52ce-11e8-8817-d4848f3d2896.png
  [3]: https://user-images.githubusercontent.com/18146142/39741884-115c94de-52ce-11e8-8643-983b68b2db3a.png
  [4]: https://user-images.githubusercontent.com/18146142/39741884-115c94de-52ce-11e8-8643-983b68b2db3a.png