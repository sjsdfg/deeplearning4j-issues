# Deeplearning4j模型保存至HDFS

---
同时支持训练出来的网络模型存储在HDFS上：
![QQ截图20180118131217.png-60.1kB][6]

上面示例图中的代码不是很全，这里进行简易补全

需要引入的HDFS的包如下:
```
importorg.apache.hadoop.conf.Configuration;
importorg.apache.hadoop.fs.FSDataOutputStream;
importorg.apache.hadoop.fs.FileSystem;
importorg.apache.hadoop.fs.Path;
```
接下来就是需要初始化hdfs，以及创造输出流等等
```
Configuration conf = initializeConf();
FileSystem hdfs = FileSystem.get(conf);
//路径
Path modelPath = new Path(hdfsPathToSaveModel);
OutputStream os = hdfs.create(modelPath);
BufferedOutputStream stream = new BufferedOutputStream(os);
ModelSerializer.writeModel(tranedNetwork, stream, true);
```
> HDFS路径通常写为  hdfs:///path/to/save/model.zip

-----
更多文档可以查看 https://github.com/sjsdfg/deeplearning4j-issues。
欢迎star

  [6]: http://static.zybuluo.com/ZzzJoe/b9ifd2wyyvr64u7tufnfyt3r/QQ%E6%88%AA%E5%9B%BE20180118131217.png