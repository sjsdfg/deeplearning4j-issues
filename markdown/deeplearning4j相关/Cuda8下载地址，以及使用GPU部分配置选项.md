[TOC]

# 1 Cuda安装

dl4j不支持8.0以上的版本，为此首先需要对cuda进行安装

## 1.1 win10版本链接
[cuda_8.0.44_win10-exe][1]

## 1.2 linux版本链接

[cuda_8.0.61_375.26_linux-run][2]

## 1.3 win7版本链接(尚未验证)
[百度云cuda地址，cuda_8.0.44_windows.exe][3]

## 1.4 注意事项

 1. GTX1050ti以下版本显卡可能安装失败
 2. 安装时候不要使用默认安装，应当选择自定义安装。将dirver勾选消除。（有可能cuda带的驱动版本和电脑自带的驱动版本有冲突）
 
## 安装检测

安装完成之后，如果是windows平台则需要在命令行中输入以下命令
```
nvcc  -V
```
如果命令行能够显示cuda的版本信息则表示安装成功。
例如博客中一个图片：
![安装成功][4]


# 2 Dl4j项目依赖

```
<dependencies>
    <dependency>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-native-platform</artifactId>
        <version>${nd4j.version}</version>
    </dependency>
    <dependency>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-cuda-7.5-platform</artifactId>
        <version>${nd4j.version}</version>
    </dependency>
    <dependency>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-cuda-8.0-platform</artifactId>
        <version>${nd4j.version}</version>
    </dependency>
</dependencies>
```
dl4j目前提供的平台依赖jar包主要有以上三个

## 2.1 CPU支持
```
<dependency>
    <groupId>org.nd4j</groupId>
    <artifactId>nd4j-native-platform</artifactId>
    <version>${nd4j.version}</version>
</dependency>
```
## 2.2 cuda7.5支持
```
<dependency>
    <groupId>org.nd4j</groupId>
    <artifactId>nd4j-cuda-7.5-platform</artifactId>
    <version>${nd4j.version}</version>
</dependency>
```

## 2.3 cuda8.0支持

```
<dependency>
    <groupId>org.nd4j</groupId>
    <artifactId>nd4j-cuda-8.0-platform</artifactId>
    <version>${nd4j.version}</version>
</dependency>
```

# 3 代码配置

## 3.1 配置数据类型
```
DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);
```
这句代码提供的是GPU运算时的数据类型，还提供如下选择
```
enum Type {
    DOUBLE, FLOAT, INT, HALF, COMPRESSED
}

enum TypeEx {
    FLOAT8, INT8, UINT8, FLOAT16, INT16, UINT16, FLOAT, DOUBLE
}
```

## 3.2 配置GPU选项
If you have several GPUs, but your system is forcing you to use just one, there’s a solution. Just add`CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true);`as first line of your `main()`method.
```
CudaEnvironment.getInstance().getConfiguration()
        // 如果有多个GPU则可以开启
        .allowMultiGPU(true)

        //设置最大的显存分配了，取决于显卡的显存大小
        .setMaximumDeviceCache(2L * 1024L * 1024L * 1024L)

        // cross-device access is used for faster model averaging over pcie
        .allowCrossDeviceAccess(true);
```

## 3.3 使用GPU训练模型
```
// ParallelWrapper will take care of load balancing between GPUs.
ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
    // DataSets预取选项。 根据实际设备的数量设置此值
    .prefetchBuffer(24)

    // 设置数量等于或高于可用设备的数量。 x1-x2是很好的值
    .workers(4)

    // 少量的平均可以提高性能，但可能会降低模型精度
    .averagingFrequency(3)

    // 如果设置为TRUE，则会报告每个平均模型得分
    .reportScoreAfterAveraging(true)

    // 可选参数，如果您的系统支持跨PCIe的P2P内存访问，则设置为false（提示：AWS不支持P2P）
    .useLegacyAveraging(true)

    .build();
```

  [1]: https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_win10-exe
  [2]: https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
  [3]: https://pan.baidu.com/s/1slpM1sD
  [4]: http://img.blog.csdn.net/20170730102504210?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaml1Z2VzaGFv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center
