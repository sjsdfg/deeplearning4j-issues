# Deeplearning4j-使用Cuda 9.1和 Cudnn7.1 加速模型训练

---

# 一、卸载Cuda （可选）

我本机原本安装的版本为 `Cuda 8.0`，因为Dl4j更新版本之后，支持 `Cuda 9.1`，因此需要先对原有软件进行卸载。

我电脑的操作系统为`win 10`，在你安装完成以后，会有如下图所示的安装软件：

<center>![cuda安装包.png-94.8kB][1]</center>

除了图中用红框标注的这三个，全部卸载。

即可完成cuda的卸载。

# 二、安装Cuda。

下载地址：

    链接：https://pan.baidu.com/s/14yvW1C3M32TZyeN-kRXEyw 密码：z9k6
    
为了保证结果的可复现。 Cuda和Cudnn的安装地址已经放在上面了。

在安装的时候，需要注意使用`自定义安装`，
<center>![cuda安装导引.png-155.9kB][2]</center>

在安装的时候需要勾掉以下属性

<center>![去掉选项.png-149.3kB][3]</center>

因为你不是 cuda 开发人员，只是使用的用户，因此不需要以下三项：

 1. Documentation: cuda开发文档
 2. Samples: cuda示例
 3. VS Studio Integration: VS开发cuda的集成插件。

也不安装`Driver components`，是害怕和你现有的软件冲突，导致显示器显示不正常。

在无限下一步之后安装完毕，在 CMD 窗口中使用`nvcc -V`命令查看 cuda 版本。
<center>![cuda版本.png-20kB][4]</center>

# 三、使用Cuda9.1加速 dl4j

dl4j使用gpu后端加速非常容易，只需要切换pom文件为：
```
<dependency>
    <groupId>org.nd4j</groupId>
    <artifactId>nd4j-cuda-9.1-platform</artifactId>
    <version>${nd4j.version}</version>
</dependency>
```
即可成功运行程序。

在运行程序之后可以看到如下的提示语：
```
o.n.l.f.Nd4jBackend - Loaded [JCublasBackend] backend
o.n.n.NativeOpsHolder - Number of threads used for NativeOps: 32
o.n.n.Nd4jBlas - Number of threads used for BLAS: 0
o.n.l.a.o.e.DefaultOpExecutioner - Backend used: [CUDA]; OS: [Windows 10]
o.n.l.a.o.e.DefaultOpExecutioner - Cores: [8]; Memory: [3.5GB];
o.n.l.a.o.e.DefaultOpExecutioner - Blas vendor: [CUBLAS]
o.n.l.j.o.e.CudaExecutioner - Device opName: [GeForce GTX 1050 Ti]; CC: [6.1]; Total/free memory: [4294967296]
o.d.n.m.MultiLayerNetwork - Starting MultiLayerNetwork with WorkspaceModes set to [training: SINGLE; inference: SEPARATE]
```
但是同样可以看到如下的报错：
```
o.d.n.l.c.ConvolutionLayer - cuDNN not found: use cuDNN for better GPU performance by including the deeplearning4j-cuda module. For more information, please refer to: https://deeplearning4j.org/cudnn
java.lang.ClassNotFoundException: org.deeplearning4j.nn.layers.convolution.CudnnConvolutionHelper
	at java.net.URLClassLoader.findClass(URLClassLoader.java:381) ~[na:1.8.0_152]
	at java.lang.ClassLoader.loadClass(ClassLoader.java:424) ~[na:1.8.0_152]
	at sun.misc.Launcher$AppClassLoader.loadClass(Launcher.java:338) ~[na:1.8.0_152]
	at java.lang.ClassLoader.loadClass(ClassLoader.java:357) ~[na:1.8.0_152]
	at java.lang.Class.forName0(Native Method) ~[na:1.8.0_152]
```
这是因为还没有安装 Cudnn 引起的报错，但是这并不影响程序的正常运行。
 
# 四、安装Cudnn并且使用Cudnn加速

## 4.1 安装 Cudnn7.1
安装Cudnn非常简单，只需要打开对应的压缩包：
<center>![cudnn压缩包.png-64.2kB][5]</center>


将图中所有的文件解压缩到`Cuda`的安装目录即可：
<center>![Cuda安装目录.png-61.2kB][6]</center>

## 4.2 使用Cudnn加速程序
使用Cudnn对dl4j程序进行加速，还需要添加以下依赖到pom文件中：
```
<dependency>
    <groupId>org.deeplearning4j</groupId>
    <artifactId>deeplearning4j-cuda-9.1</artifactId>
    <version>${dl4j.version}</version>
</dependency>
```
再次运行程序，就不会发现报错。
<center>![cudnn加速无报错.png-57.5kB][7]</center>

# 五、总结

程序运行示例为：https://github.com/sjsdfg/dl4j-tutorials/blob/master/src/main/java/lesson6/LenetMnistExample.java

即使用 Lenet 网络进行手写数字识别,**总共对全部数据集，训练1个epoch**：

机器运行配置为：**16G RAM, I7-7700HQ, 1050TI 4G显存**

|cpu|cuda9.1|cuda9.1+cudnn7.1|
|:--:|:--:|:--:|
|2.586min|0.754min|0.457min|

如果使用更好的机器，则会有更明显的加速效果。

---

更多文档可以查看 https://github.com/sjsdfg/deeplearning4j-issues。
你的star是我持续分享的动力

完整代码和pom文件可查看： https://github.com/sjsdfg/dl4j-tutorials

<center></center>


  [1]: http://static.zybuluo.com/ZzzJoe/wwsa74mlkxxd8j9xjcj6w9md/cuda%E5%AE%89%E8%A3%85%E5%8C%85.png
  [2]: http://static.zybuluo.com/ZzzJoe/r6fgavuepj0rbjhizx76o60h/cuda%E5%AE%89%E8%A3%85%E5%AF%BC%E5%BC%95.png
  [3]: http://static.zybuluo.com/ZzzJoe/o8sao2wcy843d2ad51e94kep/%E5%8E%BB%E6%8E%89%E9%80%89%E9%A1%B9.png
  [4]: http://static.zybuluo.com/ZzzJoe/pkfzy6vu1rjymbgduogkucsd/cuda%E7%89%88%E6%9C%AC.png
  [5]: http://static.zybuluo.com/ZzzJoe/k27hh5jmcongeks5emtylh20/cudnn%E5%8E%8B%E7%BC%A9%E5%8C%85.png
  [6]: http://static.zybuluo.com/ZzzJoe/elmv39bqywvgzq1ax4i6pfrp/Cuda%E5%AE%89%E8%A3%85%E7%9B%AE%E5%BD%95.png
  [7]: http://static.zybuluo.com/ZzzJoe/ar1kzs9sktt4e2ezjzstq66m/cudnn%E5%8A%A0%E9%80%9F%E6%97%A0%E6%8A%A5%E9%94%99.png