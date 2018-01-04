#Intel MKL FATAL ERROR:Cannot load mkl_intel_thread.dll

在windows上运行deeplearning4j的时候可能遇到`Intel MKL FATAL ERROR:Cannot load mkl_intel_thread.dll`错误导致JVM故障。

---
在官网上面所给出的[解决方案][1]是：
Maven Central当前的rc3.10版本中，libnd4j库在路径中发现英特尔的MKL时无法将其正确加载。解决方法是添加 `System.loadLibrary("mkl_rt")`。 
但是很多人使用官网上面提供的方法并不能解决该问题。

这里在gitter中聊天中提供另外一种解决方法：

1. 使用win + R，输入cmd打开windows下的命令行窗口
2. 使用`where mkl_rt.dll`命令查找该dll的对应路径
3. 将该路径从PATH环境变量中移除，或者删除`mkl_rt.dll`文件

![QQ截图20180104201412.png-5.9kB][2]

应当在目录中无`mkl_rt.dll`即可正常运行
**注：**
该问题可能是由用户安装`anaconda`导致的dll冲突。


  [1]: https://deeplearning4j.org/cn/quickstart
  [2]: http://static.zybuluo.com/ZzzJoe/jf0bdo6gkyadddilehzwdktl/QQ%E6%88%AA%E5%9B%BE20180104201412.png