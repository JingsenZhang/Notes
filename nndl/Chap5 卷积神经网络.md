# 卷积神经网络

### 初认识

​	卷积神经网络一般是由**卷积层、汇聚层和全连接层**交叉堆叠而成的前馈神经网络，使用反向传播算法进行训练。 全连接层一般在卷积网络的最顶层。 

​	卷积神经网络有三个结构上的特性： **局部连接，权重共享以及汇聚**。

​	主要使用在图像和视频分析的各种任务上，比如图像分类、人脸识别、物体识别、图像分割等

### 1.卷积

（1）互相关和卷积的区别仅仅在于卷积核是否进行翻转（互相关运算时卷积核不翻转，即不旋转180度）。很多深度学习工具中卷积操作其实都是互相关操作。

（2）一般常用的卷积有以下三类：
	• 窄卷积：步长s = 1，两端不补零p = 0，卷积后输出 长度为**n−m+1**。

​	• 宽卷积：步长s =1，两端补零p = m−1，卷积后输出 长度n+m−1。

​	• 等宽卷积：步长s = 1，两端补零p = (m− 1)/2，卷积后输出长度n。

（3）卷积的性质：交换性、导数

### 2.卷积神经网络

（1）一般可以通过卷积操作来实现高维特征到低维特征的转换。所以可以用卷积来代替全连接 ，减少参数，提高训练效率

​		z(l) = w(l)⊗a(l−1) +b(l)

​	满足局部连接、权重共享

（2）卷积层

​	卷积层的作用是**提取一个局部区域的特征**，不同的卷积核相当于不同的特征提取器。

![1571975646074](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1571975646074.png)

![1571975673719](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1571975673719.png)

（3）汇聚层

​	a.汇聚是指对特征映射的每个区域进行下采样得到一个值，作为这个区域的概括。汇聚层也叫子采样层，其作用是进行特征选择，降低特征数量，并从而减少参数数量。

​	b.在卷积层之后加上一个汇聚层，从而降低特征维数，避免过拟合。减少特征维数也可以通过增加卷积步长来实现。 

​	c.常用的汇聚函数

![1572056258323](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1572056258323.png)

​	典型的汇聚层是将每个特征映射划分为2×2大小的不重叠区域，然后使用最大汇聚的方式进行下采样。

（4）典型的卷积网络结构



![1572057602946](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1572057602946.png)网络结构逐渐趋向于全卷积网络，减少汇聚层和全连接层的作用

### 

