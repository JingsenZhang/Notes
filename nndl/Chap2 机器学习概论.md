# 机器学习概论

## 1.机器学习

​	机器学习是一门讨论各式各样的适用于不同问题的函数形式，以及如何使用数据来有效地获取函数参数具体值的学科。深度学习是指机器学习中的一类函数，它们的形式通常为多层神经网络。

​	从数据中获得决策（预测）函数使得机器可以根据数据进行自动学习，通过算法使得机器能从大量历史数据中学习规律从而对新的样本做决策。

## 2.类型

![1570935144829](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1570935144829.png)

## 3.集成模型

通过多个高方差模型的平均来降低方差

## 4.PAC学习（Probably Approximately Correct）

根据大数定律，当训练集大小|D|趋向无穷大时，泛化错误趋向于0，即经验风险趋近于期望风险。

![1570936131172](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1570936131172.png)

![1570936161133](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1570936161133.png)

PAC学习理论可以帮助分析一个机器学习方法在什么条件下可以学习到一个近似正确的分类器。如果希望模型的假设空间越大，泛化错误越小，其需要的样本数量越多。

## 5.常见的激活函数（连续并可导的非线性函数、单调递增）

#### a.ReLU函数

![1571210857561](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1571210857561.png)

函数图像及其导数图像

![1571210955804](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1571210955804.png)

![1571210974873](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1571210974873.png)

#### b.sigmoid函数

![1571211088620](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1571211088620.png)

函数图像

![1571211108043](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1571211108043.png)

导数及其图像![1571211132823](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1571211132823.png)

![1571211145775](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1571211145775.png)

#### c.tanh函数

![1571211270978](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1571211270978.png)

函数图像

![1571211427271](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1571211427271.png)

导数及其图像![1571211447791](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1571211447791.png)

![1571211468168](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1571211468168.png)

## 6.误差

#### 1.种类

+ 训练误差：模型在训练数据集上表现出的误差

+ 泛化误差：模型在任意一个测试数据样本上表现出的误差的期望，并常常通过测试数据集上的误差来近似。机器学习模型应关注降低泛化误差。

​	计算训练误差和泛化误差可以使用之前介绍过的损失函数，例如线性回归用到的平方损失函数和softmax回归用到的交叉熵损失函数。

​	可以使用验证数据集来进行模型选择（或K折交叉验证等方法）

#### 2.两种现象

+ 欠拟合：模型无法得到较低的训练误差（underfitting）
+ 过拟合：模型的训练误差远小于它在测试数据集上的误差（overfitting）。

###### 影响因素：

+ 模型复杂度

  ​	例如：高阶多项式函数模型参数更多，模型函数的选择空间更大，所以高阶多项式函数比低阶多项式函数的复杂度更高（可理解为更精确）。因此，高阶多项式函数比低阶多项式函数更容易在相同的训练数据集上得到更低的训练误差。

  ​	给定训练数据集，如果模型的复杂度过低，很容易出现欠拟合；如果模型复杂度过高，很容易出现过拟合。应对欠拟合和过拟合的一个办法是针对数据集选择合适复杂度的模型。如图：

  ![1571213453135](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1571213453135.png)

+ 训练数据集大小

  ​	一般来说，如果训练数据集中样本数过少，特别是比模型参数数量（按元素计）更少时，过拟合更容易发生。此外，泛化误差不会随训练数据集里样本数量增加而增大。因此，在计算资源允许的范围之内，我们通常希望训练数据集大一些，特别是在模型复杂度较高时，例如层数较多的深度学习模型。