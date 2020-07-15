# PyTorch

核心：设计计算图并自动计算

## chaper2 快速入门

### 一、Tensor

可认为是一个高维数组

1.基本用法

（1）函数名后面带下划线**_** 的函数会修改Tensor本身。例如，`x.add_(y)`和`x.t_()`会改变 `x`，但`x.add(y)`和`x.t()`返回一个新的Tensor， 而`x`不变。

### 二、autograd:自动微分

在Tensor上的所有操作，ahutograd都能为它们自动提供微分，避免了手动计算导数的复杂过程。

（1）要想使得Tensor使用autograd功能，只需要设置`tensor.requries_grad=True`.

![e1572440685398](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1572440685398.png)

（2）`grad`在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以反向传播之前需把梯度清零。

```
x.grad.data.zero_()
```

### 三、神经网络

​	torch.nn是专门为神经网络设计的模块化接口。

​	nn构建于 Autograd之上，可用来定义和运行神经网络。nn.Module是nn中最重要的类，可把它看成是一个网络的封装，包含网络各层定义以及forward方法，调用forward(input)方法，可返回前向传播的结果。

## chaper3 基础：Tensor 和 Autograd

### 一、Tensor

1.

+ t.tensor()是一个函数，括号内是一个具体的data, 返回的是一个tensor。

  torch.tensor(data, dtype=None, device=None, requires_grad=False) 

  a=t.tensor()  括号中只允许有一个data参数如[[1,2],[3,4]]                   [1,2],[3,4]则不正确

+ torch.Tensor是主要的tensor类，所有的tensor都是torch.Tensor的实例。

  可以定义所产生的tensor实例的形状

  + b=t.Tensor(20)      一维tensor（20个元素）
  + c=t.Tensor(20,10)   二维tensor（20行，10列）

2.通过`tensor.view`方法可以调整tensor的形状，但必须保证调整前后元素总数一致。`view`不会修改自身的数据，返回的**新tensor与源tensor共享内存，也即更改其中的一个，另外一个也会跟着改变**。在实际应用中可能经常需要添加或减少某一维度，这时候`unsqueeze`和`squeeze`两个函数就派上用场了。

3.索引出来的结果与原tensor共享内存，也即修改一个，另一个会跟着修改。

4.`gather`把数据从input中按index取出，而`scatter_`是把取出的数据再放回去

5.归并操作中大多数函数都有一个参数**dim**，0表示对列上进行（第一个维度），1表示行上进行（第二个维度）

6.tensor与numpy共享内存（数据类型不一致时不共享）

​    转换：

a为array类型：     c=t.Tensor(a)     则c为tensor(此时c的类型为默认值)     

​				或c=t.tensor(a)     一律不共享内存

​                                或  c=t.from_numpy(a) （类型与a一致）

b=c.numpy()     则b为array

7.广播法则

Numpy的广播法则定义如下：

- 让所有输入数组都向其中shape最长的数组看齐，shape中不足的部分通过在前面加1补齐
- 两个数组要么在某一个维度的长度一致，要么其中一个为1，否则不能计算
- 当输入数组的某个维度的长度为1时，计算时沿此维度复制扩充成一样的形状

PyTorch当前已经支持了自动广播法则，但是笔者还是建议读者通过以下两个函数的组合手动实现广播法则，这样更直观，更不易出错：

- `unsqueeze`或者`view`，或者tensor[None],：为数据某一维的形状补1，实现法则1
- `expand`或者`expand_as`，重复数组，实现法则3；该操作不会复制数组，所以不会占用额外的空间。

注意，repeat实现与expand相类似的功能，但是repeat会把相同数据复制多份，因此会占用额外的空间。

8.索引

​	高级索引一般不共享stroage，而普通索引共享storage（提示：普通索引可以通过只修改tensor的offset，stride和size（头信息），而不修改storage来实现）。

### 二、Autograd

1.   y.backward() 三个参数

+ grad_variables: 梯度的系数 ,与中间变量的形状一致

  y为标量时，可直接y.backward()

  y为矢量时，可y.backward(t.ones(y.size()))

+ retain_graph:不清空计算图缓存，默认false

+ create_graph:再次构建计算图

2. variable结构

+ data 数值

+ grad 梯度  非叶子节点为None

+ grad_fn 得到此变量需经过的function，叶子结点为None

   grad_fn_nextfunctions指grad_fn的输入函数,叶子结点为accumulate

## chaper4 神经网络

​	torch.nn的核心数据结构是`Module`，它是一个抽象概念，既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。

​	在实际使用中，最常见的做法是继承`nn.Module`，撰写自己的网络/层。

#### 1.对于前馈网络如果每次都写复杂的forward函数会有些麻烦

​		两种简化方式，ModuleList和Sequential。

+ Sequential是一个特殊的module，它包含几个子Module，前向传播时会将输入一层接一层的传递下去。

```python
net1 = nn.Sequential()
net1.add_module('conv', nn.Conv2d(3, 3, 3))
net1.add_module('batchnorm', nn.BatchNorm2d(3))
net1.add_module('activation_layer', nn.ReLU())
#或
net2 = nn.Sequential(
        nn.Conv2d(3, 3, 3),
        nn.BatchNorm2d(3),
        nn.ReLU()
        )
```

+ ModuleList也是一个特殊的module，可以包含几个子module，可以像用list一样使用它，但不能直接把输入传给ModuleList。

```python
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.module_list = nn.ModuleList([nn.Conv2d(3, 3, 3), nn.ReLU()])
    def forward(self):
        pass
model=MyModule()
input = t.randn(3,3,3,3)
for m in model.module_list:
    input = m(input)
output = model.module_list(input)
```

#### 2.卷积层：

nn.Con2d(输入维度，输出维度，卷积核，步长 默认是1，补0，核间距，groups=1,bias=True)

池化层没有可学习参数

#### 3.循环神经网络：nn.RNN()

（1）LSTM

标准的循环神经网络内部只有一个简单的层结构，而 LSTM 内部有 4 个层结构：

第一层是个忘记层：决定状态中丢弃什么信息

第二层tanh层用来产生更新值的候选项，说明状态在某些维度上需要加强，在某些维度上需要减弱

第三层sigmoid层（输入门层），它的输出值要乘到tanh层的输出上，起到一个缩放的作用，极端情况下sigmoid输出0说明相应维度上的状态不需要更新

最后一层 决定输出什么,输出值跟状态有关。候选项中的哪些部分最终会被输出由一个sigmoid层来决定。

+ nn.LSTM()

  input_size:输入特征的数目, 即每一行输入元素的个数
  hidden_size:隐层的特征数目, 即隐藏层节点的个数
  num_layers：这个是模型集成的LSTM的个数 记住这里是模型中有多少个LSTM摞起来 一般默认就1个  （递归的原因所在）
  bias：用不用偏置 默认是用
  batch_first:默认为假 若为真，则输入、输出的tensor的格式为(batch , seq , feature)
  即[batch_size, seq, hidden_size] 【batch大小，序列长度，特征数目】
  dropout:默认0 若非0，则为dropout率
  bidirectional：是否为双向LSTM 默认为否

+ 输入数据格式：
  input(seq_len, batch, input_size)
  h0(num_layers * num_directions, batch, hidden_size)
  c0(num_layers * num_directions, batch, hidden_size)

+ 输出数据格式：
  output(seq_len, batch, hidden_size * num_directions)
  hn(num_layers * num_directions, batch, hidden_size)
  cn(num_layers * num_directions, batch, hidden_size)

#### 4.词向量（nlp常用）

**torch.nn.Embedding(num_embeddings, embedding_dim)**	

(1)创建一个词嵌入模型

num_embeddings代表一共有多少个词，embedding_dim代表你想要为每个词创建一个多少维的向量来表示它

(2)输入为两个维度(batch的大小，每个batch的单词个数)，输出则在两个维度上加上词向量的大小。

Input: LongTensor (N, W), N = mini-batch, W = number of indices to extract per mini-batch
Output: (N, W, embedding_dim)

#### 5.交叉熵损失函数

```
# batch_size=3，计算对应每个类别的分数（只有两个类别）
score = t.randn(3, 2)    
# 三个样本分别属于1，0，1类，label必须是LongTensor
label = t.Tensor([1, 0, 1]).long()
# loss与普通的layer无差异
criterion = nn.CrossEntropyLoss()
loss = criterion(score, label)
```

均方差损失函数使用：

```criterion=nn.MSELoss()
criterion=nn.MSELoss()
loss=criterion(pred,y)
```



#### 6.优化器torch.optim.SGD(params,lr)

```
optimizer =t.optim.SGD([
                {'params': net.features.parameters()}, # 学习率为1e-5
                {'params': net.classifier.parameters(), 'lr': 1e-2}
            ], lr=1e-5)
optimizer.step()          
```

损失函数+优化器

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

pred = linear(x)

loss = criterion(pred, y)
print('loss: ', loss.item())

loss.backward()

print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)

optimizer.step()

pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())

#### 7.钩子 hook

​	钩子函数主要用在获取某些中间结果的情景，如中间某一层的输出或某一层的梯度。

​	下面考虑一种场景，有一个预训练好的模型，需要提取模型的某一层（不是最后一层）的输出作为特征进行分类，但又不希望修改其原有的模型定义文件，这时就可以利用钩子函数。

#### 8.保存模型

​	所有的Module对象都具有state_dict()函数，返回当前Module所有的状态数据。将这些状态数据保存后，下次使用模型时即可利用model.load_state_dict()函数将状态加载进来。

```python
#保存模型
t.save(net.state_dict(), 'net.pth')
# 加载已保存的模型
net2 = Net()
net2.load_state_dict(t.load('net.pth'))
```

## chaper5 常用工具模块

#### 1.数据处理

+ Dataset
+ Dataloader 数据加载
+ sampler 采样

#### 2.计算机视觉工具包torchvision

（1）torchvision主要包含三部分：

+ models：提供深度学习中各种经典网络的网络结构以及预训练好的模型，包括AlexNet、VGG系列、ResNet系列、Inception系列等。

+ datasets： 提供常用的数据集加载，设计上都是继承torch.utils.data.Dataset，主要包括MNIST、CIFAR10/100、ImageNet、COCO等。

+ transforms：提供常用的数据预处理操作，主要包括对Tensor以及PIL Image对象的操作。

  Transforms中涵盖了大部分对Tensor和PIL Image的常用处理。转换分为两步，第一步：构建转换操作，例如`transf = transforms.Normalize(mean=x, std=y)`，第二步：执行转换操作，例如`output = transf(input)`。另外还可将多个处理操作用Compose拼接起来，形成一个处理转换流程。

（2）torchvision还提供了两个常用的函数

+ `make_grid`，它能将多张图片拼接成一个网格中
+ `save_img`，它能将Tensor保存成图片。

#### 3.可视化工具

（1）Tensorboard

Tensorboard的安装主要分为以下两步：

- 安装TensorFlow：如果电脑中已经安装完TensorFlow可以跳过这一步，如果电脑中尚未安装，建议安装CPU-Only的版本，具体安装教程参见TensorFlow官网[^1]，或使用pip直接安装，推荐使用清华的软件源[2](https://render.githubusercontent.com/view/ipynb?commit=f765d3ff873b23c86838f1b9fae79ca938ca8af9&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6368656e79756e74632f7079746f7263682d626f6f6b2f663736356433666638373362323363383638333866316239666165373963613933386361386166392f63686170746572352d2545352542382542382545372539342541382545352542372541352545352538352542372f63686170746572352e6970796e62&nwo=chenyuntc%2Fpytorch-book&path=chapter5-%E5%B8%B8%E7%94%A8%E5%B7%A5%E5%85%B7%2Fchapter5.ipynb&repository_id=92265140&repository_type=Repository#fn-2)。
- 安装tensorboard: `pip install tensorboard`
- 安装tensorboardX：可通过`pip install tensorboardX`命令直接安装。

tensorboardX的使用非常简单。首先用如下命令启动tensorboard：

```
tensorboard --logdir <your/running/dir> --port <your_bind_port>

#在tensorflow环境下
#例如 tensorboard --logdir  G:\jupyter_files\PyTorch_Test\tensorboard_test\images
#默认端口6006
```



使用：

+ 终端打开tensorboard

tensorboard --logdir=D://repository//tensorflowbook//chapters//test (绝对路径)

tensorboard --logdir=./test  (调整终端位置之后的相对路径)

+ 代码

```python 
from tensorboardX import SummaryWriter

# 构建logger对象，logdir用来指定log文件的保存路径
# flush_secs用来指定刷新同步间隔(默认120)
logger=SummaryWriter(log_dir='test',flush_secs=2)

for ii in range(100):
    logger.add_scalar('data/loss',10-ii**0.5)
    logger.add_scalar('data/accuracy',ii**0.5/10)
    
```

* 浏览器

http://localhost:6006 



（2）Visdom

+ Visdom中有两个重要概念：

  ​	env：环境。不同环境的可视化结果相互隔离，互不影响，在使用时如果不指定env，默认使用`main`。不同用户、不同程序一般使用不同的env。

  ​	pane：窗格。窗格可用于可视化图像、数值或打印文本等，其可以拖动、缩放、保存和关闭。一个程序中可使用同一个env中的不同pane，每个pane可视化或记录某一信息。

+ `python -m visdom.server`命令启动visdom服务

  `nohup python -m visdom.server &`命令将服务放至后台运行。

  Visdom服务是一个web server服务，默认绑定8097端口，

+ ```python
  import visdom
  
  # 新建一个连接客户端！！！！！！！！！！！！！！！！！！！！
  # 指定env = u'test1'，默认端口为8097，hos是‘localhost' ！！！！！！！！！！！！
  vis = visdom.Visdom(env=u'test1',use_incoming_socket=False)
  
  x = t.arange(1, 30, 0.01)
  y = t.sin(x)
  vis.line(X=x, Y=y, win='sinx', opts={'title': 'y=sin(x)'})
  ```

+ vis.line

  主要参数

  + win：用于指定pane的名字，如果不指定，visdom将自动分配一个新的pane。如果两次操作指定的win名字一样，新的操作将覆盖当前pane的内容，因此建议每次操作都重新指定win。
  + opts：选项，接收一个字典，常见的option包括title、xlabel、ylabel、width等，主要用于设置pane的显示格式。

  我们在训练网络的过程中需不断更新数值，如损失值等，这时就需要指定参数`update='append'`来避免覆盖之前的数值。

  使用`vis.updateTrace`方法来更新图(update='new',name='xxxxx')，`updateTrace`不仅能在指定pane上新增一个和已有数据相互独立的Trace，还能像`update='append'`那样在同一条trace上追加数据。

  ```python
  # append 追加数据
  for ii in range(0, 10):
      # y = x
      x = t.Tensor([ii])
      y = x
      vis.line(X=x, Y=y, win='polynomial', update='append' if ii>0 else None)
      
  # updateTrace 新增一条线
  x = t.arange(0, 9, 0.1)
  y = (x ** 2) / 9
  vis.line(X=x, Y=y, win='polynomial', name='this is a new Trace',update='new')
  ```

+ vis.image  画图功能

  ```python
  # 可视化一个随机的黑白图片
  vis.image(t.randn(64, 64).numpy())
  
  # 随机可视化一张彩色图片
  vis.image(t.randn(3, 64, 64).numpy(), win='random2')
  
  # 可视化36张随机的彩色图片，每一行6张
  vis.images(t.randn(36, 3, 64, 64).numpy(), nrow=6, win='random3', opts={'title':'random_imgs'})
  ```

+ `vis.text`可视化文本

  ​	支持所有的html标签，同时也遵循着html的语法标准。例如，换行需使用`<br>`标签，`\r\n`无法实现换行。

#### 4.使用GPU加速：cuda

#### 5.持久化

在PyTorch中，以下对象可以持久化到硬盘，并能通过相应的方法加载到内存中：

- Tensor
- Variable
- nn.Module
- Optimizer

`t.save(obj, file_name)`等方法保存任意可序列化的对象

`obj = t.load(file_name)`方法加载保存的数据

对于Module和Optimizer对象，这里建议保存对应的`state_dict`，而不是直接保存整个Module/Optimizer对象。Optimizer对象保存的主要是参数，以及动量信息，通过加载之前的动量信息，能够有效地减少模型震荡。

```python
# Module对象的保存与加载
t.save(model.state_dict(), 'squeezenet.pth')
model.load_state_dict(t.load('squeezenet.pth'))
```

## chaper 6 补充

#### 1.detach()起到截断反向传播的梯度流的作用

​	返回一个新的Variable，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，即使之后重新将它的requires_grad置为true,它也不会具有梯度grad

如：predicted=model(t.from_numpy(x_train)).detach().numpy()

numpy()前面只能跟requires_grad为false的tensor

#### 2.数据集加载

```python
#MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                      	   transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())
#Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
```

#### 3.torch.max()

+ 返回一个tensor中的最大值，如：

  a=t.tensor([[ 1.1659, -1.5195,  0.0455,  1.7610, -0.2064],
          [-0.3443,  2.0483,  0.6303,  0.9475,  0.4364],
          [-1.5268, -1.0833,  1.6847,  0.0145, -0.2088],
          [-0.8681,  0.1516, -0.7764,  0.8244, -1.2194]])

  \>>> print(torch.max(a))
  tensor(2.0483)

+ 这个函数的参数中还有一个dim参数，指定作比较的维度，Tensor的维度从0开始算起

  （0：每列最大值，1：每行最大值）

  使用方法为re = torch.max(Tensor,dim),返回的re为一个二维向量，其中re[0]为最大值（组成）的Tensor，re[1]为最大值对应的index（组成）的Tensor。例如：

  ![1573385818616](C:\Users\18742\AppData\Roaming\Typora\typora-user-images\1573385818616.png)

  #### 4.model.train()与model.eval()

  主要是针对model 在训练时和评价时不同的 Batch Normalization 和 Dropout 方法模式。

  model.eval()，让model变成测试模式，pytorch会自动把Batch Normalization 和 Dropout 固定住，不会取平均，而是用训练好的值。

  eg：

  #定义模型

  Class Inpaint_Network()

  ......

  Model = Inpaint_Nerwoek()

  \#train:

  Model.train(mode=True)

  ......

  \#test:

  Model.eval()

  ......

  #### 5.只有0维才可用item()

  * a=t.tensor(200)   a.item()=200
  * b=t.temsor([200])   b[0]=tensor(200)    b[0].item()=200
  * c=t.tensor([20,30])  c.size(0)=c.size()[0]=2    获取c中元素个数

  