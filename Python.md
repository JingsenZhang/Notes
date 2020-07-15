# <一>Numpy学习

#### 一、Python基础语法

##### 1.数据类型

（1）数值型

+ `print((x**2))`    2次方、3次方....

+ 无自增自减运算

（2）布尔类型

```python
print(t and f) # Logical AND;  注释用 #
print(t or f)  # Logical OR; 
print(not t)   # Logical NOT; 
print(t != f)  # Logical XOR;
```

（3）字符串

```python
s1 = 'hello'    # String literals can use single quotes
s2 = "world"    # or double quotes; it does not matter.
print(len(s1))  # String length; prints "5"
s3 = s1 + ' ' + s2  # String concatenation
print(s3)  # prints "hello world"
s4 = '%s %s %d' % (s1, s2, 4)  # sprintf style string formatting
print(s4)  # prints "hello world 4"
```

很多很多字符串的方法，如：

```python
s = "hello"
print(s.capitalize())  # Capitalize a string; prints "Hello"
print(s.upper())       # Convert a string to uppercase; prints "HELLO"
print(s.rjust(7))      # 右对齐Right-justify a string, padding with spaces; prints "  hello"
print(s.center(7))     # 居中; prints " hello "
print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another;
                                # prints "he(ell)(ell)o"
print('  world '.strip())  # 去除空格 Strip leading and trailing whitespace; prints "world"
```

##### 2.容器

（1）Lists 列表

+ 列表类似数组，但可调整大小，并且可以包含不同类型的元素：

  ```
  xs = [3, 1, 2]    # Create a list
  print(xs, xs[2])  # Prints "[3, 1, 2] 2"  索引从0开始
  print(xs[-1])     # Negative indices count from the end of the list; prints "2"  -1为倒数第一个元素
  xs[2] = 'foo'     # Lists can contain elements of different types
  print(xs)         # Prints "[3, 1, 'foo']"
  xs.append('bar')  # Add a new element to the end of the list
  print(xs)         # Prints "[3, 1, 'foo', 'bar']"
  x = xs.pop()      # Remove and return the last element of the list
  print(x, xs)      # Prints "bar [3, 1, 'foo']"
  ```

+ 切片：访问子列表

```python
nums = list(range(5))     # range is a built-in function that creates a list of integers   list创建列表 range(5)从0开始5个数
print(nums)               # Prints "[0, 1, 2, 3, 4]"
print(nums[2:4])          # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"     取前不取后！！！！
print(nums[2:])           # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(nums[:2])           # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(nums[:])            # Get a slice of the whole list; prints "[0, 1, 2, 3, 4]"
print(nums[:-1])          # Slice indices can be negative; prints "[0, 1, 2, 3]"
nums[2:4] = [8, 9]        # Assign a new sublist to a slice
print(nums)               # Prints "[0, 1, 8, 9, 4]"
```

+ 循环

  ```
  animals = ['cat', 'dog', 'monkey']
  for i in animals:
      print(i)
  for idx, i in enumerate(animals):
      print('%d %s' % (idx + 1, i))   #enumerate函数：访问循环体内每个元素的索引
  ```

+ 理解：列表中可以添加各种条件

  ```python
  nums = [0, 1, 2, 3, 4]
  squares = [x ** 2 for x in nums]
  print(squares)   # Prints [0, 1, 4, 9, 16]
  even_squares = [x ** 2 for x in nums if x % 2 == 0]
  print(even_squares)  # Prints "[0, 4, 16]"
  ```


（2）Dictionaries字典

+ 遍历

  ```python
  d = {'person': 2, 'cat': 4, 'spider': 8}
  for animal, legs in d.items():
      print('A %s has %d legs' % (animal, legs))
  #或
  for animal in d:
      legs = d[animal]
      print('A %s has %d legs' % (animal, legs))
  ```

+ 与list联系

  ```python
  nums = [0, 1, 2, 3, 4]
  even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
  print(even_num_to_square)  # Prints "{0: 0, 2: 4, 4: 16}"
  ```

（3）Sets集合  

​	不同元素的无序集合（a.不存在重复的元素  b.元素是无序的）

+ ```python
  animals = {'cat', 'dog'}
  print('cat' in animals)   # Check if an element is in a set; prints "True"
  print(len(animals))       # Number of elements in a set; prints "3"
  animals.add('fish')       # Add an element to a set
  animals.add('cat')        # Adding an element that is already in the set does nothing!!!
  animals.remove('cat')     # Remove an element from a set
  ```

（4）Tuples元组

​	元组是（不可变的）有序值列表，在很多方面类似于列表。最重要的区别是，元组可以用作字典中的键和集合中的元素，而列表则不能。

```python
d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
t = (5, 6)        # Create a tuple
print(type(t))    # Prints "<class 'tuple'>"
print(d[t])       # Prints "5"
print(d[(1, 2)])  # Prints "1"
```

##### 3.函数

def

```python
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'
for x in [-1, 0, 1]:
    print(sign(x))
# Prints "negative", "zero", "positive"
```

##### 4.类

+ 不含参的构造方法

  ```python
  class Greeter():
      # Constructor
      def __init__(self):
          self.data=[]
      # Instance method
      def greet(self,loud=False):
          if loud:
              print('HELLO')
          else:
              print('Hello')
  
  g = Greeter()  # Construct an instance of the Greeter class
  g.greet()            # Call an instance method; prints "Hello"
  g.greet(loud=True)   # Call an instance method; prints "HELLO"
  ```

+ 含参数的构造方法

```python
class Greeter(object):
    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable
    # Instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"

```

#### 二、Numpy

##### 1.Arrays

Numpy数组是相同类型的值的网格，并由非负整数元组索引如a[2,0]。

The number of dimensions is the *rank* of the array. 

数组的形状是一个整数元组，给出沿每个维度的数组大小。如（3,1）（3，）

创建：

+ 嵌套列表

  ```python
  import numpy as np
  a = np.array([1, 2, 3])   # Create a rank 1 array
  print(type(a))            # Prints "<class 'numpy.ndarray'>"
  print(a.shape)            # Prints "(3,)"
  print(a[0], a[1], a[2])   # Prints "1 2 3"
  a[0] = 5                  # Change an element of the array
  print(a)                  # Prints "[5, 2, 3]"
  
  b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
  print(b.shape)                     # Prints "(2, 3)"
  print(b[0, 0], b[0, 1], b[1, 0])   # Prints "1 2 4"
  ```

+ 内置函数

  ```python
  import numpy as np
  a = np.zeros((2,2))   # Create an array of all zeros  默认dtype为float64
  print(a)              # Prints "[[ 0.  0.]
                        #          [ 0.  0.]]"
  b = np.ones((1,2))    # Create an array of all ones
  print(b)              # Prints "[[ 1.  1.]]"
  
  c = np.full((2,2), 7)  # Create a constant array
  print(c)               # Prints "[[ 7.  7.]
                         #          [ 7.  7.]]"
  d = np.eye(2)         # Create a 2x2 identity matrix
  print(d)              # Prints "[[ 1.  0.]
                        #          [ 0.  1.]]"
  e = np.random.random((2,2))  # Create an array filled with random values
  print(e)                     # Might print "[[ 0.91940167  0.08143941]
                               #               [ 0.68744134  0.87236687]]"
  #arange（）将创建具有规则递增值的数组。
  >>> np.arange(10)
  array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  >>> np.arange(2, 10, dtype=float)   #取前不取后
  array([ 2., 3., 4., 5., 6., 7., 8., 9.])
  >>> np.arange(2, 3, 0.1)			#第三个参数为步长（可以为负，表示减小）
  array([ 2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9])
  
  #linspace（）将创建具有指定数量元素的数组，并在任意指定的开始值和结束值之间等距间隔。
  >>> np.linspace(1., 4., 6)
  array([ 1. ,  1.6,  2.2,  2.8,  3.4,  4. ])
  
  #indices（）将创建一组数组（堆叠为一维数组），每个维表示该维的变化。这对于评估常规网格上的多维功能特别有用。
  >>> np.indices((3,3))
  array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]]])
  
  x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
  y = np.empty_like(x)   # Create an empty matrix with the same shape as x   用0填充
  
  ```

##### 2.Arrays indexing

(1)切片索引 slice

与list切片类似，只不过可能是多个维度上的操作

修改切片的元素也会改变原始数组

``` python
import numpy as np
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
b = a[:2, 1:3]
# [[2 3]，[6 7]]
# A slice of an array is a view into the same data, so modifying it will modify the original array.
print(a[0, 1])   # Prints "2"
b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1])   # Prints "77"
```

（2）整数索引

​	 使用切片索引到numpy数组时，所得的数组视图将始终是原始数组的子数组。相反，整数数组索引使您可以使用一个数组中的数据构造任意数组。

​	整数数组索引的一个有用技巧是从矩阵的每一行中选择或更改一个元素

```python
import numpy as np
a = np.array([[1,2], [3, 4], [5, 6]])
# An example of integer array indexing.
print(a[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"
# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]])) 
# Prints "[1 4 5]"
# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])  # Prints "[2 2]"
# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]]))  # Prints "[2 2]"
```

（3）整数索引与切片索引结合

```python
import numpy as np
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"
```

（4）布尔数组索引

​	布尔数组索引使您可以挑选出数组的任意元素。通常，这种类型的索引用于选择满足某些条件的数组元素。

```python
import numpy as np
a = np.array([[1,2], [3, 4], [5, 6]])
bool_idx = (a > 2)   
print(bool_idx)      # Prints "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"
print(a[bool_idx])  # Prints "[3 4 5 6]"
# We can do all of the above in a single concise statement:
print(a[a > 2])     # Prints "[3 4 5 6]"
```

##### 3.Datatypes

​	在创建数组时，构造数组的函数通常还包含一个可选参数，以明确指定该数据类型。

```python
x = np.array([1.0, 2.0])   # Let numpy choose the datatype
print(x.dtype)             # Prints "float64"
x = np.array([1, 2], dtype=np.int64)   
							# Force a particular datatype
print(x.dtype)				# Prints "int64"
```

##### 4.Array math

（1）基础运算（运算符重载或内置函数）

```python
import numpy as np
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)
# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(x + y)
print(np.add(x, y))
# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(x - y)
print(np.subtract(x, y))
# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(x * y)   #逐元素乘法
print(np.multiply(x, y))
# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))
# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))
```

（2）dot  用于求向量内积或矩阵乘法

```python
import numpy as np
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
v = np.array([9,10])
w = np.array([11, 12])
# Inner product of vectors; both produce 219
print(v.dot(w))
print(np.dot(v, w))
# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))
# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))
```

######            * 运算、dot()运算、multiply()运算的区别

1. 当dot()作用在数组类型或list类型时，两个一维数组时，结果为内积；其他为矩阵乘（即矢量积，叉乘）

   当dot()作用在矩阵类型时（无论矩阵规模），两个矩阵要满足矩阵乘的行列要求，结果为矩阵乘（即矢量积，叉乘）

2. *作用在数组类型(不能作用在list类型)时为点乘（即矩阵对应元素相乘）

   *作用在矩阵类型时为矩阵乘（即矢量积，叉乘）

3. multiply()无论作用在数组类型、矩阵类型、list类型，结果都为数量积，点乘（即矩阵对应元素相乘），有广播现象

（3）sum

```python
import numpy as np
x = np.array([[1,2],[3,4]])
print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"
print(np.mean(x))  #均值
print(np.argmax(x)) #值最大的元素的下标
```

（4）T 转置

```python
import numpy as np
x = np.array([[1,2], [3,4]])
print(x)    # Prints "[[1 2]
            #          [3 4]]"
print(x.T)  # Prints "[[1 3]
            #          [2 4]]"
# Note that taking the transpose of a rank 1 array does nothing:
v = np.array([1,2,3])
print(v)    # Prints "[1 2 3]"
print(v.T)  # Prints "[1 2 3]"
```

##### 5.Broadcasting

​	广播通常会使您的代码更简洁，更快捷

​	广播允许numpy在执行算术运算时使用不同形状的数组。通常，我们有一个较小的数组和一个较大的数组，并且我们想多次使用较小的数组对较大的数组执行某些操作。如：

+ 向矩阵的每一行添加一个常数向量

  法一：循环

```python
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   
# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v
```

​	法二：tile()函数

​	np.tile(v, (4, 1))	纵向复制4次

​	np.tile(v, (1,4))	横向复制4次

​	np.tile(v, (4, 3))	纵向4次，横向3次

```python
import numpy as np
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))   # Stack 4 copies of v on top of each other
print(vv)                 # Prints "[[1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]]"
y = x + vv  
print(y)  
```

​	法三：广播功能直接实现

```python
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  
print(y)
```

#### 三、SciPy

##### 1.Image operations

​	SciPy提供了一些处理图像的基本功能。例如，它具有将磁盘中的图像读取到numpy数组，将numpy数组作为图像写入磁盘以及调整图像大小的功能。

```python
from scipy.misc import imread, imsave, imresize
# Read an JPEG image into a numpy array
img = imread('assets/cat.jpg')
print(img.dtype, img.shape)  # Prints "uint8 (400, 248, 3)"
# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
img_tinted = img * [1, 0.95, 0.9]
# Resize the tinted image to be 300 by 300 pixels.
img_tinted = imresize(img_tinted, (300, 300))
# Write the tinted image back to disk
imsave('assets/cat_tinted.jpg', img_tinted)
```

##### 2.Distance between points

SciPy定义了一些函数来计算点集之间的距离。

+ scipy.spatial.distance.pdist 计算点集中 所有点之间的欧氏距离

  ```python
  import numpy as np
  from scipy.spatial.distance import pdist, squareform
  
  # Create the following array where each row is a point in 2D space:
  # [[0 1]
  #  [1 0]
  #  [2 0]]
  x = np.array([[0, 1], [1, 0], [2, 0]])
  print(x)
  
  # Compute the Euclidean distance between all rows of x.
  # d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
  # and d is the following array:
  # [[ 0.          1.41421356  2.23606798]
  #  [ 1.41421356  0.          1.        ]
  #  [ 2.23606798  1.          0.        ]]
  d = squareform(pdist(x, 'euclidean'))
  print(d)
  ```

#### 四、Matplotlib

​	matplotlib.pyplot模块，该模块提供了类似于MATLAB的绘图系统。

matplotlib中最重要的功能是`plot`，它允许您绘制2D数据。

##### 1.Plot

```python
import numpy as np
import matplotlib.pyplot as plt
# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()
```

![1571386723949](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1571386723949.png)

##### 2.Subplots

使用`subplot`函数在同一图中绘制不同的事物。

```python
import numpy as np
import matplotlib.pyplot as plt
# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)    #子图网格由（2,1）两行一列共两个子图构成
# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')
# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')
# Show the figure.
plt.show()
```

##### 3.Images

```python
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
img = imread('assets/cat.jpg')
img_tinted = img * [1, 0.95, 0.9]
# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(img)
# Show the tinted image
plt.subplot(1, 2, 2)
# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))   #限定格式 
plt.show()
```

#### 五、补充

##### 1.除运算

+ /  结果为浮点数
+ //  结果为整数

##### 2.np.vstack()   np.hstack()

+ np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组

  ![1584946791169](C:\Users\18742\AppData\Roaming\Typora\typora-user-images\1584946791169.png)

  ![1584946814118](C:\Users\18742\AppData\Roaming\Typora\typora-user-images\1584946814118.png)

+ np.hstack:按水平方向（列顺序）堆叠数组构成一个新的数组

![1584946738918](C:\Users\18742\AppData\Roaming\Typora\typora-user-images\1584946738918.png)

![1584946769909](C:\Users\18742\AppData\Roaming\Typora\typora-user-images\1584946769909.png)

# <二> python的‘’main函数‘’

```python
#hello.py
def foo():
    str="function"
    print(str);
if __name__=="__main__":
    print("main")
    foo()
```

+ 其中`if __name__=="__main__":`这个程序块类似与Java和C语言的中main（主）函数
  在Cmd中运行结果
  C:\work\python\divepy>python hello.py
  main
  function

+ 在Python Shell中运行

  ```
  >>>import hello
  >>>hello.foo()
  function
  >>>hello.__name__
  'hello'
  可以发现这个内置属性__name__自动的发生了变化。这是由于当你以单个文件运行时，__name__便是__main__,当你以模块导入使用时，这个属性便是这个模块的名字。
  ```

​	在C/C++/Java中，main是程序执行的起点，Python中，也有类似的运行机制，但方式却截然不同：Python使用缩进对齐组织代码的执行，所有没有缩进的代码（非函数定义和类定义），都会在载入时自动执行，这些代码，可以认为是Python的main函数。

 	每个文件（模块）都可以任意写一些没有缩进的代码，并且在载入时自动执行。

​	为了区分主执行文件还是被调用的文件，Python引入了一个变量__name__。

 + 当文件是被调用时，__name__的值为模块名，被调用模块中`if __name__=="__main__":`部分不被执行

 + 当文件被执行时，__name__为'__main__'。`if __name__=="__main__":`部分参与执行

   这个特性，为测试驱动开发提供了极好的支持，我们可以在每个模块中写上测试代码，这些测试代码仅当模块被Python直接执行时才会运行，代码和测试完美的结合在一起。

# <三>lambda

## 一、lambda函数也叫匿名函数，即函数没有具体的名称。

先来看一个最简单例子：

def  f(x):
    return x**2
print f(4)

Python中使用lambda的话，写成这样

g = lambda x : x**2

print g(4)

## 二、一般作用

1. 使用Python写一些执行脚本时，使用lambda可以省去定义函数的过程，让代码更加精简。

2. 对于一些抽象的，不会别的地方再复用的函数，有时候给函数起个名字也是个难题，使用lambda不需要考虑命名的问题。

3. 使用lambda在某些时候让代码更容易理解。

## 三、lambda基础

**lambda语句中，冒号前是参数，可以有多个，用逗号隔开，冒号右边的返回值。**

lambda语句构建的其实是一个函数对象：

\>>> foo = [2, 18, 9, 22, 17, 24, 8, 12, 27]
\>>> print **filter**(lambda x: **x % 3 == 0**, foo)   # **x % 3==0 是返回值**，foo是输入参数

[18, 9, 24, 12, 27]

\>>> print **map**(lambda x: **x \* 2 + 10,** foo)    #  x*2+10 是返回值，foo是输入参数

[14, 46, 28, 54, 44, 58, 26, 34, 64]  

\>>> print **reduce**(lambda x, y: x + y, foo)   #  **x,y是输入参数，x+y是返回值，foo是输入参数**

139

在对象遍历处理方面，其实Python的for..in..if语法已经很强大，并且在易读上胜过了lambda。

# <四>`__init__.py`

`__init__.py` 文件的作用是将文件夹变为一个Python模块,Python 中的每个模块的包中，都有__init__.py 文件。

通常__init__.py 文件为空，但是我们还可以为它增加其他的功能。我们在导入一个包时，实际上是导入了它的__init__.py文件。这样我们可以在__init__.py文件中批量导入我们所需要的模块，而不再需要一个一个的导入。

```python
#package
#__init__.py

import re
import urllib
import sys
import os

#a.py

import package 
print(package.re, package.urllib, package.sys, package.os)
```

注意这里访问__init__.py文件中的引用文件，需要加上包名。

__init__.py中还有一个重要的变量，__all__, 它用来将模块全部导入。

```python
#__init__.py

__all__ = ['os', 'sys', 're', 'urllib']

#a.py

from package import *
```

这时就会把注册在__init__.py文件中__all__列表中的模块和包导入到当前文件中来。

# <五>命令行工具

## 1.fire

https://blog.csdn.net/u010099080/article/details/70332074

```python
import fire

xxxxxxxx

if __name__ == '__main__':
    fire.Fire(main)     #括号内为参数所在的function
    
    
命令行格式示例：
export DATA_FILE=/path/to/corpus       设置环境变量
export BERT_PRETRAIN=/path/to/pretrain
export SAVE_DIR=/path/to/save

python pretrain.py \
    --train_cfg config/pretrain.json \
    --model_cfg config/bert_base.json \
    --data_file $DATA_FILE \           $表示环境变量
    --vocab $BERT_PRETRAIN/vocab.txt \
    --save_dir $SAVE_DIR \
    --max_len 512 \
    --max_pred 20 \
    --mask_prob 0.15
```

## 2.argparse

https://www.jianshu.com/p/a41fbd4919f8

#### a.python的一个命令行解析包，用于编写可读性非常好的程序

#### b.用法

```python
import argparse  #step1

def main(config):
    '''
    main只是函数名称，无特殊意义。
    '''
    print(config.lr)
    print(config.style__weight)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    #step2
    parser.add_argument('--content', type=str, default='png/content.png')    #step3
    parser.add_argument('--style_weight', type=float, default=100)
    parser.add_argument('--lr', type=float, default=0.003,help='学习率') #help对参数解释说明
    config = parser.parse_args()   #step4
    #jupyter notebook中以下两种方式：
    #config = parser.parse_known_args()[0]   
    #config = parser.parse_args(args=[])     []中加参数
    print(config)
    #显示Namespace(content='png/content.png', style_weight=100,lr=0.003)
    main(config)
```

- python是脚本语言，所以程序会从第一行按层次开始执行

  `if __name__=="__main__":`为程序建立虚拟入口  ，作用等同于C语言的main()函数

  

* 清华镜像使用：

  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow

  

# <六>gym模块（强化学习）

## 一、入门

+ import gym

+ **env = gym.make('CartPole-v0')**

  导入Gym库之后，可以通过 make()函数来得到环境对象，每个环境都有一个ID，格式"Xxxx-vd"，d表示版本号

+ 查看Gym库已经注册了哪些环境

​        from gym import env

​       senv_specs = envs.registry.all()

​       env_ids = [env_spec.id for env_spec in env_specs]

​       env_ids

+ 每个环境都定义了自己的观测空间和动作空间。

​       环境env的观测空间用env.observation_space表示，动作空间用env.action_space 表示。

​       观测空间分为离散空间和，表示为gym.spaces.Discrete，和连续空间，表示为gym.spaces.Box。

​       例如，环境‘MountainCar-v0'的观测空间是Box（2），表示观测可以用两个float值表示，而动作空间是      Discrete(3)，表示动作取值{1，2，3}。

+ 接下来使用环境对象env，首先初始化环境对象env，  reset()为重新初始化函数

​         env.reset()

​       该调用能返回智能体的初始观测，是np.array对象，环境初始化之后就可以使用了。

+ 使用环境的核心是使用环境对象的**step()**方法，该函数在仿真器中扮演物理引擎的角色。其输入是动作a，输出是：下一步状态，立即回报，是否终止，调试项。

  接收智能体的动作作为参数，并返回以下4个参数。

​                1.observation：np.array对象，表示观测，和env.reset()的返回值意义相同。

​                2.reward : float类型的值。

​                3.done:  bool类型的数值，本回合结束提示。Gym库里的实验环境大多是回合制的，这个返回值可以指示当前动作后游戏是否结束，如果结束，可以通过env.reset()开始下一回合。

​                4. info : dict类型的值，其他信息，包含以下调试信息。对于调试信息，可以为空，但不能缺少，否则会报错，常用{}来代替。

​	每次调用env.step()只会让环境前进一步。所以env.step()往往放在循环结构里面，通过循环调用来完成整个回合。

+ **env.reset()**的参数取自动作空间。可以使用以下语句从动作空间中随机选取一个动作：

​           action = env.action_space.sample()

+ 在env.reset()和env.step()后，可以用以下语句以图形化的方法显示当前环境。

​           **env.render()**

+ 使用完环境后，可以用下列语句关闭环境。注意：强行关闭图形界面可能导致死机，最好用以下命令关闭。

​           env.close()

# <七> 随时记

### 1.random.seed（）

![1592383987423](C:\Users\18742\AppData\Roaming\Typora\typora-user-images\1592383987423.png)

理解：

+ random.seed(x)创造了一个size无限大的数组，x不同则数组不同

+ 紧接着调用一次random()时，取数组中的第一个值，所以每次seed(x)之后最近取得一次随机数是相同的
+ seed()后多次调用random()时，依次取得数组中的前几个元素，给人的直观表现就是随机数值是不同的，但是这一组随机数是确定的，即seed(x)产生的固定数组中的值

### 2.sorted()函数

+ sort()只是做排序操作，无返回值；sorted()排序后形成新的list

+ 

  ```
  sorted(iterable, cmp=None, key=None, reverse=False)[0:k]
  ```

  参数说明：

  - iterable -- 可迭代对象。
  - cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
  - key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
  - reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
  - [0:k]取排序后的前k个元素

​     

### 3.log()

+ math.log() 底数为e 即ln   math.log10()   math.log2()

+ math.log1p(x) := math.log(x+1)  前者表示方法使数值更有效（如当x=0.00000000000000000001时）
+ math.expm1() := math.exp(x)-1

### 4. max(iterator,key)  字典中的用法

+ 1、key后面是函数,通过这个值去查找对应的键
  2、不加key这个函数的话，默认遍历的是字典的key，最后输出最大的key
+ max(a，key = a.get)将返回一个a.get(item)的值最大的项。即最大值对应的键。

### 5.数据结构

1.List列表         [1,2,3,4,4]           有序     可变

2.Tuple元组   （1,2,3,4,4）         有序    不可改变，只读

3.Dict字典        {‘a':1,'b':2,....}      无序     可更改value

4.Set集合          ([1,2,3,4])            无序     元素不可修改，可添加删除，元素不可重复

5.Map               Map(function,iterable)    返回iterable中的每个元素经过function处理后的值，原iterable不改变

### 6.*与**操作符的[两种用法]

+ 一种是用作运算符，即*表示乘号，**表示次方。
+ 第二种是用于指定函数传入参数的类型，*用于参数前面，表示传入的（多个）参数将按照元组的形式存储；**用于参数前则表示传入的（多个）参数将按照字典的形式存储，且传入的参数需是以key_0=key_value_0, key_1=key_value_1...赋值形式传入，key_0对应字典键，key_value_0对应键key_0的值。

+ Python中的**还可用于求两个字典的并集。

```html
a1 = {'x':1, 'y':2,'z':3, 'w':12}
b1 = {'x':11, 'y':12,'z':13, 'p':34}
dict(a1, **b1)      #在b1的基础上进行求并集，公共部分取b1
dict(b1, **a1)      #在a1的基础上进行求并集，公共部分取a1
#结果显示
#{'p': 34, 'w': 12, 'x': 11, 'y': 12, 'z': 13}
#{'p': 34, 'w': 12, 'x': 1, 'y': 2, 'z': 3}
```

### 7.pandas中的DataFrame

pd.DataFrame( data, index, columns, dtype, copy)


data表示要传入的数据 ，包括 ndarray，series，map，lists，dict，constant和另一个DataFrame

index和columns 行索引和列索引  格式['x1','x2']

dtype:每列的类型

copy: 从input输入中拷贝数据。默认是false，不拷贝
