#### 一、向远程仓库中提交项目

若没有远程库，先new  repository（不要勾选init）

客户端方式或网页方式，网页方式包含https和ssh两种验证方式，以下以ssh为例：如果没有密钥的话，git命令行输入 ssh-keygen 生成新的密钥，复制在github账号设置中

1.git init （本地生成 .git文件）

![1570786623127](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1570786623127.png)

2.git add 文件名（ git add .    表示目录下所有文件）

![1570786645523](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1570786645523.png)

3.git commit -m '第一次提交' （必要的说明）

![1553524691625](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1553524691625.png)



4.git remote add origin 仓库地址 （本地与远程关联）

![1553524770561](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1553524770561.png)

5.git push  origin master  (push操作，若勾选了init 则加--force)

![1570786677558](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1570786677558.png)

#### 二、将远程仓库的项目复制到本地

1.git clone 仓库地址（或者直接点击clone or downlaod）

![1553524846123](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1553524846123.png)

2.若需提交项目：可将项目文件复制到此仓库中，后续进行上述的提交等操作

#### 三、Git

1.当与其他开发者协作时，请确保创建一个新的分支，并使用描述性的名称说明它所包含的更改。

2.从远程仓库拉取修改

​	git pull origin master    (origin是远程仓库的简写名)

3.`git remote -v` 用于查看远程仓库与连接之间的详细信息。

