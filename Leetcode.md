## Leetcode刷题：

### 3.14

#### 1.Java数组两种定义方式：  

  a. ` int[] array={1,2};`

  b. 

```java
 int[] array=new int[2];
    array[0]=1;
    array[1]=2;
```

#### 2.整型越界

Integer.MAX_VALUE   ……..7

Integer.MIN_VALUE   -……..8

有时在初始化最小值时使用Integer.MAX_VALUE，初始化最大值则Integer.MIN_VALUE

#### 3.整数反转与回文序列思路：

while(x!=0){

​	rev=rev*10+x%10;

​	x/=10;

}

#### 4.return最后写，防止if等句中的返回值利用不到时程序就没有了返回值

#### 5.双指针使用

##### 链表找中点：快慢指针

##### 数组前后的值是否相等：前后指针

  补充：链表节点结构：   

public class ListNode{

​	 int val;  

​	 ListNode next;

​	ListNode(int x) {val=x}    

 }

#### 6.所有问题先写特殊情况

#### 7.链表反转：

   a.原地反

   b.新建

#### 8.字符串.length()    数组.length

#### 9.null与isEmpty()

null  不存在，未分配内存

isEmpty()  存在，但什么都没有 适用于多种结构，在String中相当于“”（空字符串）

#### 10.字符串常见函数：

   Str.substring(0,2);    //2不包括

   Str.isEmpty();

   Str.indexOf(hello);    //-1,0,1….

   Str.charAt(2)     //e…

#### 11.对于key\value问题：

Map: HashTable\HashMap\TreeMap

以HashMap为例：

```java
Map<String,String> map=new HashMap<String,String>();

​                //添加

​                map.put("09", "zhaoliu");    

​                map.put("01", "zhangsan");            

​                map.put("02", "wangwu03");

​                map.put("03", "wangwu04");

​                map.put("04", "qqquqq");

​                //获取map集合中的所有键的集合

​                Set<String> keySet=map.keySet();

​                //迭代所有键来获取值

​                Iterator<String> iterator=keySet.iterator();

​                while(iterator.hasNext()){

​                    String key=iterator.next();

​                    //通过map.get(键)的方式来获取值

​                    System.out.println(key+".........."+map.get(key));

​                }
```

常用函数：

put(..,..)  containsKey(..)  containsValue(…)  get(….)   remove(….)    

Sys……println(map);

#### 12.HashTable

a.来源：https://blog.csdn.net/u010297957/article/details/51974340

b.简介：根据设定的Hash函数 - H(key) 和处理冲突的方法，将一组关键字映象 到一个有限的连续的地址集（区间）上，并以关键字在地址集中的象 作为记录在表中的存储位置，这样的映射表便称为Hash表。

c.两数之和问题的hash实现：

```java
class Solution{

 public int[] twoSum(int[] nums, int target) {

​    Map<Integer, Integer> map = new HashMap<>();

​    for (int i = 0; i < nums.length; i++) {

​        map.put(nums[i], i);

​    }

​    for (int i = 0; i < nums.length; i++) {

​        int complement = target - nums[i];

​        if (map.containsKey(complement) && map.get(complement) != i) {

​            return new int[] { i, map.get(complement) };

​        }

​    }

​    throw new IllegalArgumentException("No two sum solution");

}
}
```

#### 11、12要点总结:

1.Map<Integer, Integer> map = new HashMap<>();

   map.put("09", "zhaoliu");    

   map.get("09");

   map.containskey("09");

2.Set<String> set=map.keySet();

   或Set<String> set=new HashSet<>();

   set.add("hello");

   set.remove("hello");

   set.contains("hello");

3.List<String> list=new ArrayList<>();
    list.add("cc");

​    list.remove("cc");

3.Iterator<String> iterator= list/map/set.iterater();

  while(iterator.hasNext()){

​                String key=iterator.next();

...........

​            }

#### 13.链表哑节点

ListNode dummyHead=new ListNode(0);

ListNode p=dummyHead;

..........

return dummyHead.next;

#### 14.有序数组的查找可以考虑二分法

#### 15.使用系统函数加速！！Arrays.sort(arr);    Math.max(a,b);

Math.min(a,b);

#### 16.二叉树 深搜&广搜：常用递归\迭代

相等：同空\同非空且值相等（对称也类似）

1.父节点相等

2.左孩子相等

3.右孩子相等

```java
public class TreeNode {
      int val;
      TreeNode left;
      TreeNode right;
      TreeNode(int x) { val = x; }
  }
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p==null&&q==null)
            return true;
        else if(p!=null&&q!=null&&p.val==q.val)
            return isSameTree(p.left,q.left)&&isSameTree(p.right,q.right);
        else
            return false;
    }
```

#### 17.遍历数组小技巧：
for(int bill:bills)