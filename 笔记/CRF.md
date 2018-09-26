CRF(Conditional Random Field)

Dense-CRF，全连接条件随机场。

能量函数为 ![1527082524449](C:\Users\Chuck\AppData\Local\Temp\1527082524449.png)

E(x)=一元函数 + 二元函数

一元函数：来自前端FCN的输出。

二元函数：是描述像素点与像素点之间的关系，鼓励相似像素分配相同的标签，而相差较大的像素分配不同标签，而这个“距离”的定义与颜色值和实际相对距离有关。所以这样CRF能够使图片尽量在边界处分割。  **全连接条件随机场的不同就在于，二元势函数描述的是每一个像素与其他所有像素的关系，所以叫“全连接”。** 

























https://github.com/torrvision/crfasrnn



https://blog.csdn.net/u012759136/article/details/52434826



https://blog.csdn.net/sinat_26917383/article/details/54882279