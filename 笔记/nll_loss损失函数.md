#### nll_loss损失函数

负对数似然（Negative Log Likelihood）损失函数。计算loss的input需要先进行log_softmax处理。对于一个C类的分类问题，对于某一个样本，经过log_softmax后为$x=(x_{1},...,x_{c})$，$x_{i}$表示该样本属于每个类别i的概率的log值。则：

​                                      $$loss(x,t)=-x_{t}$$

其中，t为该样本对应的真实值，0<=t<C。


例子：

```python
import torch
import torch.nn.functional as F

input = torch.randn(3, 5)   #（N，C）
target = torch.tensor([1, 0, 4])  # （N,）
log_p = F.log_softmax(input,dim=1)
print(log_p)
loss = F.nll_loss(log_p, target,reduce=False)
print(loss)
print(F.cross_entropy(input,target,reduce=False))
'''
输出：
tensor([[-2.8496, -4.2248, -1.5553, -0.5869, -1.8304],
        [-2.2646, -1.6039, -1.3193, -2.0360, -1.2135],
        [-2.2817, -1.5133, -3.2705, -1.0174, -1.2795]])
tensor([ 4.2248,  2.2646,  1.2795])
tensor([ 4.2248,  2.2646,  1.2795])
'''
'''
由上输出，可知：cross_entropy函数，内部实现就是先对input计算log_softmax，然后再调用nll_loss.
'''
```

