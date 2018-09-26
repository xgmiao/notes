1. **torch.nn.functional.softmax**(input, dim=None)

   该函数采用softmax的定义，对input张量进行softmax。dim参数控制按照哪个维度进行softmax.

   ```python
   from torch.nn.functional import softmax
   a = np.array([[1, 2], [1, 4]], dtype=np.float32)
   x = torch.from_numpy(a)
   print(x)
   '''
   tensor([[ 1.,  2.],
           [ 1.,  4.]])
   '''
   y =  softmax(x,dim=0)
   print(y)
   '''
   tensor([[ 0.5000,  0.1192],
           [ 0.5000,  0.8808]])
   '''
   z = softmax(x,dim=1)
   print(z)
   '''
   tensor([[ 0.2689,  0.7311],
           [ 0.0474,  0.9526]])
   '''
   ```

2. **torch.nn.functional.log_softmax**(input, dim=None)

   该函数相当于先对input进行softmax，然后再对**每个元素**进行log取对数。

   ```python
   import torch
   from torch.nn.functional import softmax
   a = np.array([[1, 2], [1, 4]], dtype=np.float32)
   x = torch.from_numpy(a)
   print(x)
   '''
   tensor([[ 1.,  2.],
           [ 1.,  4.]])
   '''
   y = softmax(x,dim=1)
   print(y)
   '''
   tensor([[ 0.2689,  0.7311],
           [ 0.0474,  0.9526]])
   '''
   log_y = torch.log(y)
   print(log_y)
   '''
   tensor([[-1.3133, -0.3133],
           [-3.0486, -0.0486]])
   '''
   z = log_softmax(x,dim=1)
   print(z)
   '''
   tensor([[-1.3133, -0.3133],
           [-3.0486, -0.0486]])
   '''
   ```

3. 121

   ```python
   import torch
   x = torch.rand(3)
   x.requires_grad_()
   print(x)
   y = x*x
   print(y)
   
   # 系数
   grad = torch.Tensor([0.1,1,2])
   y.backward(grad)
   print(x.grad)
   
   z = torch.sum(y)
   print(z)
   z.backward()
   print(x.grad)
   ```

4. Add,Concatenate层

   ```
   x = Concatenate()([y,z])     # 正确用法
   x = Concatenate([y,z])()     # 错误用法
   x = Concatenate([y,z])       # 错误用法
   x = Add()([y,z])     # 正确用法
   x = Add([y,z])()     # 错误用法
   x = Add([y,z])       # 错误用法
   ```

   

   ​     https://www.cnblogs.com/demian/p/8011733.html



































参考：

http://pytorch-cn.readthedocs.io/zh/latest/