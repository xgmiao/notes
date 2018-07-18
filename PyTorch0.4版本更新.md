1. Tensor类和Variable类合并。
   之前的版本中，通常是用Variable来表示一些可以求导的变量，而Tensor类型的变量是不可以求导的，在0.4版本中，Tensor类型变量也可以求导，既然Tensor类型包含了Variable的全部功能，估计在后续的版本中会弃用Varibale。
   PyTorch 0.4之前的代码：

   ```python
   import torch
   from torch.autograd import Variable
   tensor_x = torch.rand(1,2)
   
   #'torch.FloatTensor' object has no attribute 'requires_grad'
   # print(tensor_x.requires_grad)
   
   print(type(tensor_x))  # <class 'torch.FloatTensor'>
   print(tensor_x.type()) # torch.FloatTensor
   
   variable_x = Variable(tensor_x,requires_grad=True)
   print(type(variable_x)) # <class 'torch.autograd.variable.Variable'>
   print(variable_x.requires_grad)  # True
   
   variable_y = Variable(tensor_x)
   print(variable_x.requires_grad)  # False
   
   #'Variable' object has no attribute 'requires_grad_'
   #variable_y.requires_grad_()
   '''
   通过上面的代码，可以发现：
   1.Tensor变量是没有requires_grad标志位的.
   2.Variable变量需要通过Tensor变量转换得到，并且在初始化时，就要指定requires_grad.
   3.Variable没有requires_grad_()方法.
   '''
   ```

   PyTorch 0.4的代码：

   ```python
   import torch
   tensor_x = torch.rand(1,2)
   print(type(tensor_x)) # <class 'torch.Tensor'>
   print(tensor_x.requires_grad) # False
   
   x = torch.tensor([1, 2], requires_grad=True)
   print(x.requires_grad)
   
   y = torch.tensor([1,2]).requires_grad_()
   print(y.requires_grad)
   
   z = torch.tensor([1,2])
   z.requires_grad_()
   print(z.requires_grad)
   
   '''
   通过以上代码，可以发现:
   1.Tensor变量已经有了requires_grad属性，说明Tensor可以完全具有了Variable的功能.
   2.Tensor变量有requires_grad_()方法，来更改其是否计算梯度，这是Variable中没有的.
   '''
   ```

2. type()函数，不再反应张量的数据类型。
   PyTorch 0.4之前代码：

   ```python
   import torch
   tensor_x = torch.Tensor([1,2])
   # TypeError: 'module' object is not callable
   # y = torch.tensor([1,2])
   print(type(tensor_x))  # <class 'torch.FloatTensor'>
   print(tensor_x.type()) # torch.FloatTensor
   '''
   1.type()包含了x.type()的功能.
   2.张量类型是Tensor，而不是tensor.
   3.通常在定义一个张量时，直接根据张量数据类型，定义其张量，如torch.ByteTensor、torch.IntTensor、torch.FloatTensor等.
   '''
   ```

   PyTorch 0.4的代码:

   ```python
   import torch
   x = torch.tensor([1, 2])
   print(type(x)) # <class 'torch.Tensor'>
   print(x.type()) # torch.LongTensor
   y = torch.Tensor([1,2])
   print(type(y)) # <class 'torch.Tensor'>
   print(y.type()) # torch.FloatTensor
   '''
   1.type()函数不在返回张量的数据类型，而是通过x.type()返回.
   2.张量类型，既可以用tensor也可以用Tensor，但是为了兼容方便，最好建议使用Tensor.
   '''
   ```