### Tensorflow笔记

```python
tf.train.string_input_producer 生成一个Filename Queue 文件名队列
tf.train.slice_input_producer
```
Filename Queue 文件名队列

QueueRunner工作线程

文件阅读器，有个read()方法，参数为文件名队列，阅读器的`read`方法会输出一个key来表征输入的文件和其中的纪录(对于调试非常有用)，同时得到一个字符串标量， 这个字符串标量可以被一个或多个解析器，或者转换操作将其解码为张量并且构造成为样本。 
```
tf.TextLineReader()
```
在调用`run`或者`eval`去执行`read`之前， 你必须调用`tf.train.start_queue_runners`来将文件名填充到队列。否则`read`操作会被阻塞到文件名队列中有值为止。 