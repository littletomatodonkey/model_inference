## onnx 模型导出准则

> more details：[DETR模型导出](https://www.hbblog.cn/%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2/2022%E5%B9%B404%E6%9C%8811%E6%97%A5%2023%E6%97%B612%E5%88%8655%E7%A7%92/#_1)

1. 对于任何用到shape、size返回值的参数时，例如：`tensor.view(tensor.size(0), -1)，B,C,H,W = x.shape` 这类操作，避免直接使用`tensor.size`的返回值，而是加上`int`转换，`tensor.view(int(tensor.size(0)), -1)`, `B,C,H,W = map(int, x.shape)`，断开跟踪。
1. 对于`nn.Upsample`或`nn.functional.interpolate`函数，一般使用`scale_factor`指定倍率，而不是使用`size`参数指定大小。如果源码中就是插值为固定大小，则该条忽略。
1. 对于`reshape、view`操作时，-1的指定请放到`batch`维度。其他维度计算出来即可。`batch`维度禁止指定为大于-1的明确数字。如果是一维，那么直接指定为-1就好。
1. `torch.onnx.export`指定`dynamic_axes`参数，并且只指定batch维度，禁止其他动态
1. 使用`opset_version=11`，不要低于11
1. 避免使用`inplace`操作，例如`y[…, 0:2] = y[…, 0:2] * 2 - 0.5`，可以采用如下代码代替 `tmp = y[…, 0:2] * 2 - 0.5; y = torch.cat((y[..., 2:], tmp), dim = -1)`
1. 尽量少的出现5个维度，例如`ShuffleNet Module`，可以考虑合并wh避免出现5维
1. 尽量把让后处理部分在`onnx`模型中实现，降低后处理复杂度。比如在目标检测网络中最终输出设置为`xywh`或者`xyxy`，而不是一个中间结果。

