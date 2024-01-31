| torch.nn.Conv1d      | torch.nn.Conv2d      | torch.nn.Conv3d      | Conv                               |
| -------------------- | -------------------- | -------------------- | ---------------------------------- |
| kernel_size          | kernel_size          | kernel_size          | kernel_shape                       |
| stride               | stride               | stride               | strides                            |
| dilation             | dilation             | dilation             | dilations                          |
| groups               | groups               | groups               | group                              |
| padding/padding_mode | padding/padding_mode | padding/padding_mode | 比较复杂，见备注                   |
| bias                 | bias                 | bias                 | 判断onnx中的偏置向量bias是否为None |
| in_channels          | in_channels          | in_channels          | weights.shape[1] * groups          |
| out_channels         | out_channels         | out_channels         | weights.shape[0]                   |

转换成几d的Conv，由：len(weights.shape) - 2 计算，意思是权重矩阵的维度减2

注1： 对于onnx中一些optional属性：

1. 如果kernel_shape没获取到，kernel_size就取weights.shape[2:]
2. 如果strides没获取到，stride就取1
3. 如果dilations没获取到，dilation就取1
4. 如果group没获取到，group就取1
5. 如果pads没获取到，padding就取 [0] * dim * 2, 也就是对应维度的0，例如Conv2d就取[0, 0, 0, 0]

注2： torch这里的padding_mode参数总是取默认值，并没有实现，这里默认是“zeros”。

对于onnx中的4种auto_pad，

1. 如果是NOTSET模式且pads是非对称的（即维度为1,3,5），直接加一层F.pad在这个Conv之前来处理这样的padding。然后padding参数取0。
2. 如果是NOTSET模式且pads是对称的，padding参数取pads[:half_len], 也就是后半。
3. 如果是NOTSET模式且pads未设置，padding参数取0。
4. 如果是VALID模式，padding参数取0。
5. 如果是'SAME_UPPER', 'SAME_LOWER'模式，直接raise NotImplementedError(f'"{auto_pad}" auto_pad is not implemented')

