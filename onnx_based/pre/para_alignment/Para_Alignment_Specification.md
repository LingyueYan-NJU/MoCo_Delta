ONNX_OP_NAME: {

​	INPUTS: [?input1name?, ?input2name?, ......]

​	ATTRIBUTES_REQUIRED: [{name: ''?ar1?'', type: INT, value: {INT: 1}}, {name: ''?ar2?'', type: INTS, value: {INT: 1, INT: 1}}, ......]

​	ATTRIBUTES_OPTIONAL: [{name: ''?ao1?'', type: INT, value: {INT: 1}}, {name: ''?ao2?'', type: INTS, value: {INT: 1, INT: 1}}, ......]

​	TO_TORCH: {

​		torch.nn.ar1:{

​			para1: {mode: "DIRECTLY", expression: "?ar1?", default: "NONE"}

​			para2: {mode: "DIRECTLY", expression: "?ao1?", default: 1}

​			para3: {mode: "EXPRESSION", expression: "?ao1? * ?ar1? + ?ao2?", default: 1}

​			......

​		}

​		torch.nn.ar2: { ...... }

​		......

​	}

​	TO_TENSORFLOW: { ...... }

}