# 问题描述

环境
mindspore 2.1.0

自定义了三个损失函数<br>
1.CenterLoss.py<br>
2.CenterTripletLoss<br>
3.triplet_margin_loss<br>

与一个测试函数test_loss.py

三个损失函数py文件均可运行；在test_loss.py中单独使用CenterLoss和TripletMarginLoss均能运行，但是CenterTripletLoss运行时报错

(MindSpore) [ma-user encoder]$/home/ma-user/anaconda3/envs/MindSpore/bin/python /home/ma-user/work/encoder/model/loss/test_loss.py
[WARNING] GE_ADPT(1225602,fffe61ffb1e0,python):2024-09-03-16:26:07.635.299 [mindspore/ccsrc/transform/graph_ir/utils.cc:70] FindAdapter] Can't find OpAdapter for FillV2
[WARNING] GE_ADPT(1225602,fffe61ffb1e0,python):2024-09-03-16:26:07.638.913 [mindspore/ccsrc/transform/graph_ir/utils.cc:70] FindAdapter] Can't find OpAdapter for Eye
[WARNING] KERNEL(1225602,fffe61ffb1e0,python):2024-09-03-16:26:07.638.987 [mindspore/ccsrc/kernel/framework_utils.cc:523] ParseMetadata] The size of inputs in OpIOInfo should be great than real input. Inputs size in OpIOInfo:0, real input num: 3, node: Default/Eye-op387
[WARNING] KERNEL(1225602,fffe61ffb1e0,python):2024-09-03-16:26:07.639.011 [mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_kernel_metadata.cc:53] AicpuMetadataInfo] Aicpu parsed metadata op [Eye] failed.
[WARNING] GE_ADPT(1225602,fffe61ffb1e0,python):2024-09-03-16:26:07.644.628 [mindspore/ccsrc/transform/graph_ir/utils.cc:70] FindAdapter] Can't find OpAdapter for Eye
[WARNING] KERNEL(1225602,fffe61ffb1e0,python):2024-09-03-16:26:07.644.685 [mindspore/ccsrc/kernel/framework_utils.cc:523] ParseMetadata] The size of inputs in OpIOInfo should be great than real input. Inputs size in OpIOInfo:0, real input num: 3, node: Default/Eye-op388
[WARNING] KERNEL(1225602,fffe61ffb1e0,python):2024-09-03-16:26:07.644.706 [mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_kernel_metadata.cc:53] AicpuMetadataInfo] Aicpu parsed metadata op [Eye] failed.
Traceback (most recent call last):
  File "/home/ma-user/work/encoder/model/loss/test_loss.py", line 68, in <module>
    model.train(10, train_dataset, callbacks=[LossMonitor(10)])
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/train/model.py", line 1066, in train
    initial_epoch=initial_epoch)
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/train/model.py", line 113, in wrapper
    func(self, *args, **kwargs)
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/train/model.py", line 613, in _train
    self._train_process(epoch, train_dataset, list_callback, cb_params, initial_epoch, valid_infos)
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/train/model.py", line 914, in _train_process
    outputs = self._train_network(*next_element)
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/nn/cell.py", line 664, in __call__
    raise err
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/nn/cell.py", line 660, in __call__
    output = self._run_construct(args, kwargs)
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/nn/cell.py", line 444, in _run_construct
    output = self.construct(*cast_inputs, **kwargs)
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/nn/wrap/cell_wrapper.py", line 422, in construct
    return self._no_sens_impl(*inputs)
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/nn/wrap/cell_wrapper.py", line 437, in _no_sens_impl
    loss = self.network(*inputs)
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/nn/cell.py", line 664, in __call__
    raise err
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/nn/cell.py", line 660, in __call__
    output = self._run_construct(args, kwargs)
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/nn/cell.py", line 444, in _run_construct
    output = self.construct(*cast_inputs, **kwargs)
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/nn/wrap/cell_wrapper.py", line 123, in construct
    return self._loss_fn(out, label)
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/nn/cell.py", line 664, in __call__
    raise err
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/nn/cell.py", line 660, in __call__
    output = self._run_construct(args, kwargs)
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/nn/cell.py", line 444, in _run_construct
    output = self.construct(*cast_inputs, **kwargs)
  File "/home/ma-user/work/encoder/model/loss/CenterTripletLoss.py", line 22, in construct
    return self.triplet_loss_fn(embeddings,labels) + self.alpha * self.center_loss_fn(embeddings,labels)
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/nn/cell.py", line 664, in __call__
    raise err
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/nn/cell.py", line 660, in __call__
    output = self._run_construct(args, kwargs)
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/nn/cell.py", line 444, in _run_construct
    output = self.construct(*cast_inputs, **kwargs)
  File "/home/ma-user/work/encoder/model/loss/CenterLoss.py", line 16, in construct
    for label in y:
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/common/tensor.py", line 456, in __getitem__
    out = tensor_operator_registry.get('__getitem__')(self, index)
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/ops/composite/multitype_ops/_compile_utils.py", line 182, in _tensor_getitem
    self, index)
  File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/ops/operations/_inner_ops.py", line 2556, in __call__
    return Tensor_.getitem_index_info(data, index, self.is_ascend)
RuntimeError: Query kernel type failed, node name: Default/Eye-op388, node info: @kernel_graph_201:[CNode]7{[0]: ValueNode<Primitive> Eye, [1]: ValueNode<Tensor> Tensor(shape=[], dtype=Int64, value=16), [2]: ValueNode<Tensor> Tensor(shape=[], dtype=Int64, value=16), [3]: ValueNode<Tensor> Tensor(shape=[], dtype=Int64, value=43)}

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/plugin/device/ascend/hal/hardware/ge_kernel_executor.cc:272 OptimizeGraph