15:19:21 LOG> tokenzing train data
15:19:23 LOG> initializing train data
15:19:23 LOG> creating model
(50002, 128)
[[ 3.87249851  3.18059158 -1.96747088 ...,  0.78257984  2.24364209
   0.2932277 ]
 [ 1.10917771 -1.23216248 -1.67313206 ...,  0.25728047 -1.43838263
   2.71200585]
 [ 0.53421867 -0.46636033  0.70915627 ..., -1.04841268 -1.21687067
   1.59254968]
 ..., 
 [ 0.          0.          0.         ...,  0.          0.          0.        ]
 [ 0.          0.          0.         ...,  0.          0.          0.        ]
 [ 0.          0.          0.         ...,  0.          0.          0.        ]]
2017-11-07 15:19:30.256816: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2017-11-07 15:19:30.257537: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
name: GeForce GTX 950M major: 5 minor: 0 memoryClockRate(GHz): 1.124
pciBusID: 0000:01:00.0
totalMemory: 3.95GiB freeMemory: 3.91GiB
2017-11-07 15:19:30.257574: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 950M, pci bus id: 0000:01:00.0, compute capability: 5.0)
15:22:20 LOG> train launched.
15:34:10 LOG> [0/20] total loss=1675.1949620246887, average loss=5.3411287585104406
15:34:57 LOG> start accuracy: 0.399959903769, end accuracy: 0.439254210104
15:46:35 LOG> [1/20] total loss=1399.9664897918701, average loss=4.463600625052543
15:47:23 LOG> start accuracy: 0.407778668805, end accuracy: 0.468524458701
15:59:01 LOG> [2/20] total loss=1207.9838256835938, average loss=3.8514903025830716
15:59:50 LOG> start accuracy: 0.453488372093, end accuracy: 0.528468323978
16:11:39 LOG> [3/20] total loss=980.8150866031647, average loss=3.1271940189609198
16:12:28 LOG> start accuracy: 0.566760224539, end accuracy: 0.606655974338
16:24:06 LOG> [4/20] total loss=825.5322005748749, average loss=2.632095891834404
16:24:53 LOG> start accuracy: 0.608259823577, end accuracy: 0.636728147554
16:36:32 LOG> [5/20] total loss=717.5178948640823, average loss=2.2877071325313243
16:37:20 LOG> start accuracy: 0.63612670409, end accuracy: 0.661587810746
16:48:59 LOG> [6/20] total loss=633.2575488090515, average loss=2.0190546068738753
16:49:46 LOG> start accuracy: 0.647554129912, end accuracy: 0.672012830794
17:01:24 LOG> [7/20] total loss=565.5024482011795, average loss=1.8030267864731475
17:02:13 LOG> start accuracy: 0.63993584603, end accuracy: 0.670408981556
17:13:50 LOG> [8/20] total loss=510.21919095516205, average loss=1.626763723465868
17:14:39 LOG> start accuracy: 0.632117080994, end accuracy: 0.659382518043
17:26:17 LOG> [9/20] total loss=458.64832401275635, average loss=1.4623371064024513
17:27:06 LOG> start accuracy: 0.631716118685, end accuracy: 0.658380112269
^CTraceback (most recent call last):
  File "./base_model.py", line 336, in <module>
    vquerys,vpassages,vstarts,vends,voverlaps,batch_size=64,iter_num=20)
  File "./base_model.py", line 234, in train
    _,err=self.sess.run([self.opt,self.loss],feed_dict=feed_dict)
  File "/usr/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 889, in run
    run_metadata_ptr)
  File "/usr/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1120, in _run
    feed_dict_tensor, options, run_metadata)
  File "/usr/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1317, in _do_run
    options, run_metadata)
  File "/usr/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1323, in _do_call
    return fn(*args)
  File "/usr/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1302, in _run_fn
    status, run_metadata)
KeyboardInterrupt
'./base_model.py' has 已终止
recolic@RECOLICPC ~/m/Factoid-QA-subtask (master)> fg
将任务 1，“./base_model.py” 发送到前台

