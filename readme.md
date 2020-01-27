## TF1
1. 使用自带的timeline工具
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/client/timeline.py
简单的例子：
如何读timeline的结果
https://www.jianshu.com/p/937a0ce99f56
如何用timeline
https://www.cnblogs.com/yinghuali/p/7589977.html
https://zhuanlan.zhihu.com/p/74551298
https://towardsdatascience.com/howto-profile-tensorflow-1a49fb18073d

## TF2.0:
这个工具可行
https://stackoverflow.com/questions/56690089/how-to-graph-tf-keras-model-in-tensorflow-2-0
```
import timeit

import numpy as np
import tensorflow as tf
from tensorflow import keras

lstm = keras.layers.LSTM(1024, return_sequences=True)

@tf.function
def traceme(x):
    return lstm(x)

if __name__ == "__main__":
    #tf.config.threading.set_intra_op_parallelism_threads(1)
    #tf.config.threading.set_inter_op_parallelism_threads(112)
    logdir = "log"
    writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on(graph=True, profiler=True)

    lstm = keras.layers.LSTM(1024, return_sequences=True)
    inputs = tf.random.normal((1, 100, 1024))
    time_used = np.mean(timeit.repeat(lambda: lstm(inputs), repeat=1, number=1))
    print("Time used: {}s".format(time_used))

    with writer.as_default():
        tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=logdir)
```
然后运行tensorboard
tensorboard --logdir log/
tensorboard --host=10.10.101.2 --port=6099 --logdir="my_graph"

右上角切换到profile tab
网页上就可以看到profile的数据了

https://tensorflow.google.cn/tensorboard/tensorboard_profiling_keras
不用timeline，可以用自带的其它profile工具
https://stackoverflow.com/questions/58344162/how-to-profile-networks-in-tensorflow-v2
