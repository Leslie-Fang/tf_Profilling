import tensorflow as tf
from tensorflow.python.client import timeline
run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
run_metadata = tf.compat.v1.RunMetadata()
logistic = sess1.run(output_tensor, dict(zip(input_tensor, features_list[i][0:2])), options=run_options, run_metadata=run_metadata)
tl = timeline.Timeline(run_metadata.step_stats)
ctf = tl.generate_chrome_trace_format()
with open('/home/lesliefang/debug_matmul/timeline.json', 'w') as f:
  f.write(ctf)
