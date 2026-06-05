import tensorflow as tf

x=tf.constant([1.,2.,3.])
i=tf.constant([0,0,1])
@tf.function(jit_compile=True)
def f(x,i): return tf.math.unsorted_segment_sum(x,i,num_segments=2)
print(f(x,i))
