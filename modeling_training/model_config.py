
import tensorflow as tf
config = {
    
        'model_dimension': 512,
        'embeddings_dimension': 128,
        'attention_num_heads': 8,
        'attention_dropout': 0.0,
        'feedforward_dimension':2048,
        'num_layers': 4,
        'timediff_attn': True,
        'attn_weights_initializer': tf.keras.initializers.RandomUniform(minval = 0.0, maxval = 1.0)
}