
import tensorflow as tf

import typing
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import math


class WeightedMultiHeadAttention(tfa.layers.MultiHeadAttention):
    r"""W.r.t tensorflow_addons.layers.MultiHeadAttention, we add a call argument attn_weight
        attn_weight: a float (non-negative) Tensor of shape `[batch_size, num_heads?, query_elements, key_elements]`
    """

    def __init__(
        self,
        head_size: int,
        num_heads: int,
        **kwargs
    ):
        super().__init__(head_size,
                         num_heads,
                         **kwargs)

    def build(self, input_shape):

        super().build(input_shape)

    def call(self, inputs, training=None, mask=None, attn_weight = None):

        # einsum nomenclature
        # ------------------------
        # N = query elements
        # M = key/value elements
        # H = heads
        # I = input features
        # O = output features

        query = inputs[0]
        key = inputs[1]
        value = inputs[2] if len(inputs) > 2 else key

        # verify shapes
        if key.shape[-2] != value.shape[-2]:
            raise ValueError(
                "the number of elements in 'key' must be equal to the same as the number of elements in 'value'"
            )

        if mask is not None:
            if len(mask.shape) < 2:
                raise ValueError("'mask' must have atleast 2 dimensions")
            if query.shape[-2] != mask.shape[-2]:
                raise ValueError(
                    "mask's second to last dimension must be equal to the number of elements in 'query'"
                )
            if key.shape[-2] != mask.shape[-1]:
                raise ValueError(
                    "mask's last dimension must be equal to the number of elements in 'key'"
                )

        # Linear transformations
        query = tf.einsum("...NI , HIO -> ...NHO", query, self.query_kernel)
        key = tf.einsum("...MI , HIO -> ...MHO", key, self.key_kernel)
        value = tf.einsum("...MI , HIO -> ...MHO", value, self.value_kernel)

        # Scale dot-product, doing the division to either query or key
        # instead of their product saves some computation
        depth = tf.constant(self.head_size, dtype=tf.float32)
        query /= tf.sqrt(depth)

        # Calculate dot product attention
        logits = tf.einsum("...NHO,...MHO->...HNM", query, key)

        # apply mask
        if mask is not None:
            mask = tf.cast(mask, tf.float32)

            # possibly expand on the head dimension so broadcasting works
            if len(mask.shape) != len(logits.shape):
                mask = tf.expand_dims(mask, -3)

            logits += -10e9 * (1.0 - mask)
        if attn_weight is not None:
            if len(attn_weight.shape) != len(logits.shape):
                attn_weight = tf.expand_dims(attn_weight, -3)
            logits -= attn_weight
        attn_coef = tf.nn.softmax(logits)

        # attention dropout
        attn_coef_dropout = self.dropout(attn_coef, training=training)

        # attention * value
        multihead_output = tf.einsum("...HNM,...MHI->...NHI", attn_coef_dropout, value)

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done.
        output = tf.einsum(
            "...NHI,HIO->...NO", multihead_output, self.projection_kernel
        )

        if self.projection_bias is not None:
            output += self.projection_bias

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output

def gelu(x):
    """
    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created. For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) Also see
    https://arxiv.org/abs/1606.08415
    """
    x = tf.convert_to_tensor(x)
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))

    return x * cdf

def point_wise_feed_forward_network(d_model, dff, activation = 'relu'):
  actv = gelu if activation == 'gelu' else activation
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation=actv),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class EncoderLayer(tf.keras.layers.Layer):
    r"""
    Transformer encoder layer
    """
    def __init__(self,
                 model_dimension = 64,
                 attention_num_heads = 8,
                 feedforward_dimension = 2048,
                 attention_dropout = 0.0, 
                 return_attn_coef = False,
                 activation = 'relu',
                 attn_weights_initializer = tf.keras.initializers.RandomUniform(
                     minval = 0, maxval =2.0
                 ),
                 **kwargs
                 ):
        super(EncoderLayer, self).__init__(**kwargs)
        self.model_dimension = model_dimension
        self.attention_num_heads = attention_num_heads
        self.feedforward_dimension = feedforward_dimension
        self.attention_dropout = attention_dropout
        self.return_attn_coef = return_attn_coef
        self.attn_weights_initializer = tf.keras.initializers.get(
            attn_weights_initializer
        )
        self.activation  = activation
        self.mha = WeightedMultiHeadAttention(
            model_dimension // attention_num_heads,
            attention_num_heads,
            dropout = attention_dropout,
            return_attn_coef = return_attn_coef,
            )
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = point_wise_feed_forward_network(model_dimension, 
                                                   feedforward_dimension,
                                                   activation = activation)
        self.ffn_dropout = tf.keras.layers.Dropout(attention_dropout)
        

    def build(self, input_shape):
        assert(isinstance(input_shape, dict))
        assert('value' in input_shape)
        if 'timediff' in input_shape:
            self.attn_weight_coef = self.add_weight(name="attn_weight_coef",
                                          shape=[self.attention_num_heads, 1, 1],
                            initializer=self.attn_weights_initializer,
                            regularizer=None,
                            constraint= tf.keras.constraints.NonNeg()
                        )
        super(EncoderLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.return_attn_coef:
            seq_length = input_shape['value'][-2]
            attn_shape =  input_shape['value'][:-2] +(self.attention_num_heads, 
                                                      seq_length,
                                                      seq_length)
            return input_shape['value'], attn_shape
        else:
            return input_shape['value']

    def call(self, inputs, training = None):
        value = inputs['value']
        if 'mask' in inputs:
            mask = inputs['mask']
        else:
            mask = None
        if 'timediff' in inputs:
            attn_weight = inputs['timediff']*self.attn_weight_coef
        else:
            attn_weight = None


        attn_output = self.mha([value, value], 
                               mask=mask, 
                               attn_weight = attn_weight,
                               training = training)
        if self.return_attn_coef:
            attn_output, attn_coef = attn_output
        out1 = self.layer_norm_1(value + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.ffn_dropout(ffn_output, training = training)
        out2 = self.layer_norm_2(out1 + ffn_output)
        if self.return_attn_coef:
            return out2, attn_coef  
        else:
            return out2

class DecoderLayer(tf.keras.layers.Layer):
    r"""
    Transformer decoder layer
    """
    def __init__(self,
                 model_dimension = 64,
                 attention_num_heads = 8,
                 feedforward_dimension = 2048,
                 attention_dropout = 0.0, 
                 return_attn_coef = False,
                 activation = 'relu',
                 attn_weights_initializer = tf.keras.initializers.RandomUniform(
                     minval = 0, maxval =2.0
                 ),
                 **kwargs
                 ):
        super(DecoderLayer, self).__init__(**kwargs)
        self.model_dimension = model_dimension
        self.attention_num_heads = attention_num_heads
        self.feedforward_dimension = feedforward_dimension
        self.attention_dropout = attention_dropout
        self.return_attn_coef = return_attn_coef
        self.activation = activation
        self.attn_weights_initializer = tf.keras.initializers.get(
            attn_weights_initializer
        )
        self.mha_1 = WeightedMultiHeadAttention(
            model_dimension//attention_num_heads,
            attention_num_heads, 
            dropout = attention_dropout,
            return_attn_coef = return_attn_coef)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha_2 = WeightedMultiHeadAttention(
            model_dimension//attention_num_heads,
            attention_num_heads, 
            dropout = attention_dropout,
            return_attn_coef = return_attn_coef)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn = point_wise_feed_forward_network(model_dimension, 
                                                   feedforward_dimension,
                                                   activation = activation)
        self.dropout = tf.keras.layers.Dropout(attention_dropout)
        self.layer_norm_3 =  tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    def build(self, input_shape):
        assert(isinstance(input_shape, dict))
        assert('enc_out' in input_shape)
        assert('dec_query' in input_shape)
        if 'dec_dec_timediff' in input_shape:
            self.ed_attn_weight_coef = self.add_weight(name="ed_attn_weight_coef",
                                          shape=[self.attention_num_heads, 1, 1],
                            initializer=self.attn_weights_initializer,
                            regularizer=None,
                            constraint= tf.keras.constraints.NonNeg()
                        )
            self.dd_attn_weight_coef = self.add_weight(name="dd_attn_weight_coef",
                                          shape=[self.attention_num_heads, 1, 1],
                            initializer=self.attn_weights_initializer,
                            regularizer=None,
                            constraint= tf.keras.constraints.NonNeg()
                        )
        super(DecoderLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.return_attn_coef:
            qseq_length = input_shape['dec_query'][-2]
            vseq_length = input_shape['enc_out'][-2]
            eq_attn_shape =  input_shape['dec_query'][:-2] +(self.attention_num_heads, 
                                                      qseq_length,
                                                      vseq_length)
            qq_attn_shape =  input_shape['dec_query'][:-2] +(self.attention_num_heads, 
                                                      qseq_length,
                                                      qseq_length)
            return input_shape['dec_query'], qq_attn_shape, eq_attn_shape
        else:
            return input_shape['dec_query']
    
    def call(self, inputs, training = None):
        encoder_out = inputs['enc_out']
        query = inputs['dec_query']
        if 'enc_dec_mask' in inputs:
            enc_dec_mask = inputs['enc_dec_mask']
        else:
            enc_dec_mask = None

        
        if 'dec_dec_mask' in inputs:
            dec_dec_mask = inputs['dec_dec_mask']
        else:
            dec_dec_mask = None

        if 'dec_dec_timediff' in inputs:
            enc_dec_attn_weight = inputs['enc_dec_timediff']*self.ed_attn_weight_coef
            dec_dec_attn_weight = inputs['dec_dec_timediff']*self.dd_attn_weight_coef
        else:
            enc_dec_attn_weight = None
            dec_dec_attn_weight = None
        
        attn1 = self.mha_1([query, query],
                           mask=dec_dec_mask,
                           attn_weight = dec_dec_attn_weight,
                           training = training)
        if self.return_attn_coef:
            attn1, qq_attn_coef = attn1
        out1 = self.layer_norm_1(attn1 + query)
        
        attn2 = self.mha_2([out1, encoder_out],
                           mask=enc_dec_mask,
                           attn_weight = enc_dec_attn_weight,
                           training = training)
        if self.return_attn_coef:
            attn2, eq_attn_coef = attn2
        
        out2 =  self.layer_norm_2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output =  self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout(ffn_output, training = training)
        out3 = self.layer_norm_3(ffn_output + out2)
        if self.return_attn_coef:
            return out3, qq_attn_coef, eq_attn_coef
        else:
            return out3
class ContinuousEmbedding(tf.keras.layers.Layer):
    def __init__(self, output_dims,
                 num_points = 64,
                 minval = -1.0, maxval = 1.0, 
                 embeddings_initializer = 'uniform',
                 window_size = 8,
                 window_type = 'hann',
                 normalized = True,
                 **kwargs):
        super(ContinuousEmbedding, self).__init__(**kwargs)
        self.output_dims = output_dims
        self.minval = minval
        self.maxval = maxval
        self.num_points = num_points
        self.embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)
        self.window_size = window_size
        assert window_type in {'triangular', 'rectangular', 'hann'}
        self.window_type = window_type
        self.normalized = normalized
    def _rect_window(self, x, window_size  =16):
        w_2 = window_size/2
        return (tf.sign(x+w_2) - tf.sign(x-w_2))/2
    def _triangle_window(self, x, window_size = 16):
        w_2 = window_size/2
        return (tf.abs(x+w_2)+tf.abs(x-w_2) - 2*tf.abs(x))/window_size
    def _hann_window(self, x, window_size = 16):
        y = tf.cos(math.pi*x/window_size)
        y = y*y*self._rect_window(x, window_size=window_size)
        return y
    def build(self, input_shape):
        self.points = tf.range(self.num_points, 
                               dtype = tf.float32)
        if self.window_type =='hann':
            self.window_func = self._hann_window
        elif self.window_type == 'triangular':
            self.window_func = self._triangle_window
        else:
            self.window_func = self._rect_window
          
        self.embeddings = self.add_weight(
              name = 'embeddings',
              shape = [self.num_points, self.output_dims],
              initializer = self.embeddings_initializer,
              regularizer = None,
              constraint = None
        )

    def compute_output_shape(self, input_shape):
        return input_shape +(self.output_dims,)

    def call(self, x):
        x -= tf.constant(self.minval, tf.float32)
        x *= tf.constant(
            self.num_points/(self.maxval - self.minval),
            tf.float32)
        w = tf.expand_dims(x, -1) - self.points
        w = self.window_func(w, window_size = self.window_size)
        if self.normalized:
            w = tf.math.divide_no_nan(w, 
                                      tf.reduce_sum(w, 
                                                    axis = -1, 
                                                    keepdims=True))
        output = tf.matmul(w, self.embeddings)
        return output
class ContentEmbeddingLayer(tf.keras.layers.Layer):
    r"""
    Embedding Layer for content (question or lecture)
    Given the id of the content (encoded_content_id in the input data), this 
    layer will compute the corresponding embeddings, which is the concatenation and linear-transformation
    of different elements : encoded_question_id, bundle_id, question_part, question_difficulty,
    question_popularity, question_tags, encoded_lecture_id, lecture_part, lecture_tag, lecture_type_of
    """
    def __init__(self, embeddings_dimension,
                 encoded_question_id = None, encoded_lecture_id = None, 
                 question_bundle_id = None, question_part = None,
                 question_tags = None,
                 question_difficulty = None, question_popularity = None,
                 lecture_part = None, lecture_tag = None, 
                 lecture_type_of = None, 
                 embeddings_initializer='uniform',
                 **kwargs):
        super(ContentEmbeddingLayer, self).__init__(**kwargs)
        self.embeddings_dimension = embeddings_dimension
        self.encoded_question_id = encoded_question_id
        self.encoded_lecture_id = encoded_lecture_id
        self.question_bundle_id = question_bundle_id
        self.question_popularity = question_popularity
        self.question_difficulty = question_difficulty
        self.question_part = question_part
        self.question_tags = question_tags
        self.lecture_part = lecture_part
        self.lecture_tag = lecture_tag
        self.lecture_type_of = lecture_type_of
        self.embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)
        #Questions
        question_emb_dim = 0
        if encoded_question_id is not None:
            self.qid_emb_layer = tf.keras.layers.Embedding(
                1 + max(encoded_question_id),
                embeddings_dimension,
                embeddings_initializer = self.embeddings_initializer,
                name = 'question_id_embedding',
            )
            question_emb_dim += embeddings_dimension
        if question_bundle_id is not None:
            self.qbid_emb_layer = tf.keras.layers.Embedding(
                1 + max(question_bundle_id),
                embeddings_dimension,
                embeddings_initializer = self.embeddings_initializer,
                name = 'question_bundle_embedding'
                )
            question_emb_dim += embeddings_dimension

        if question_part is not None:
            self.qp_emb_layer = tf.keras.layers.Embedding(
                1 + max(question_part),
                embeddings_dimension,
                embeddings_initializer = self.embeddings_initializer,
                name = 'question_part_embedding'
                )
            question_emb_dim += embeddings_dimension

        if question_tags is not None:
            question_tags = np.vstack([np.array(tags) for tags in question_tags])
            self.qt_emb_layer = tf.keras.layers.Embedding(
                1+question_tags.max(),
                embeddings_dimension,
                embeddings_initializer = self.embeddings_initializer,
                name = 'question_tags_embedding'
                )
            question_emb_dim += embeddings_dimension

        if question_popularity is not None:
            self.qpopularity_emb_layer = ContinuousEmbedding(
                embeddings_dimension,
                num_points = 1024,
                window_size = 8,
                minval = 0, maxval = 1.0,
                embeddings_initializer = self.embeddings_initializer,
                name = 'question_popularity_embedding'
            )
            question_emb_dim += embeddings_dimension

         
        if question_difficulty is not None:
            self.qdifficulty_emb_layer = ContinuousEmbedding(
                embeddings_dimension,
                num_points = 1024,
                window_size = 8,
                minval = 0, maxval = 1.0,
                embeddings_initializer = self.embeddings_initializer,
                name = 'question_difficulty_embedding'
            )
            question_emb_dim += embeddings_dimension

        ##### LEctures
        lecture_emb_dim = 0
        if encoded_lecture_id is not None:
            self.lid_emb_layer = tf.keras.layers.Embedding(
                1 + max(encoded_lecture_id),
                embeddings_dimension,
                embeddings_initializer = self.embeddings_initializer,
                name = 'lecture_id_embedding'
            )
            lecture_emb_dim += embeddings_dimension
        if lecture_part is not None:
            self.lp_emb_layer = tf.keras.layers.Embedding(
                1 + max(lecture_part),
                embeddings_dimension,
                embeddings_initializer = self.embeddings_initializer,
                name = 'lecture_part_embedding'
                )
            lecture_emb_dim += embeddings_dimension
        if lecture_tag is not None:
            self.lt_emb_layer = tf.keras.layers.Embedding(
                1 + max(lecture_tag),
                embeddings_dimension,
                embeddings_initializer = self.embeddings_initializer,
                name = 'lecture_tag_embedding'
                )
            lecture_emb_dim += embeddings_dimension
        if lecture_type_of is not None:
            self.ltype_emb_layer = tf.keras.layers.Embedding(
                1 + max(lecture_type_of),
                embeddings_dimension,
                embeddings_initializer = self.embeddings_initializer,
                name = 'lecture_type_embedding'
                )
            lecture_emb_dim += embeddings_dimension
        self.question_emb_dim = question_emb_dim
        self.lecture_emb_dim = lecture_emb_dim
        self.dense_1 = tf.keras.layers.Dense(4*embeddings_dimension)
        self.dense_2 = tf.keras.layers.Dense(4*embeddings_dimension)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
    def build(self, input_shape):
        #Questions
        self.ts_encoded_question_id = tf.convert_to_tensor(
            self.encoded_question_id)
 
        if self.question_bundle_id is not None:

            self.ts_question_bundle_id= tf.convert_to_tensor(
                self.question_bundle_id)
                
        if self.question_part is not None:
            self.ts_question_part = tf.convert_to_tensor(
                self.question_part
                )
        if self.question_tags is not None:
            self.ts_question_tags = tf.convert_to_tensor(
              np.vstack(
                [np.array(tags) for tags in self.question_tags]
                )
            )
            self.ts_tags_mask = tf.expand_dims(
                tf.cast(self.ts_question_tags > 0, tf.float32),-1)
        if self.question_popularity is not None:
            self.ts_question_popularity = tf.convert_to_tensor(
                self.question_popularity
            )
        if self.question_difficulty is not None:
            self.ts_question_difficulty = tf.convert_to_tensor(
                self.question_difficulty
            )
        self.computed_params = self.add_weight(
            shape = [
                     len(self.encoded_question_id) + 
                     len(self.encoded_lecture_id),
                     4*self.embeddings_dimension
                     ],
            initializer = 'zeros',
            trainable = False,
            aggregation = tf.VariableAggregation.ONLY_FIRST_REPLICA,
            name = 'computed_params'
        )
        self.params_computed = self.add_weight(
            shape=(),
            initializer = 'zeros',
            trainable = False,
            aggregation = tf.VariableAggregation.ONLY_FIRST_REPLICA,
            name = 'params_computed'
        )
        ##### Lectures

        self.ts_encoded_lecture_id = tf.convert_to_tensor(
            self.encoded_lecture_id)
          
        if self.lecture_part is not None:
            self.ts_lecture_part = tf.convert_to_tensor(self.lecture_part)

        if self.lecture_tag is not None:
            self.ts_lecture_tag = tf.convert_to_tensor(self.lecture_tag)
              
        if self.lecture_type_of is not None:

            self.ts_lecture_type_of =  tf.convert_to_tensor(self.lecture_type_of)
               
        super(ContentEmbeddingLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape + (self.model_dimension,)

    def _compute_content_params(self):
        question_emb = []
        if self.encoded_question_id is not None:
            question_emb.append(self.qid_emb_layer(
                self.ts_encoded_question_id
              )
            )
        if self.question_bundle_id is not None:

            question_emb.append(self.qbid_emb_layer(
                self.ts_question_bundle_id
                )
            )
        if self.question_part is not None:
            question_emb.append(self.qp_emb_layer(
                self.ts_question_part
                )
            )     
        if self.question_tags is not None:

            question_tags_emb = self.qt_emb_layer(
                self.ts_question_tags
                )*self.ts_tags_mask
            question_tags_emb = tf.reduce_sum(question_tags_emb, axis=1)
            question_emb.append(question_tags_emb)
        if self.question_popularity is not None:
            question_emb.append(self.qpopularity_emb_layer(
                self.ts_question_popularity
                )
            )
        if self.question_difficulty is not None:
            question_emb.append(self.qdifficulty_emb_layer(
                self.ts_question_difficulty
                )
            )
        ##### Lectures
        lecture_emb = []
        if self.encoded_lecture_id:
            lecture_emb.append(self.lid_emb_layer(
              self.ts_encoded_lecture_id
            ))

        if self.lecture_part is not None:
            lecture_emb.append(self.lp_emb_layer(
                self.ts_lecture_part
                )
            )
        if self.lecture_tag is not None:

            lecture_emb.append(self.lt_emb_layer(
                self.ts_lecture_tag
                )
            )
        if self.lecture_type_of is not None:
            lecture_emb.append(self.ltype_emb_layer(
                self.ts_lecture_type_of
                )
            )
        lecture_emb = tf.concat(lecture_emb, axis = -1)
        lecture_emb = self.dense_1(lecture_emb)
        question_emb = tf.concat(question_emb, axis = -1)
        question_emb = self.dense_2(question_emb)
        content_emb_params = tf.concat([question_emb,
                                        lecture_emb], axis=0)
        content_emb_params = self.layer_norm(content_emb_params)
        return content_emb_params

    def call(self, encoded_content_id, training = True):
        if training:
            content_emb_params = self._compute_content_params()
            if self.params_computed > 0.5:
                self.params_computed.assign(0.0)
            return  tf.gather(content_emb_params, encoded_content_id)
        else:
            if self.params_computed < 0.5: 
                self.computed_params.assign(self._compute_content_params())
                self.params_computed.assign(1.0)
            return tf.gather(self.computed_params, encoded_content_id)
class RiiidAnswerModel(tf.keras.Model):
    r"""
    RiiidAnswerModel to output the logits of the target variable (answered_correctly)
    """
    def __init__(self, 
                 encoded_content_map,
                 model_dimension = 512,
                 embeddings_dimension = 64,
                 attention_num_heads = 8,
                 attention_dropout = 0.0,
                 feedforward_dimension = 2048,
                 num_layers = 4,
                 num_encoder_layers = None,
                 num_decoder_layers = None,
                 timestamp_scale =  60*1000.0, #1 minute
                 max_elapsed_time = 300000.0, #300 seconds
                 elapsed_time_bins = 2048,
                 elapsed_time_wsize = 41,
                 max_time_lag =  1*24*60*60*1000, #1 days
                 time_lag_bins = 2048,
                 time_lag_wsize = 41,
                 timediff_attn: bool = True,
                 return_attn_coef: bool = False,
                 activation = 'gelu',
                 embeddings_initializer = 
                    tf.keras.initializers.TruncatedNormal(stddev=0.02),
                 attn_weights_initializer = 
                    tf.keras.initializers.RandomUniform(
                     minval = 0.0,
                     maxval = 2.0),
                 **kwargs
                 ):
        super(RiiidAnswerModel, self).__init__(**kwargs)
        self.encoded_content_map  = encoded_content_map
        self.num_question  = len(encoded_content_map['encoded_question_id'])
        self.model_dimension = model_dimension
        self.embeddings_dimension = embeddings_dimension
        self.attention_num_heads = attention_num_heads
        self.attention_dropout = attention_dropout
        self.feedforward_dimension = feedforward_dimension
        self.num_layers = num_layers
        if num_encoder_layers is None:
            num_encoder_layers = num_layers
        self.num_encoder_layers = num_encoder_layers
        if num_decoder_layers is None:
            num_decoder_layers = num_layers
        self.num_decoder_layers = num_decoder_layers

        self.timestamp_scale = timestamp_scale
        self.max_elapsed_time = max_elapsed_time
        self.max_time_lag = max_time_lag
        self.elapsed_time_bins = elapsed_time_bins
        self.time_lag_bins = time_lag_bins
        self.elapsed_time_wsize = elapsed_time_wsize
        self.time_lag_wsize = time_lag_wsize
        self.timediff_attn = timediff_attn
        self.return_attn_coef = return_attn_coef
        self.activation = activation
        self.embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)
        self.attn_weights_initializer = tf.keras.initializers.get(attn_weights_initializer)
        self.content_emb_layer = ContentEmbeddingLayer(
            embeddings_dimension,
            **encoded_content_map,
            embeddings_initializer = self.embeddings_initializer,
            name = 'content_embedding'
                                  )
        
        self.first_token_emb_layer = tf.keras.layers.Embedding(
            1,
            model_dimension,
            embeddings_initializer = self.embeddings_initializer,
            name='first_token_embedding')
        
        self.aswcor_emb_layer = tf.keras.layers.Embedding(
            3,
            embeddings_dimension,
            embeddings_initializer = self.embeddings_initializer,
            name='answer_correctly_embedding')
        self.uasw_emb_layer = tf.keras.layers.Embedding(
            4*self.num_question,
            embeddings_dimension,
            embeddings_initializer = self.embeddings_initializer,
            name='user_answer_embedding')
        self.qexpl_emb_layer = tf.keras.layers.Embedding(
            2,
            embeddings_dimension,
            embeddings_initializer = self.embeddings_initializer,
            name='question_had_explanation_embedding')

        self.qelapsed_time_emb_layer = ContinuousEmbedding(
            embeddings_dimension, 
            embeddings_initializer = self.embeddings_initializer,
            num_points = elapsed_time_bins,
            minval  = 0, maxval  = 1.0,
            window_size = elapsed_time_wsize,
            window_type = 'hann',
            normalized = True,
            name='question_elapsed_time_embedding')

        self.timelag_emb_layer = ContinuousEmbedding(
            embeddings_dimension, 
            embeddings_initializer = self.embeddings_initializer,
            num_points = time_lag_bins,
            minval  = 0, maxval  = 1.0,
            window_size = time_lag_wsize,
            window_type = 'hann',
            normalized = True,
            name = 'time_lag_embedding' )
        self.emb_dense_layer_1 = tf.keras.layers.Dense(model_dimension)
        self.emb_dense_layer_2 = tf.keras.layers.Dense(model_dimension)
        self.encoders = [ EncoderLayer(model_dimension, attention_num_heads,
                                       feedforward_dimension, attention_dropout,
                                       return_attn_coef = return_attn_coef,
                                       activation = activation,
                                       attn_weights_initializer=self.attn_weights_initializer,
                                       name = f'encoder_layer_{i}')
                         for i in range(num_encoder_layers)
                         ]
        
        self.decoders = [ DecoderLayer(model_dimension, attention_num_heads, 
                                       feedforward_dimension, attention_dropout,
                                       return_attn_coef = return_attn_coef,
                                       activation = activation,
                                       attn_weights_initializer=self.attn_weights_initializer,
                                       name = f'decoder_layer_{i}')
                         for i in range(num_decoder_layers)
                         ]

        self.final_dense_layer = tf.keras.layers.Dense(1, name='final_dense_layer')

    def build(self, input_shape):
        assert(isinstance(input_shape, dict))
        assert('encoded_content_id' in input_shape)
        


    def call(self, inputs, training = None):
        timestamp = inputs['timestamp']
        time_lag = inputs['time_lag']
        encoded_content_id = inputs['encoded_content_id']
        answered_correctly = inputs['answered_correctly']
        user_answer = inputs['user_answer']
        question_elapsed_time = inputs['question_elapsed_time']
        question_had_explanation = inputs['question_had_explanation']
        non_padding_mask = inputs['non_padding_mask']
        #### Calculate embedding of questions and lectures and appending them
        content_emb = self.content_emb_layer(
            encoded_content_id,
            training = training
            )


        question_mask = tf.expand_dims(
            tf.cast(tf.less(encoded_content_id, self.num_question), 
                    tf.float32),
            -1)
        
        
        answered_correctly_emb = self.aswcor_emb_layer(
            1+answered_correctly
            ) * question_mask
        encoded_question_id = tf.clip_by_value(encoded_content_id, clip_value_min = 0, 
                                       clip_value_max = self.num_question - 1)
        user_answer_id = (
            4*encoded_question_id
            + tf.clip_by_value(user_answer, 
                               clip_value_min = 0, 
                               clip_value_max = 3)
        )
    
    
        user_answer_emb = self.uasw_emb_layer(user_answer_id) * question_mask
        
        question_had_explanation_emb = self.qexpl_emb_layer(
            tf.clip_by_value(
                question_had_explanation,
                clip_value_min  = 0,
                clip_value_max = 2
            )
         ) * question_mask

        ########## Question elapsed time embedding
        question_elapsed_time_float = tf.clip_by_value(
            question_elapsed_time/tf.constant(self.max_elapsed_time, dtype= tf.float32),
            clip_value_min = 0,
            clip_value_max = 999999.0 
        )
        question_elapsed_time_float =tf.math.sqrt(question_elapsed_time_float)
        question_elapsed_time_emb = self.qelapsed_time_emb_layer(
              question_elapsed_time_float
        )
        question_elapsed_time_emb *= question_mask

        ##### Time lag embedding
        time_lag_float = tf.clip_by_value(
            time_lag/tf.constant(self.max_time_lag, dtype= tf.float32),
            clip_value_min = 0,
            clip_value_max = 999999.0
            
        )
        time_lag_float = tf.math.sqrt(time_lag_float)
        time_lag_emb = self.timelag_emb_layer(
           time_lag_float
        )
        
        ### Transformer inputs
        encoder_value = self.emb_dense_layer_1(
            tf.concat([
                       content_emb,
                       answered_correctly_emb,
                       question_elapsed_time_emb,
                       question_had_explanation_emb,
                       user_answer_emb,
                       time_lag_emb
                       ], axis = -1)
            )
                    
    
        decoder_query =  self.emb_dense_layer_2(
            tf.concat([
                       content_emb,
                       time_lag_emb
                       ], axis = -1)
            )
                     
        first_token_emb = self.first_token_emb_layer(tf.zeros_like(timestamp[:, 0:1])) 
        #
        #To avoid this situation where no encoder position attends to the first position of the decoder,
        #the encoder value is pre-padded with a first_token_embedding which attends to all positions of the decoder
        #
        
        encoder_value = tf.concat([first_token_emb, encoder_value], axis = 1)
        # Compute the masks

        non_padding_mask_expanded = tf.expand_dims(non_padding_mask,1)
        dec_dec_mask = tf.cast(tf.greater_equal(tf.expand_dims(timestamp, -1), 
                  tf.expand_dims(timestamp, -2)), tf.float32)
        dec_dec_mask *= non_padding_mask_expanded

        enc_enc_mask = tf.pad(dec_dec_mask, [[0,0], [1, 0], [0,0]], constant_values = 0)
        enc_enc_mask = tf.pad(enc_enc_mask, [[0,0], [0,0], [1,0]], constant_values = 1)

        excluding_ahead_mask = tf.cast(
            tf.logical_or(
                  tf.greater(
                      tf.expand_dims(timestamp, -1),
                      tf.expand_dims(timestamp, -2)),
                  tf.expand_dims(timestamp[:1,:], -2) == -1    
                  ), 
                  tf.float32)
        excluding_ahead_mask *= non_padding_mask_expanded
        
        #
        # First token of the encoder output attends to all positions of the decoder
        #
        enc_dec_mask = tf.pad(excluding_ahead_mask, [[0,0], [0,0], [1,0]], constant_values = 1)
        if self.timediff_attn:
            timestamp_float = tf.cast(timestamp, tf.float32)

            timediff = tf.expand_dims(
                timestamp_float,
                -1) - tf.expand_dims(
                    timestamp_float, 
                -2)
            timediff *= tf.cast(timediff > 0, tf.float32)

            timediff /= tf.constant(self.timestamp_scale, tf.float32)
            timediff = tf.math.log1p(timediff)
            timediff = tf.expand_dims(timediff, -3) #batch_size, 1, seq_length, seq_length 
            enc_enc_timediff = tf.pad(timediff, [[0,0], [0,0], [1,0], [1,0]])
            enc_dec_timediff = tf.pad(timediff, [[0,0], [0,0], [0,0], [1,0]])
        if self.return_attn_coef:
            ee_attn_coefs = []
            dd_attn_coefs = []
            ed_attn_coefs = []

        for i, encoder in enumerate(self.encoders):
            enc_input = {
                'value': encoder_value, 
                 'mask': enc_enc_mask,
             }
            if self.timediff_attn:
                enc_input['timediff'] = enc_enc_timediff

            encoder_value = encoder(enc_input,
                                    training = training)
            if self.return_attn_coef:
                encoder_value, ee_attn_coef = encoder_value
                ee_attn_coefs.append(ee_attn_coef)
        for i, decoder in enumerate(self.decoders):
            dec_input = {
                'dec_query': decoder_query,
                'enc_out': encoder_value,
                'dec_dec_mask': dec_dec_mask,
                'enc_dec_mask': enc_dec_mask,
            }
            if self.timediff_attn:
                dec_input['dec_dec_timediff'] = timediff
                dec_input['enc_dec_timediff'] = enc_dec_timediff


            decoder_query = decoder(dec_input, 
                                      training = training)
            if self.return_attn_coef:
                decoder_query, dd_attn_coef, ed_attn_coef = decoder_query
                dd_attn_coefs.append(dd_attn_coef)
                ed_attn_coefs.append(ed_attn_coef)

        output = self.final_dense_layer(decoder_query)
        if self.return_attn_coef:
            return output, ee_attn_coefs, dd_attn_coefs, ed_attn_coefs
        else:
            return output
        