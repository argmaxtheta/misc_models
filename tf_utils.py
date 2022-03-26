import json
from os import lstat
import tensorflow as tf


K = tf.keras.backend
layers = tf.keras.layers
TimeDist = layers.TimeDistributed


def calc_sharpe(captured_returns):
    ave_ret = tf.math.reduce_mean(captured_returns)  # , axis=1)
    # ave_sq_rets = tf.reduce_mean(tf.square(captured_returns))  # , axis=1)
    # variance = ave_sq_rets - tf.square(ave_ret)
    # std = tf.sqrt(variance + 1e-9)
    std = tf.math.reduce_std(ave_ret)

    sharpe = ave_ret / std * tf.sqrt(260.0)
    return sharpe


def corr_loss(returns, forecasts):
    r_centred = returns - tf.math.reduce_mean(returns)
    f_centred = forecasts - tf.math.reduce_mean(forecasts)
    num = tf.math.reduce_mean(r_centred * f_centred)
    den = tf.math.reduce_std(returns) * tf.math.reduce_std(forecasts) + 1e-8

    return -num / den


def r2_loss(returns, forecasts):

    rss = tf.math.reduce_mean((returns - forecasts) ** 2)
    tss = tf.math.reduce_mean((returns - tf.math.reduce_mean(returns)) ** 2)

    return 1.0 - rss / tss


def make_embeddings(inputs, num_categories, embedding_size=1):

    real_indices = [i for i, categories in enumerate(num_categories) if categories <= 0]
    categorical_indices = [
        i for i, categories in enumerate(num_categories) if categories > 0
    ]
    categories_per_variable = [num_categories[k] for k in categorical_indices]

    encoder_size = K.int_shape(inputs)[1]

    real_inputs = tf.gather(inputs, real_indices, axis=-1)
    embeddings = [real_inputs]

    if categorical_indices:
        categorical_inputs = tf.gather(inputs, categorical_indices, axis=-1)

        for i, n in enumerate(categories_per_variable):
            encoded_variable = layers.Embedding(
                n, output_dim=embedding_size, input_length=encoder_size
            )(categorical_inputs[..., i])
            embeddings.append(encoded_variable)

    return tf.concat(embeddings, axis=-1)


# --------------------------------------------------------------------------------------------------
# layer making
def linear_layer(size, activation=None, use_time_distributed=False, use_bias=True):
    linear = layers.Dense(size, activation=activation, use_bias=use_bias)
    if use_time_distributed:
        linear = layers.TimeDistributed(linear)
    return linear


def make_mlp_block(
    inputs, hidden_layer_size, output_size=None, dropout_rate=0.0, activation="elu"
):
    if output_size is None:
        output_size = hidden_layer_size

    x = layers.Dense(hidden_layer_size, activation=activation, use_bias=True)(inputs)
    x = layers.Dropout(dropout_rate)(x)
    return layers.Dense(output_size, activation=None, use_bias=True)(x)


def make_resnet_block(inputs, hidden_layer_size, dropout_rate, activation="elu"):

    norm = layers.LayerNormalization()(inputs)

    x = make_mlp_block(
        norm,
        hidden_layer_size,
        dropout_rate=dropout_rate,
        activation=activation,
    )

    return layers.Dropout(dropout_rate)(x) + norm


# ... TFT bits
class GatedLinearUnit(tf.keras.layers.Layer):
    def __init__(
        self,
        output_dims,
        dropout_rate=0.0,
        use_time_distributed=False,
        l1_reg=0.0,
        l2_reg=0.0,
    ):
        super().__init__()
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.output_dims = output_dims

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.gate = tf.keras.layers.Dense(
            output_dims,
            activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.L1L2(
                l1=self.l1_reg,
                l2=self.l2_reg,
            ),
            bias_regularizer=tf.keras.regularizers.L1L2(l1=self.l1_reg, l2=self.l2_reg),
        )
        self.transform = tf.keras.layers.Dense(
            output_dims,
            kernel_regularizer=tf.keras.regularizers.L1L2(
                l1=self.l1_reg,
                l2=self.l2_reg,
            ),
            bias_regularizer=tf.keras.regularizers.L1L2(l1=self.l1_reg, l2=self.l2_reg),
        )

        if use_time_distributed:
            self.gate = TimeDist(self.gate)
            self.transform = TimeDist(self.transform)

    def call(self, x):
        dropout = self.dropout(x)
        return self.gate(dropout) * self.transform(dropout)

    @classmethod
    def from_config(cls, config):
        print(config)
        return cls(
            **{
                k: v
                for k, v in config.items()
                if k not in {"name", "trainable", "dtype"}
            }
        )


class GatedResidualNetwork(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_dims,
        output_dims=None,
        output_activation=None,
        hidden_activation="elu",
        dropout_rate=0.0,
        use_time_distributed=False,
        l2_reg=0.0,
        l1_reg=0.0,
    ):

        super().__init__()

        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.dropout_rate = dropout_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        self.mlp_hidden = tf.keras.layers.Dense(
            hidden_dims,
            activation=hidden_activation,
            kernel_regularizer=tf.keras.regularizers.L1L2(
                l1=self.l1_reg, l2=self.l2_reg
            ),
            bias_regularizer=tf.keras.regularizers.L1L2(l1=self.l1_reg, l2=self.l2_reg),
        )
        self.mlp_output = tf.keras.layers.Dense(
            hidden_dims,
        )
        self.glu = GatedLinearUnit(
            output_dims if output_dims is not None else hidden_dims,
            dropout_rate,
            use_time_distributed=use_time_distributed,
        )
        self.skip_transform = (
            tf.keras.layers.Dense(output_dims) if output_dims is not None else None
        )
        self.layer_norm = tf.keras.layers.LayerNormalization()

        self.output_activation = (
            tf.keras.layers.Activation(output_activation)
            if output_activation is not None
            else None
        )

        if use_time_distributed:
            self.mlp_hidden = TimeDist(self.mlp_hidden)
            self.mlp_output = TimeDist(self.mlp_output)
            self.skip_transform = (
                TimeDist(self.skip_transform)
                if self.skip_transform is not None
                else None
            )

        self.use_time_distributed = use_time_distributed

    def call(self, x, c=None):

        if self.use_time_distributed:

            _, time_steps, _ = x.get_shape().as_list()

            if c is not None:
                c = K.stack([c for _ in range(time_steps)], axis=1)

        if c is not None:
            combined = K.concatenate([x, c], axis=-1)
        else:
            combined = x
        layer = self.mlp_hidden(combined)
        layer = self.mlp_output(layer)

        if self.skip_transform is not None:
            x = self.skip_transform(x)

        skip = x + self.glu(layer)
        output = self.layer_norm(skip)

        if self.output_activation is not None:
            output = self.output_activation(output)

        return output

    def get_config(self):

        config = super().get_config()

        config.update(
            {
                "hidden_dims": self.hidden_dims,
                "output_dims": self.output_dims,
                "output_activation": self.output_activation,
                "hidden_activation": self.hidden_activation,
                "dropout_rate": self.dropout_rate,
                "use_time_distributed": self.use_time_distributed,
                "l1_reg": self.l1_reg,
                "l2_reg": self.l2_reg,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        print(config)
        return cls(
            **{
                k: v
                for k, v in config.items()
                if k not in {"name", "trainable", "dtype"}
            }
        )


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += mask * -1e9

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1
    )  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def create_causal_attention_mask(size):
    """Ensures each output only refers to previous input."""
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


class InterpretableMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate):
        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(
            self.depth
        )  # common set of values across matrix.

        self.dense = tf.keras.layers.Dense(d_model)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, depth)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = K.stack(
            [v for _ in range(self.num_heads)], axis=1
        )  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )

        # Dropout to improve generalisation across representation subspaces
        scaled_attention = self.dropout(scaled_attention)

        # Average across heads
        scaled_attention = K.mean(
            scaled_attention, axis=1
        )  # (batch_size, num_heads, seq_len_q, seq_len_k)

        # Average attention across heads
        attention_weights = K.mean(attention_weights, axis=1)

        output = self.dense(scaled_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def apply_interp_multi_head_attention_block(
    input,
    encoder_size,
    hidden_layer_size,
    num_heads,
    dropout_rate,
    apply_causal_mask=True,
):

    # temporal self attention
    mask = create_causal_attention_mask(encoder_size) if apply_causal_mask else None
    mha, attn = InterpretableMultiHeadAttention(
        hidden_layer_size, num_heads, dropout_rate
    )(input, k=input, q=input, mask=mask)
    mha = GatedLinearUnit(hidden_layer_size, dropout_rate, use_time_distributed=True)(
        mha
    )
    mha = tf.keras.layers.LayerNormalization()(mha + input)

    output = GatedResidualNetwork(
        hidden_layer_size, dropout_rate=dropout_rate, use_time_distributed=True
    )(mha)
    return output, attn
