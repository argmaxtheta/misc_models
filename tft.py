from email.errors import HeaderMissingRequiredValue
from unicodedata import bidirectional
import tensorflow as tf
import json

K = tf.keras.backend
layers = tf.keras.layers
TimeDist = tf.keras.layers.TimeDistributed

from libs.models.tf_utils import (
    linear_layer,
    GatedLinearUnit,
    GatedResidualNetwork,
    apply_interp_multi_head_attention_block,
)


def SMAPE(y_true, y_pred):
    return tf.reduce_mean(
        tf.abs(y_true - y_pred) / (tf.abs(y_true) + tf.abs(y_pred)) * 200
    )


def make_tft_embeddings(x, category_defn, hidden_layer_size, time_steps):
    outputs = []
    for i, num_categories in enumerate(category_defn):

        traj = x[..., i]

        if num_categories > 0:
            # Categorical embeddings
            traj = tf.keras.layers.Embedding(
                input_dim=num_categories,
                output_dim=hidden_layer_size,
                input_length=time_steps,
                dtype=tf.float32,
            )(traj)
        else:
            # Linear transform
            traj = tf.expand_dims(traj, axis=-1)

            traj = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(hidden_layer_size)
            )(traj)

        outputs.append(traj)

    return outputs


def build_tft_graph(params, inputs=None, bidirectional=False):

    # input codes
    default_input_types = {"static": 0, "known": 1, "observed": 2}

    # Sizes
    time_steps = int(params["total_time_steps"])
    input_size = int(params["input_size"])
    output_size = int(params["output_size"])
    num_categories = json.loads(str(params["num_categories"]))

    # Network params
    hidden_layer_size = int(params["hidden_layer_size"])
    learning_rate = float(params["learning_rate"])
    max_gradient_norm = float(params["max_norm"])
    dropout_rate = float(params["dropout_rate"])
    minibatch_size = int(params["minibatch_size"])

    decoder_size = int(params["prediction_horizon"])
    encoder_size = int(params["total_time_steps"]) - decoder_size

    num_heads = int(params["num_heads"])
    num_stacks = int(params["num_stacks"])

    input_types = json.loads(str(params["input_types"]))

    # ----------------------------
    # Create input embeddings
    # ----------------------------
    if inputs is None:
        all_inputs = tf.keras.layers.Input(shape=(time_steps, input_size))
    else:
        all_inputs = inputs

    embeddings_list = make_tft_embeddings(
        all_inputs, num_categories, hidden_layer_size, time_steps
    )  # (batch x time x all feats x embed)

    static_inputs = [
        embed[:, 0, :]
        for i, embed in enumerate(embeddings_list)
        if input_types[i] == default_input_types["static"]
    ]  # feat x (batch x embed)
    known_inputs = [
        embed
        for i, embed in enumerate(embeddings_list)
        if input_types[i] == default_input_types["known"]
    ]  # feat x (batch x time x embed)

    obs_inputs = [
        embed
        for i, embed in enumerate(embeddings_list)
        if input_types[i] == default_input_types["observed"]
    ]  # feat x (batch x time x embed)

    encoder_inputs = [inputs[:, :encoder_size, ...] for inputs in known_inputs] + [
        inputs[:, :encoder_size, ...] for inputs in obs_inputs
    ]
    decoder_inputs = [inputs[:, encoder_size:, ...] for inputs in known_inputs]

    # ----------------------------
    # Create static contexts
    # ----------------------------
    def apply_variable_selection(
        embeddings, use_time_distributed=True, selection_context=None
    ):

        # Setup selection weights per time step
        flattened_vector = K.concatenate(
            embeddings, axis=-1
        )  # batch x time x embed.feats
        variable_selection_weights = GatedResidualNetwork(
            hidden_layer_size,
            output_dims=len(embeddings),
            output_activation="softmax",
            dropout_rate=dropout_rate,
            use_time_distributed=use_time_distributed,
        )(
            flattened_vector, c=selection_context
        )  # batch x time x feats

        # Get nonlinear features
        features = K.stack(
            [
                GatedResidualNetwork(
                    hidden_layer_size,
                    dropout_rate=dropout_rate,
                    use_time_distributed=use_time_distributed,
                )(l)
                for l in embeddings
            ],
            axis=-1,
        )  # batch x time x embeds x feats
        weights = tf.expand_dims(
            variable_selection_weights, axis=-2
        )  # batch x time x 1 x feats
        latent = tf.reduce_sum(features * weights, axis=-1)

        return latent, variable_selection_weights

    def get_static_contect_vectors(inputs, num_contexts):
        # Get static latent
        static_features, _ = apply_variable_selection(
            inputs, use_time_distributed=False
        )

        # Project to context vectors
        contexts = (
            GatedResidualNetwork(hidden_layer_size, dropout_rate=dropout_rate)(
                static_features
            )
            for _ in range(num_contexts)
        )

        return contexts

    if static_inputs:

        if bidirectional:
            (
                var_select_context,
                lstm_fwd_h_context,
                lstm_fwd_c_context,
                lstm_bak_h_context,
                lstm_bak_c_context,
                enrichment_context,
            ) = get_static_contect_vectors(static_inputs, num_contexts=6)
            lstm_context = [lstm_fwd_h_context, lstm_fwd_c_context]
            bak_context = [lstm_bak_h_context, lstm_bak_c_context]

        else:
            (
                var_select_context,
                lstm_fwd_h_context,
                lstm_fwd_c_context,
                enrichment_context,
            ) = get_static_contect_vectors(static_inputs, num_contexts=4)

            lstm_context = [lstm_fwd_h_context, lstm_fwd_c_context]
    else:
        raise NotImplementedError()

    # ----------------------------
    # Local processing (seq2seq)
    # ----------------------------
    # .. create input embeddings
    encoder, _ = apply_variable_selection(
        encoder_inputs, selection_context=var_select_context
    )
    decoder, _ = apply_variable_selection(
        decoder_inputs, selection_context=var_select_context
    )
    local_inputs = tf.concat([encoder, decoder], axis=1)

    # ... lstm encoder
    def make_seq2seq_layer(lstm_context, encoder, decoder, go_backwards=False):

        if go_backwards:
            # reverse encoder/decoder order to feed in backwards
            tmp = decoder
            decoder = encoder
            encoder = tmp

        lstm_encoder, h, c = tf.keras.layers.LSTM(
            hidden_layer_size,
            return_sequences=True,
            return_state=True,
            stateful=False,
            unroll=False,
            use_bias=True,
            go_backwards=go_backwards,
        )(encoder, initial_state=lstm_context)

        lstm_decoder = tf.keras.layers.LSTM(
            hidden_layer_size,
            return_sequences=True,
            return_state=False,
            stateful=False,
            unroll=False,
            use_bias=True,
            go_backwards=go_backwards,
        )(decoder, initial_state=[h, c])

        lstm_out = tf.concat([lstm_encoder, lstm_decoder], axis=1)

        if go_backwards:
            lstm_out = tf.reverse(lstm_out, axis=[1])

        return lstm_out

    lstm_out = make_seq2seq_layer(lstm_context, encoder, decoder, go_backwards=False)

    if bidirectional:
        back_out = make_seq2seq_layer(bak_context, encoder, decoder, go_backwards=True)
        lstm_out = tf.concat([lstm_out, back_out], axis=-1)

    # ... apply skip connection for lstm
    lstm_layer = GatedLinearUnit(
        hidden_layer_size, dropout_rate, use_time_distributed=True
    )(lstm_out)
    temporal_features = tf.keras.layers.LayerNormalization()(lstm_layer + local_inputs)

    # ----------------------------
    # Static enrichment layer
    # ----------------------------
    mha_input = GatedResidualNetwork(
        hidden_layer_size, dropout_rate=dropout_rate, use_time_distributed=True
    )(temporal_features, c=enrichment_context)

    # ----------------------------
    # Apply masked self-attention blocks
    # ----------------------------
    for _ in range(num_stacks):
        mha_input, _ = apply_interp_multi_head_attention_block(
            mha_input,
            time_steps,
            hidden_layer_size,
            num_heads,
            dropout_rate,
            apply_causal_mask=not bidirectional,
        )

    # ----------------------------
    # Output skip connections
    # ----------------------------
    # LSTM skip connection
    outputs = tf.keras.layers.LayerNormalization()(mha_input + temporal_features)

    # final dense output
    outputs = TimeDist(layers.Dense(output_size))(outputs[..., encoder_size:, :])

    return all_inputs, outputs


class TFTFactory:

    is_tensorflow_model = True
    evaluation_metric = "loss"
    loss_function = "MAE"

    @classmethod
    def make(
        cls, params,
    ):
        learning_rate = float(params["learning_rate"])
        max_gradient_norm = float(params["max_norm"])
        minibatch_size = int(params["minibatch_size"])

        inputs, outputs = build_tft_graph(params, bidirectional=True)
        # configure model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        adam = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, clipnorm=max_gradient_norm
        )
        model.compile(loss=cls.loss_function, optimizer=adam)

        return model

    @staticmethod
    def get_hyperparam_ranges():
        return {
            "dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "hidden_layer_size": [20, 40, 80],
            "minibatch_size": [64],  # [32, 64, 128, 256, 512],
            "learning_rate": [0.0001, 0.001, 0.01, 0.1],
            "max_norm": [10000.0],  # remove
            "num_stacks": [1],
            "num_heads": [1],
        }


class HybridTFTfactory(TFTFactory):
    loss_function = "MAPE"

    @classmethod
    def make(
        cls, params,
    ):

        params = dict(params)
        learning_rate = float(params["learning_rate"])
        max_gradient_norm = float(params["max_norm"])
        time_steps = int(params["total_time_steps"])
        input_size = int(params["input_size"])
        output_size = int(params["output_size"])
        decoder_size = int(params["prediction_horizon"])
        output_type = str(params["output_type"])

        inputs = tf.keras.layers.Input(shape=(time_steps, input_size))

        # setup inputs for graph and build it
        num_categories = json.loads(str(params["num_categories"]))
        params["num_categories"] = num_categories[:-4]
        input_types = json.loads(str(params["input_types"]))
        params["input_types"] = input_types[:-4]
        params["output_size"] = output_size * 2
        params["input_size"] = input_size - 4
        _, outputs = build_tft_graph(
            params, inputs[..., :-4], bidirectional=bool(params["bidirectional"])
        )

        # process outputs
        scalers = inputs[..., -4:]
        scale_min, scale_max, log_mean, log_std = (
            scalers[:, :decoder_size, i : i + 1] for i in range(4)
        )

        # ... absolute output
        abs_output = outputs[..., :1]
        abs_output *= scale_max - scale_min
        abs_output += scale_min

        # ... log output
        log_output = outputs[..., 1:]
        log_output *= log_std
        log_output += log_mean
        log_output = tf.keras.activations.exponential(log_output)

        # ... combine
        if output_type == "abs":
            pred = abs_output
        elif output_type == "log":
            pred = log_output
        elif output_type == "combi":
            pred = (abs_output + log_output) / 2
        else:
            raise ValueError(f"Unrecognised output type: {output_type}")

        # configure model
        model = tf.keras.Model(inputs=inputs, outputs=pred)

        adam = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, clipnorm=max_gradient_norm
        )
        model.compile(loss=cls.loss_function, optimizer=adam)

        return model

    @staticmethod
    def get_hyperparam_ranges():
        return {
            "dropout_rate": [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "hidden_layer_size": [20, 80, 160],
            "minibatch_size": [32],  # [32, 64, 128, 256, 512],
            "learning_rate": [0.0001, 0.001, 0.01],
            "max_norm": [10000.0],  # remove
            "num_stacks": [1, 2, 4],  # [1, 2, 4],
            "num_heads": [1],  # [1, 2, 4],
            "output_type": ["combi"],  # ["abs", "log", "combi"],
            "bidirectional": [True],
        }
