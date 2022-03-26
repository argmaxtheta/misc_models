import json
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from typing import Optional, Union

from libs.models.tf_utils import (
    make_embeddings,
    GatedResidualNetwork,
    corr_loss,
    r2_loss,
)
from libs.hyperparam import HyperparamOptManager

K = tf.keras.backend
layers = tf.keras.layers
MSE = tf.keras.losses.MeanSquaredError()


class FeatureTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, feature_dims, batchnorm_mom, virtual_batch_size):
        super().__init__()
        self.feature_dims = feature_dims
        self.batchnorm_mom = batchnorm_mom
        self.virtual_batch_size = virtual_batch_size

        self.dense_f1 = layers.Dense(self.feature_dims * 2, use_bias=False)
        self.batchnorm_f1 = layers.BatchNormalization(
            momentum=self.batchnorm_mom, virtual_batch_size=self.virtual_batch_size
        )

    def call(self, x):

        x = self.dense_f1(x)
        x = self.batchnorm_f1(x)

        return x[..., : self.feature_dims] * tf.math.sigmoid(
            x[..., self.feature_dims :]
        )

    def get_config(self):

        config = super().get_config()

        config.update(
            {
                "feature_dims": self.feature_dims,
                "batchnorm_mom": self.batchnorm_mom,
                "virtual_batch_size": self.virtual_batch_size,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        user_attrs = ["feature_dims", "batchnorm_mom", "virtual_batch_size"]
        return cls(**{k: v for k, v in config.items() if k in user_attrs})


class TabNetFactory:

    is_tensorflow_model = True
    evaluation_metric = "pred_corr_loss"
    loss_function = MSE
    output_activation = None

    @classmethod
    def make(cls, params, final_only=False):

        # Sizes
        time_steps = int(params["total_time_steps"])
        input_size = int(params["input_size"])
        output_size = int(params["output_size"])
        num_asset_ids = 0  # int(params["num_asset_ids"])

        hidden_layer_size = int(params["hidden_layer_size"])
        batchnorm_mom = float(params["batchnorm_mom"])
        minibatch_size = int(params["minibatch_size"])
        learning_rate = float(params["learning_rate"])
        virtual_batch_size = int(params.get("virtual_batch_size", 0))
        virtual_batch_size = (
            None if virtual_batch_size == 0 or final_only else virtual_batch_size
        )
        num_enc_stacks = int(params["num_encoder_stacks"])
        num_dec_stacks = int(params["num_decoder_stacks"])
        sparsemax_relax_factor = float(params.get("sparsemax_relax_factor", 1.5))
        max_gradient_norm = float(params["max_norm"])
        lr_decay_steps = int(params["lr_decay_steps"])
        lr_decay_rate = float(params["lr_decay_rate"])
        seed = int(params["feature_seed"])
        features_per_latent = int(params["features_per_latent"])
        num_latents = int(params["num_latents"])
        latent_dims = int(params["latent_dims"])
        multitask_alpha = float(params["multitask_alpha"])
        l1_weight = float(params["l1_weight"])

        # Build model
        # ... set up inputs -- only permit tabular format
        inputs = tf.keras.layers.Input(
            shape=(
                time_steps,
                input_size,
            ),
        )
        if time_steps > 1:
            raise NotImplementedError()

        # ... setup input embeddings
        if num_asset_ids > 0:
            # create embeddings for static variables
            features, asset_ids = inputs[..., :-1], inputs[..., -1]
            id_embedding = layers.Embedding(num_asset_ids, 30, input_length=1)(
                asset_ids
            )[:, 0, :]
            context = id_embedding
            num_temporal_features = input_size - 1
        else:
            features = inputs
            context = None
            num_temporal_features = input_size
        features = features[:, 0, :]

        # ... setup inital inputs
        features = tf.reshape(features, [-1, num_temporal_features])
        features = layers.BatchNormalization(momentum=batchnorm_mom)(features)
        batch_size = tf.shape(features)[0]

        # ... add autoencoder features
        r = np.random.RandomState(seed)
        latents = []
        decoded_latents = []
        for _ in range(num_latents):
            feature_idx = r.choice(
                num_temporal_features,
                size=features_per_latent,
                replace=False,
            )
            latent = layers.Dense(
                int((features_per_latent + latent_dims) / 2), activation="swish"
            )(tf.gather(features, feature_idx, axis=-1))
            latent = layers.Dense(
                latent_dims, activity_regularizer=tf.keras.regularizers.L1(l1=l1_weight)
            )(latent)

            decoded = layers.Dense(
                int((latent_dims + output_size) / 2), activation="swish"
            )(latent)
            decoded = layers.Dense(int(output_size))(decoded)
            latents.append(latent)
            decoded_latents.append(decoded)

        # features = tf.concat([features] + latents, axis=-1)
        features = tf.concat(latents, axis=-1)
        decoded_latents = tf.concat(decoded_latents, axis=-1)
        num_temporal_features = features.get_shape().as_list()[-1]

        # ... initializes decision-step dependent variables.
        output_aggregated = tf.zeros([batch_size, hidden_layer_size])
        masked_features = features
        mask_values = tf.zeros([batch_size, num_temporal_features])
        aggregated_mask_values = tf.zeros([batch_size, num_temporal_features])
        complementary_aggregated_mask_values = tf.ones(
            [batch_size, num_temporal_features]
        )
        # total_entropy = 0

        # ... apply decision steps
        def apply_feature_transformer(shared_feature_blocks, x, state_size):
            # ... shared blocks
            transform_f1 = shared_feature_blocks[0](x)
            transform_f2 = shared_feature_blocks[1](transform_f1)
            transform_f2 = (transform_f1 + transform_f2) * np.sqrt(0.5)

            # ... indiv blocks
            transform_f3 = FeatureTransformerBlock(
                state_size, batchnorm_mom, virtual_batch_size
            )(transform_f2)
            transform_f3 = (transform_f2 + transform_f3) * np.sqrt(0.5)

            transform_f4 = FeatureTransformerBlock(
                state_size, batchnorm_mom, virtual_batch_size
            )(transform_f3)
            transform_f4 = (transform_f3 + transform_f4) * np.sqrt(0.5)

            return transform_f4

        shared_feature_blocks = [
            FeatureTransformerBlock(
                hidden_layer_size * 2, batchnorm_mom, virtual_batch_size
            )
            for _ in range(2)
        ]
        for ni in range(num_enc_stacks):

            transform_f4 = apply_feature_transformer(
                shared_feature_blocks, masked_features, state_size=hidden_layer_size * 2
            )

            # ... aggregate outputs
            if ni > 0 or num_enc_stacks == 1:
                decision_out = tf.nn.elu(transform_f4[:, :hidden_layer_size])

                # Decision aggregation.
                output_aggregated += decision_out

                # Aggregated masks are used for visualization of the
                # feature importance attributes.
                scale_agg = tf.reduce_sum(decision_out, axis=1) / (num_enc_stacks - 1)
                aggregated_mask_values += mask_values * scale_agg

            # ... keep coef features
            features_for_coef = transform_f4[:, hidden_layer_size:]

            # ... Apply attentive transformer
            if ni < num_enc_stacks - 1:
                # Determines the feature masks via linear and nonlinear
                # transformations, taking into account of aggregated feature use.
                mask_values = layers.Dense(num_temporal_features, use_bias=False)(
                    features_for_coef
                )
                mask_values = layers.BatchNormalization(
                    momentum=batchnorm_mom, virtual_batch_size=virtual_batch_size
                )(mask_values)
                mask_values *= complementary_aggregated_mask_values
                mask_values = tfa.layers.Sparsemax()(mask_values)

                # Relaxation factor controls the amount of reuse of features between
                # different decision blocks and updated with the values of
                # coefficients.
                complementary_aggregated_mask_values *= (
                    sparsemax_relax_factor - mask_values
                )

                # Entropy is used to penalize the amount of sparsity in feature
                # selection.
                # total_entropy += tf.reduce_mean(
                #    tf.reduce_sum(-mask_values * tf.log(mask_values + 1e-8), axis=1)
                # ) / (num_stacks - 1)

                # Feature selection.
                masked_features = tf.multiply(mask_values, features)

        # ... apply decoder
        h_size = hidden_layer_size
        x = output_aggregated
        for _ in range(num_dec_stacks):
            h_size /= 2
            tmp = FeatureTransformerBlock(
                int(h_size), batchnorm_mom, virtual_batch_size
            )(x)
            x = (tmp + layers.Dense(int(h_size))(x)) * np.sqrt(0.5)

        # ... format output
        outputs = layers.Dense(output_size, activation=cls.output_activation)(x)
        if cls.output_activation is not None:
            outputs *= 5

        outputs = K.reshape(outputs, (-1, time_steps, output_size))
        decoded_latents = K.reshape(decoded_latents, (-1, time_steps, num_latents))
        if final_only:
            outputs = outputs[:, -1, :output_size]
        else:
            # combine with latents
            outputs = tf.concat([outputs, decoded_latents], axis=-1)

        # set losses
        def multitask_loss(y_true, y_pred):

            reconstruction_loss = 0.0
            for i in range(num_latents):
                reconstruction_loss += (
                    cls.loss_function(
                        y_true,
                        y_pred[..., output_size * (i + 1) : output_size * (i + 2)],
                    )
                    / num_latents
                )

            train_loss = (
                cls.loss_function(y_true, y_pred[..., :output_size])
                + multitask_alpha * reconstruction_loss
            )

            return train_loss

        def pred_corr_loss(y_true, y_pred):
            return corr_loss(y_true, y_pred[..., :output_size])

        # configure model & losses
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=lr_decay_steps,
            decay_rate=lr_decay_rate,
            staircase=True,
        )
        adam = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, clipnorm=max_gradient_norm
        )

        if final_only:
            model.compile(optimizer=adam)
        else:
            model.compile(loss=multitask_loss, optimizer=adam, metrics=[pred_corr_loss])

        return model

    @staticmethod
    def get_hyperparam_ranges():
        return {
            "latent_dims": [5],
            "features_per_latent": [30],
            "num_latents": [60],
            "hidden_layer_size": [128],
            "minibatch_size": [1024 * 2],
            "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1],
            "max_norm": [10000.0],  # remove
            "num_encoder_stacks": [1, 3],  # [1, 3, 5],
            "num_decoder_stacks": [1, 3],  # [1, 3, 5],
            "batchnorm_mom": [0.6, 0.7, 0.8, 0.9, 0.95, 0.98],
            "sparsemax_relax_factor": [1.0, 1.2, 1.5, 2.0],
            "virtual_batch_size": [
                0,
            ],
            "lr_decay_rate": [0.4, 0.8, 0.9, 0.95],
            "lr_decay_steps": [100, 500, 1000, 2000, 4000],
            "multitask_alpha": [0.0, 0.001, 0.01, 0.1, 1.0],
            "l1_weight": [0.0, 0.001, 0.01, 0.1, 1.0, 10.0],
            "feature_seed": list(HyperparamOptManager.get_random_seeds(1000)),
        }


class BasicTabNetFactory:

    is_tensorflow_model = True
    evaluation_metric = "corr_loss"
    loss_function = MSE
    output_activation = None

    @classmethod
    def make(cls, params, final_only=False):

        # Sizes
        time_steps = int(params["total_time_steps"])
        input_size = int(params["input_size"])
        output_size = int(params["output_size"])
        num_asset_ids = 0  # int(params["num_asset_ids"])

        hidden_layer_size = int(float(params["hidden_layer_size"]) * input_size)
        batchnorm_mom = float(params["batchnorm_mom"])
        minibatch_size = int(params["minibatch_size"])
        learning_rate = float(params["learning_rate"])
        virtual_batch_size = int(params.get("virtual_batch_size", 0))
        virtual_batch_size = (
            None if virtual_batch_size == 0 or final_only else virtual_batch_size
        )
        num_enc_stacks = int(params["num_encoder_stacks"])
        num_dec_stacks = int(params["num_decoder_stacks"])
        sparsemax_relax_factor = float(params.get("sparsemax_relax_factor", 1.5))
        max_gradient_norm = float(params["max_norm"])
        lr_decay_steps = int(params["lr_decay_steps"])
        lr_decay_rate = float(params["lr_decay_rate"])

        # Build model
        # ... set up inputs -- only permit tabular format
        inputs = tf.keras.layers.Input(
            shape=(
                time_steps,
                input_size,
            ),
        )
        if time_steps > 1:
            raise NotImplementedError()

        # ... setup input embeddings
        if num_asset_ids > 0:
            # create embeddings for static variables
            features, asset_ids = inputs[..., :-1], inputs[..., -1]
            id_embedding = layers.Embedding(num_asset_ids, 30, input_length=1)(
                asset_ids
            )[:, 0, :]
            context = id_embedding
            num_temporal_features = input_size - 1
        else:
            features = inputs
            context = None
            num_temporal_features = input_size
        features = features[:, 0, :]

        # ... setup inital inputs
        features = tf.reshape(features, [-1, num_temporal_features])
        features = layers.BatchNormalization(momentum=batchnorm_mom)(features)
        batch_size = tf.shape(features)[0]

        num_temporal_features = features.get_shape().as_list()[-1]

        # ... initializes decision-step dependent variables.
        output_aggregated = tf.zeros([batch_size, hidden_layer_size])
        masked_features = features
        mask_values = tf.zeros([batch_size, num_temporal_features])
        aggregated_mask_values = tf.zeros([batch_size, num_temporal_features])
        complementary_aggregated_mask_values = tf.ones(
            [batch_size, num_temporal_features]
        )
        # total_entropy = 0

        # ... apply decision steps
        def apply_feature_transformer(shared_feature_blocks, x, state_size):
            # ... shared blocks
            transform_f1 = shared_feature_blocks[0](x)
            transform_f2 = shared_feature_blocks[1](transform_f1)
            transform_f2 = (transform_f1 + transform_f2) * np.sqrt(0.5)

            # ... indiv blocks
            transform_f3 = FeatureTransformerBlock(
                state_size, batchnorm_mom, virtual_batch_size
            )(transform_f2)
            transform_f3 = (transform_f2 + transform_f3) * np.sqrt(0.5)

            transform_f4 = FeatureTransformerBlock(
                state_size, batchnorm_mom, virtual_batch_size
            )(transform_f3)
            transform_f4 = (transform_f3 + transform_f4) * np.sqrt(0.5)

            return transform_f4

        shared_feature_blocks = [
            FeatureTransformerBlock(
                hidden_layer_size * 2, batchnorm_mom, virtual_batch_size
            )
            for _ in range(2)
        ]
        for ni in range(num_enc_stacks):

            transform_f4 = apply_feature_transformer(
                shared_feature_blocks, masked_features, state_size=hidden_layer_size * 2
            )

            # ... aggregate outputs
            if ni > 0 or num_enc_stacks == 1:
                decision_out = tf.nn.elu(transform_f4[:, :hidden_layer_size])

                # Decision aggregation.
                output_aggregated += decision_out

                # Aggregated masks are used for visualization of the
                # feature importance attributes.
                scale_agg = tf.reduce_sum(decision_out, axis=1) / (num_enc_stacks - 1)
                aggregated_mask_values += mask_values * scale_agg

            # ... keep coef features
            features_for_coef = transform_f4[:, hidden_layer_size:]

            # ... Apply attentive transformer
            if ni < num_enc_stacks - 1:
                # Determines the feature masks via linear and nonlinear
                # transformations, taking into account of aggregated feature use.
                mask_values = layers.Dense(num_temporal_features, use_bias=False)(
                    features_for_coef
                )
                mask_values = layers.BatchNormalization(
                    momentum=batchnorm_mom, virtual_batch_size=virtual_batch_size
                )(mask_values)
                mask_values *= complementary_aggregated_mask_values
                mask_values = tfa.layers.Sparsemax()(mask_values)

                # Relaxation factor controls the amount of reuse of features between
                # different decision blocks and updated with the values of
                # coefficients.
                complementary_aggregated_mask_values *= (
                    sparsemax_relax_factor - mask_values
                )

                # Entropy is used to penalize the amount of sparsity in feature
                # selection.
                # total_entropy += tf.reduce_mean(
                #    tf.reduce_sum(-mask_values * tf.log(mask_values + 1e-8), axis=1)
                # ) / (num_stacks - 1)

                # Feature selection.
                masked_features = tf.multiply(mask_values, features)

        # ... apply decoder
        h_size = hidden_layer_size
        x = output_aggregated
        for _ in range(num_dec_stacks):
            h_size /= 2
            tmp = FeatureTransformerBlock(
                int(h_size), batchnorm_mom, virtual_batch_size
            )(x)
            x = (tmp + layers.Dense(int(h_size))(x)) * np.sqrt(0.5)

        # ... format output
        outputs = layers.Dense(output_size, activation=cls.output_activation)(x)
        if cls.output_activation is not None:
            outputs *= 5

        if not final_only:
            outputs = K.reshape(outputs, (-1, time_steps, output_size))

        # configure model & losses
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=lr_decay_steps,
            decay_rate=lr_decay_rate,
            staircase=True,
        )
        adam = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, clipnorm=max_gradient_norm
        )

        if final_only:
            model.compile(optimizer=adam)
        else:
            model.compile(loss=cls.loss_function, optimizer=adam, metrics=[corr_loss])

        return model

    @staticmethod
    def get_hyperparam_ranges():
        return {
            "minibatch_size": [1024 * 2],
            "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1],
            "max_norm": [10000.0],  # remove
            "num_encoder_stacks": [1, 3],  # [1, 3, 5],
            "num_decoder_stacks": [1, 3],  # [1, 3, 5],
            "batchnorm_mom": [0.6, 0.7, 0.8, 0.9, 0.95, 0.98],
            "sparsemax_relax_factor": [1.0, 1.2, 1.5, 2.0],
            "virtual_batch_size": [
                0,
            ],
            "lr_decay_rate": [0.4, 0.8, 0.9, 0.95],
            "lr_decay_steps": [100, 500, 1000, 2000, 4000],
            "hidden_layer_size": [0.1, 0.25, 0.5, 1.0, 1.5, 2.0],
        }
