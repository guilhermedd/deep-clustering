import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.optimizers import Adam
import os


class Autoencoder:
    def __init__(self, input_shape: tuple, latent_dim_to_search: list = None, encoder_save_path: str = None, decoder_save_path: str = None):
        self.input_shape = input_shape
        self.latent_dim_to_search = latent_dim_to_search if latent_dim_to_search else [4]
        self.normalization_layer = layers.Normalization(axis=-1)
        self.encoder_save_path = encoder_save_path
        self.decoder_save_path = decoder_save_path

    def build_encoder(self, hp):
        inp = Input(shape=self.input_shape, name="encoder_input")
        x = self.normalization_layer(inp)
        x = layers.Flatten()(x)

        encoder_layer_configs = []

        units_enc_1 = hp.Int('units_enc_1', min_value=128, max_value=1024, step=64)
        units_enc_2 = hp.Int('units_enc_2', min_value=16, max_value=units_enc_1 // 2, step=8)
        units_enc_3 = hp.Int('units_enc_3', min_value=4, max_value=units_enc_2 // 2, step=4)
        
        
        x = layers.Dense(units_enc_1, activation='relu')(x)
        x = layers.Dense(units_enc_2, activation='relu')(x)
        x = layers.Dense(units_enc_3, activation='relu')(x)

        # Camada do espaço latente
        latent_dim = hp.Choice("latent_dim", self.latent_dim_to_search)
        z = layers.Dense(latent_dim, name='latent_space')(x)
        
        encoder_model = Model(inputs=inp, outputs=z, name="encoder")
        
        encoder_layer_configs.append({'units': units_enc_1, 'activation': 'relu'})
        encoder_layer_configs.append({'units': units_enc_2, 'activation': 'relu'})
        encoder_layer_configs.append({'units': units_enc_3, 'activation': 'relu'})

        return encoder_model, encoder_layer_configs

    def build_decoder(self, hp, encoder_layer_configs):
        """Constrói o modelo do Decoder usando a configuração invertida do encoder."""
        latent_dim = hp.get("latent_dim")
        
        decoder_input = Input(shape=(latent_dim,), name="decoder_input")
        x = decoder_input

        for config in reversed(encoder_layer_configs):
            x = layers.Dense(units=config['units'], activation=config['activation'])(x)

        reconstructed_flat = layers.Dense(np.prod(self.input_shape), activation='linear')(x)
        
        ### ALTERAÇÃO: Remova a camada Lambda e use uma camada de Normalização invertida.
        # Esta camada será parte do modelo do decoder, tornando-o autocontido.
        denormalization_layer = layers.Normalization(axis=-1, invert=True, name='denormalization_layer')
        
        # A camada de desnormalização precisa dos mesmos pesos (mean/variance) da camada original.
        # Passaremos esses pesos no método _build_autoencoder.
        denormalized_output = denormalization_layer(reconstructed_flat)

        out = layers.Reshape(self.input_shape)(denormalized_output)

        decoder_model = Model(inputs=decoder_input, outputs=out, name="decoder")
        return decoder_model

    def _build_autoencoder(self, hp):
        encoder, encoder_configs = self.build_encoder(hp)
        decoder = self.build_decoder(hp, encoder_configs)
        
        decoder.get_layer('denormalization_layer').set_weights(
            self.normalization_layer.get_weights()
        )
        
        inp = encoder.input
        z = encoder(inp)
        reconstruction = decoder(z)
        
        autoencoder = Model(inputs=inp, outputs=reconstruction, name="autoencoder")
        
        learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-6])
        autoencoder.compile(optimizer=Adam(learning_rate), loss='mse')
        
        return autoencoder

    def search_best_model(self, x_train, x_val):
        """Busca os melhores hiperparâmetros e retorna o melhor modelo não treinado."""
        self.normalization_layer.adapt(x_train)

        tuner = kt.BayesianOptimization(
            self._build_autoencoder,
            objective="val_loss",
            max_trials=100, 
            directory="keras-heart",
            project_name="dense_autoencoder_bayesian" 
        )

        stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2, # Fator de redução (new_lr = lr * factor)
            patience=5,
            min_lr=1e-6, 
            verbose=1
        )

        print("\n--- Iniciando a busca por hiperparâmetros ---")
        tuner.search(
            x_train, x_train,
            validation_data=(x_val, x_val),
            epochs=50,
            batch_size=64, 
            callbacks=[stop_early, reduce_lr],
            verbose=1
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("\n--- Melhores hiperparâmetros encontrados: ---")
        for param, value in best_hps.values.items():
            print(f"{param}: {value}")

        return tuner.hypermodel.build(best_hps)

    def train(self, model, x_train, x_val, epochs=200, batch_size=32):
        stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)

        try:
            print(f"\nCarregando modelo salvos...")
            self.load_model()
            history = None
        except FileNotFoundError:
            print("\n--- Modelo não encontrado, iniciando o treinamento final do melhor modelo ---")
            stop_early_fit = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15, 
                verbose=1,
                restore_best_weights=True
            )

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2, # Fator de redução (new_lr = lr * factor)
                patience=5,
                min_lr=1e-6, 
                verbose=1
            )

            all_callbacks = [stop_early_fit, reduce_lr]

            history = model.fit(
                x_train, x_train,
                validation_data=(x_val, x_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=all_callbacks,
                verbose=1
            )

            final_val_loss = model.evaluate(x_val, x_val, verbose=0)
            print(f"\nPerda final no conjunto de validação (MSE): {final_val_loss:.6f}")

            self.autoencoder_model = model
            self.encoder_model = model.get_layer("encoder")
            self.decoder_model = model.get_layer("decoder")

            self.save_model()

        np.set_printoptions(precision=4, suppress=True)
        sample_to_predict = x_val[0:1]
        reconstructed_sample = self.autoencoder_model.predict(sample_to_predict)

        print("\n--- Comparando Amostra Original vs. Reconstruída ---")
        print(f"Original:     {sample_to_predict[0][:10]}")
        print(f"Reconstruída: {reconstructed_sample[0][:10]}")

        mse = np.mean(np.square(sample_to_predict - reconstructed_sample))
        print(f"\nErro Quadrático Médio (MSE) para esta amostra: {mse:.6f}")

        tf.keras.backend.clear_session()
        return self.autoencoder_model, history

    def save_model(self):
        if self.encoder_model:
            self.encoder_model.save(self.encoder_save_path, save_format="tf")
        if self.decoder_model:
            self.decoder_model.save(self.decoder_save_path, save_format="tf")
        print("Modelos salvos com sucesso!")

    def load_model(self):
        if os.path.exists(self.encoder_save_path) and os.path.exists(self.decoder_save_path):
            self.encoder_model = tf.keras.models.load_model(self.encoder_save_path)
            self.encoder_model.summary()
            self.decoder_model = tf.keras.models.load_model(self.decoder_save_path)
            self.decoder_model.summary()
            
            self.join_parts() 
            
            print("Modelos carregados e juntados com sucesso!")
        else:
            raise FileNotFoundError("Não foi possível encontrar os arquivos salvos do encoder ou decoder.")

    def join_parts(self):
        encoder_input = self.encoder_model.input

        encoded_output = self.encoder_model(encoder_input)
        decoded_output = self.decoder_model(encoded_output)

        self.autoencoder_model = tf.keras.Model(inputs=encoder_input, outputs=decoded_output, name='autoencoder')
        
        print("Modelos juntados com sucesso!")
        self.autoencoder_model.summary()
