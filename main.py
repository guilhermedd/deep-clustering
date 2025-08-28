from autoencoder import Autoencoder
from kmeans import KMeansClustering
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

def get_data(data_file):
    """Lê os dados do arquivo, corrigindo o problema do append."""
    data = []
    with open(data_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        if line.startswith('#') or line.startswith('@'):
            continue
        parts = line.strip().split(':')
        features = [[float(x) for x in parts[0].split(',')], int(parts[1])]
        data.append(features) 

    return data

def main():
    train_file = 'datasets/heart/AbnormalHeartbeat_TRAIN.ts'
    sample_data = get_data(train_file)

    test_file = 'datasets/heart/AbnormalHeartbeat_TEST.ts'
    test_data = get_data(test_file)

    x_train = np.array([x[0] for x in sample_data])
    y_train_labels = np.array([data[1] for data in sample_data])

    x_val   = np.array([x[0] for x in test_data])
    y_test_labels = np.array([data[1] for data in test_data])
    
    print(f"Formato dos dados de treino: {x_train.shape}")
    print(f"Formato dos dados de validação: {x_val.shape}")

    input_shape = x_train.shape[1:]
    
    print("Pretraining autoencoder...")
    latent_dims_to_try = [4, 8, 16, 32] 
    
    autoencoder_handler = Autoencoder(
        input_shape=input_shape, 
        latent_dim_to_search=latent_dims_to_try,
        encoder_save_path="encoder_model_tf",
        decoder_save_path="decoder_model_tf"
    )

    best_model = autoencoder_handler.search_best_model(x_train, x_val)
    trained_model, history = autoencoder_handler.train(best_model, x_train, x_val)
    print("Autoencoder pretrained!")

    
    encoder = autoencoder_handler.encoder_model
    
    print(f"shape do teste {x_train.shape}")
    latent_spaces = encoder.predict(x_train)

    kmeans = KMeansClustering(n_clusters=2)
    kmeans.fit(latent_spaces)

    print(latent_spaces)
    predictions = kmeans.predict(latent_spaces) 

    kmeans.visualize_cluster_distribution(predictions, y_train_labels)
    kmeans.visualize_clusters_2d(latent_spaces, y_train_labels)

if __name__ == '__main__':
    main()