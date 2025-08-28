import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd


class KMeansClustering:
    def __init__(
            self,
            n_clusters: int = 5, # Alterado para 5, como no seu exemplo
            max_iter: int = 300,
            n_init: int = 10,
            random_state: int = 42):
        
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.model = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state
        )

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        X_formatted = np.array(X, dtype=np.float32)
        
        if X_formatted.ndim == 1:
            X_formatted = X_formatted.reshape(1, -1)
            
        return self.model.predict(X_formatted)
        
    def visualize_confusion_matrix_normalized(self, y_pred, y_true):
        cm = confusion_matrix(y_true, y_pred, normalize='true')  # normaliza por linha
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Matriz de Confusão Normalizada')
        plt.ylabel('Classes Reais')
        plt.xlabel('Clusters Previstos')
        plt.savefig('heatmap_normalized.png')

    def visualize_cluster_distribution(self, y_pred, y_true):
        df = pd.DataFrame({'cluster': y_pred, 'true_label': y_true})
        counts = df.groupby(['cluster', 'true_label']).size().unstack(fill_value=0)
        counts.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')
        plt.xlabel('Cluster')
        plt.ylabel('Número de amostras')
        plt.title('Distribuição das Classes Reais em cada Cluster')
        plt.savefig('cluster_distribution.png')
        

    def visualize_clusters_2d(self, X, y_true):
        X_2d = PCA(n_components=2).fit_transform(X)
        plt.figure(figsize=(12, 5))

        # Classes reais
        plt.subplot(1, 2, 1)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='tab10', alpha=0.7)
        plt.title('Classes Reais')
        plt.xlabel('PC1')
        plt.ylabel('PC2')

        # Clusters previstos
        y_pred = self.predict(X)
        plt.subplot(1, 2, 2)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, cmap='tab10', alpha=0.7)
        plt.title('Clusters Previstos (KMeans)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')

        plt.tight_layout()
        plt.savefig('cluster_visualization.png')