# Time Series Clustering com Autoencoder + KMeans

Este projeto implementa um pipeline de aprendizado **não supervisionado** para séries temporais rotuladas. Ele combina:

- **Autoencoder**: para extrair representações compactas (espaço latente) das séries.
- **KMeans**: para agrupar essas representações e analisar padrões.

O exemplo usa o dataset `AbnormalHeartbeat` no formato `.ts` (estilo UCR/UEA), com labels por amostra.


## Estrutura do Projeto

```
├── autoencoder.py # Classe Autoencoder (construção, treino e busca do melhor modelo)
├── kmeans.py # Classe KMeansClustering (fit, predict, visualizações)
├── main.py # Script principal (pipeline completo)
├── datasets/
├── encoder_model_tf/ # (gerado) pesos do encoder
├── decoder_model_tf/ # (gerado) pesos do decoder
└── README.md
```