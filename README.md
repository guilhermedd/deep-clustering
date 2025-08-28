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

O codigo espera dados do tipo `<ARQUIVO>.ts`, que segue o padrão:
```
@<INFO>
@<INFO>
@problemName Heartbeat
@timeStamps false
@missing  false
@univariate true
@equalLength true
@seriesLength 18530
@classLabel true 0 1
@data
<LISTA_DE_DADOS>:<CLASSE_RESPECTIVA>
```
Sendo  que:
- <LISTA_DE_DADOS> espera valores de tipo float separados por ',' (1.3,2.5,-3.0)
- <CLASSE_RESPECTIVA> espera valores inteiros (0, 1, 2, ...)


Para mudar o tipo de entrada dos dados, você pode modificar a função `get_data`, dentro de `main.py` para interpretar os dados de acordo com o novo formato desejado.