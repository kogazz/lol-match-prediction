# Predição de Vitórias no League of Legends com Machine Learning

Este repositório contém um estudo sobre o sistema de formação de times no **League of Legends** (LoL), utilizando **Machine Learning** para prever se um time vencerá ou perderá antes mesmo do jogo começar. O objetivo é entender se o balanceamento das partidas realmente funciona ou se há padrões que favorecem um dos lados.

## 📌 Objetivo do Projeto

Este projeto tem como principal objetivo:

- ✅ Analisar dados históricos de partidas ranqueadas
- ✅ Identificar padrões que possam influenciar no resultado das partidas
- ✅ Testar se um modelo de Machine Learning consegue prever a vitória antes do jogo começar
- ✅ Avaliar a eficiência do sistema de balanceamento do jogo

## 🏗️ Estrutura do Código

O código está dividido nas seguintes etapas:

1. **Análise Exploratória de Dados (EDA)**: Carregamos e exploramos os dados para entender sua estrutura e principais características.
2. **Pré-processamento**: Selecionamos variáveis relevantes, normalizamos os dados e codificamos variáveis categóricas.
3. **Treinamento do Modelo**: Treinamos um **Random Forest Classifier** para prever o resultado das partidas.
4. **Avaliação do Modelo**: Medimos a acurácia, geramos a matriz de confusão e analisamos a importância das variáveis.

## 📊 Análise Exploratória

Para compreender melhor os dados, realizamos as seguintes análises:

- 📌 Distribuição das vitórias e derrotas
- 📌 Correlação entre variáveis (LP, maestria, histórico de vitórias e derrotas)
- 📌 Comparação entre os times para verificar possíveis desequilíbrios

### 📌 Resultados da EDA

- ✅ As vitórias e derrotas estão distribuídas de forma relativamente equilibrada (\~50% para cada lado)
- ✅ Nenhuma variável pré-jogo mostrou uma forte correlação com a vitória
- ✅ O sistema de formação de partidas da Riot Games parece estar balanceando os jogos de forma eficiente

## 🤖 Modelagem e Avaliação

### 🔎 Modelo Utilizado

Utilizamos um **Random Forest Classifier** devido à sua capacidade de lidar com dados tabulares e capturar interações entre variáveis.

### 📈 Métricas de Avaliação

O modelo foi avaliado com:

- 📌 **Acurácia**: Percentual de previsões corretas
- 📌 **Matriz de Confusão**: Mostra os erros e acertos na previsão de vitórias e derrotas



### 📌 Resultados do Modelo

- ✅ **Acurácia de 56%**: Apenas um pouco acima de um chute aleatório (50%)
- ✅ Nenhuma variável pré-jogo teve um impacto significativo na previsão da vitória
- ✅ O modelo não conseguiu prever as vitórias com alta precisão, sugerindo que o sistema de balanceamento já equilibra as partidas

## 🚀 Como Executar o Código

Para rodar o código no seu ambiente:

1. **Clone o repositório:**

```bash
git clone https://github.com/seu-repositorio/lol-match-prediction.git
cd lol-match-prediction
```

2. **Instale as bibliotecas necessárias:**

```bash
pip install pandas matplotlib seaborn scikit-learn
```

3. **Execute o script:**

```bash
python lol_match_analysis.py
```

