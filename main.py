import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# Carregar o dataset
df = pd.read_csv("/mnt/data/League of Legends Ranked Match Data  Season 15 (EUN).csv")

# Exibir informações gerais do dataset
print("Visão geral dos dados:")
print(df.info())
print("\nPrimeiras linhas do dataset:")
print(df.head())

# Verificar a distribuição da variável alvo (win)
plt.figure(figsize=(6, 4))
sns.countplot(x=df['win'], palette='Blues')
plt.xlabel("Resultado")
plt.ylabel("Número de Partidas")
plt.title("Distribuição de Vitórias e Derrotas")
plt.xticks([0, 1], ["Derrota", "Vitória"])
plt.show()

# Selecionar colunas relevantes para a modelagem
selected_columns = ["win", "solo_lp", "solo_losses", "mastery_points"]
df_selected = df[selected_columns].dropna()

# Exibir estatísticas descritivas
df_selected.describe()

# Transformar a variável alvo (win) para valores binários
df_selected["win"] = df_selected["win"].astype(int)

# Analisar a correlação entre as variáveis
plt.figure(figsize=(6, 4))
sns.heatmap(df_selected.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlação entre as Variáveis")
plt.show()

# Separar dados para treino e teste
X = df_selected.drop(columns=["win"])
y = df_selected["win"]

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Treinar o modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Avaliação do modelo
accuracy = (y_pred == y_test).mean()
print(f"Acurácia do Modelo: {accuracy:.2f}")
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# Gerar Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Derrota", "Vitória"], yticklabels=["Derrota", "Vitória"])
plt.xlabel("Previsões")
plt.ylabel("Valores Reais")
plt.title("Matriz de Confusão")
plt.show()

# Gerar Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
plt.xlabel("Falsos Positivos")
plt.ylabel("Verdadeiros Positivos")
plt.title("Curva ROC")
plt.legend()
plt.show()

# Importância das Variáveis
importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=features, palette="Blues")
plt.xlabel("Importância")
plt.title("Importância das Variáveis para a Previsão")
plt.show()
