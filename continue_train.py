import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Carregar o arquivo que você gerou
df = pd.read_csv('dataset_libras_final.csv')

# 2. Separar as coordenadas (X) do rótulo (y)
# Assumindo que a última coluna é o rótulo
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 3. Converter letras/nomes em números (Ex: 'A' -> 0, 'B' -> 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Dividir: 80% para treino e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 5. Criar a Rede Neural (MLP)
# hidden_layer_sizes=(128, 64) cria duas camadas internas na rede
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)

print("Iniciando o treinamento...")
model.fit(X_train, y_train)

# 6. Avaliar a precisão
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Treinamento concluído! Precisão no teste: {accuracy * 100:.2f}%")

# 7. Salvar o modelo e o encoder para usar na webcam depois
joblib.dump(model, 'modelo_libras.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Modelo salvo como 'modelo_libras.pkl'")