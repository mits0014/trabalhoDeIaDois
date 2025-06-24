from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt
import clientdb as db
from sklearn.utils import shuffle


# 1. Carregar dataset
letter_recognition = fetch_ucirepo(id=59)
X = letter_recognition.data.features
y = letter_recognition.data.targets

print("Dados de entrada (X):")
print(X.head())

print("\nAlvo (y):")
print(y.head())

# 1. Embaralhar a base original
X, y = shuffle(X, y, random_state=42)  # Garante aleatoriedade

# 2. Dividir: 50% treino, 50% temporário (validação + teste)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=42
)

# 3. Dividir os 50% restantes igualmente: 25% validação e 25% teste
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)


# Função auxiliar para avaliação
def avaliar_modelo(nome, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    print(f"\n=== {nome.upper()} ===")
    print(f"Acurácia: {acc}")
    print(f"Precisão: {report['macro avg']['precision']}")
    print(f"Recall: {report['macro avg']['recall']}")
    print(f"F1-score: {report['macro avg']['f1-score']}")

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Classes Preditas')
    ax.set_ylabel('Classes Reais')
    ax.set_title(f'Matriz de Confusão - {nome}')
    plt.show()

def salva_resultados_modelo(nome, y_true, y_pred, parametros=None):
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()  # converte para lista se for salvar em JSON/DB
    report = classification_report(y_true, y_pred, output_dict=True)

    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1_score = report['macro avg']['f1-score']

    # Garante que parametros seja sempre um dicionário, mesmo se não for passado
    if parametros is None:
        parametros = {}

    # Chama sua função que salva no banco
    db.salvar_resultado_modelo(
        nome=nome,
        matriz_confusao=cm,
        acuracia=acc,
        precisao=precision,
        recall=recall,
        f1_score=f1_score,
        parametros=parametros,
        report=report
    )


# 3. KNN
clf_knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
clf_knn.fit(X_train, y_train)
y_pred_knn = clf_knn.predict(X_test)
# avaliar_modelo("KNN", y_test, y_pred_knn)
salva_resultados_modelo(
    nome="KNN",
    y_true=y_test,
    y_pred=y_pred_knn,
    parametros={
        "n_neighbors": clf_knn.n_neighbors,
        "weights": clf_knn.weights
    }
)
# Salvar resultados do KNN no banco

# 4. SVM
clf_svm = SVC(C=1.0, kernel='rbf')
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
# avaliar_modelo("SVM", y_test, y_pred_svm)
salva_resultados_modelo(
    nome="SVM",
    y_true=y_test,
    y_pred=y_pred_svm,
    parametros={
        "C": clf_svm.C,
        "kernel": clf_svm.kernel
    }
)
# Salvar resultados da SVM no banco

# 5. Árvore de Decisão
clf_ad = DecisionTreeClassifier(max_depth=None, min_samples_split=2)
clf_ad.fit(X_train, y_train)
y_pred_ad = clf_ad.predict(X_test)
# avaliar_modelo("Árvore de Decisão", y_test, y_pred_ad)
salva_resultados_modelo(
    nome="Árvore de Decisão",
    y_true=y_test,
    y_pred=y_pred_ad,
    parametros={
        "max_depth": clf_ad.max_depth,
        "min_samples_split": clf_ad.min_samples_split
    }
)
# Salvar resultados da Árvore de Decisão no banco

# 6. MLP
clf_mlp = MLPClassifier(
    hidden_layer_sizes=(200, 100,50,25),
    activation='relu',
    max_iter=30000,
    learning_rate='constant',
    learning_rate_init=0.001
)
clf_mlp.fit(X_train, y_train)
y_pred_mlp = clf_mlp.predict(X_test)
# avaliar_modelo("MLP", y_test, y_pred_mlp)
salva_resultados_modelo(
    nome="MLP",
    y_true=y_test,
    y_pred=y_pred_mlp,
    parametros={
        "hidden_layer_sizes": clf_mlp.hidden_layer_sizes,
        "activation": clf_mlp.activation,
        "max_iter": clf_mlp.max_iter,
        "learning_rate": clf_mlp.learning_rate,
        "learning_rate_init": clf_mlp.learning_rate_init
    }
)
# Salvar resultados da MLP no banco

# 7. Naive Bayes
clf_nb = GaussianNB()
clf_nb.fit(X_train, y_train)
y_pred_nb = clf_nb.predict(X_test)
# avaliar_modelo("Naive Bayes", y_test, y_pred_nb)
salva_resultados_modelo(
    nome="Naive Bayes",
    y_true=y_test,
    y_pred=y_pred_nb
)
# Salvar resultados do Naive Bayes no banco
