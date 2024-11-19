import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class Modelo:
    def __init__(self):
        self.dados = None
        self.modelo_svm = None
        self.modelo_lr = None
        self.X_treino = None
        self.X_teste = None
        self.y_treino = None
        self.y_teste = None

    def CarregarDados(self):
        try:
            self.dados = pd.read_csv("iris.data", header=None, names=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"])
            print("Dataset carregado com sucesso.")
            print(self.dados.head(151))  # Printando os dados
        except Exception as e:
            print(f"Erro ao carregar o dataset: {e}") # Estava dando muito erro, pois o data set nao estav sendo carregado

    def TratamentoDeDados(self):
        if self.dados is None:
            print("Erro: os dados ainda não foram carregados. Verifique se CarregarDados foi executado.")
            return
        
        print("Distribuição das espécies no dataset:")
        print(self.dados['Species'].value_counts())
        
        # variáveis independentes (X) e dependente (y)
        X = self.dados.iloc[:, :-1]
        y = self.dados['Species']
        
        self.X_treino, self.X_teste, self.y_treino, self.y_teste = train_test_split(X, y, test_size=0.3, random_state=42)
        print("Pré-processamento concluído.")

    def TreinarModelos(self):
        if self.X_treino is None or self.y_treino is None:
            print("Erro: os dados de treino não estão disponíveis.")
            return

        # modelo SVM
        self.modelo_svm = SVC()
        self.modelo_svm.fit(self.X_treino, self.y_treino)
        
        # modelo de Regressão Logística
        self.modelo_lr = LogisticRegression(max_iter=200)
        self.modelo_lr.fit(self.X_treino, self.y_treino)

        print("Modelos treinados com sucesso.")

    def TestarModelos(self):
        if self.X_teste is None or self.y_teste is None:
            print("Erro: os dados de teste não estão disponíveis.")
            return

        predicoes_svm = self.modelo_svm.predict(self.X_teste)
        acuracia_svm = accuracy_score(self.y_teste, predicoes_svm)
        print(f"Acurácia do modelo SVM: {acuracia_svm:.2f}")

        predicoes_lr = self.modelo_lr.predict(self.X_teste)
        acuracia_lr = accuracy_score(self.y_teste, predicoes_lr)
        print(f"Acurácia do modelo Logistic Regression: {acuracia_lr:.2f}")

    def ValidacaoCruzada(self):
        if self.dados is None:
            print("Erro: os dados não foram carregados.")
            return
        
        X = self.dados.iloc[:, :-1]
        y = self.dados['Species']

        scores_svm = cross_val_score(SVC(), X, y, cv=5)
        print(f"Acurácia média com validação cruzada (SVM): {scores_svm.mean():.2f} (+/- {scores_svm.std() * 2:.2f})")

        scores_lr = cross_val_score(LogisticRegression(max_iter=200), X, y, cv=5)
        print(f"Acurácia média com validação cruzada (Logistic Regression): {scores_lr.mean():.2f} (+/- {scores_lr.std() * 2:.2f})")

    def Executar(self):
        self.CarregarDados()
        self.TratamentoDeDados()
        self.TreinarModelos()
        self.TestarModelos()
        self.ValidacaoCruzada()

if __name__ == "__main__":
    modelo = Modelo()
    modelo.Executar()
