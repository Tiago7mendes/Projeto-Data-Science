import pandas as pd


colunas = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species'] # Adicionei uma parte nas colunas no dataset para ficar mais legivel

dados = pd.read_csv('iris.data', header=None, names=colunas) # Lendo o dataset iris.data


class AnaliseIrisSemAgg:
    def __init__(self, file_path):
        self.file_path = file_path
        self.dados = None

    def Load(self):
        try:
            self.dados = pd.read_csv(self.file_path, header=None, names=colunas)
            self.dados.columns = self.dados.columns.str.strip()  # Tirar os espaços
            print("Dataset carregado e colunas renomeadas com sucesso.")
        except Exception as e:
            print(f"Erro ao carregar o dataset: {e}") # Estava dando muito erro, pois o data set nao estav sendo carregado

    def Estatisticas(self):
        if self.dados is not None:
            especies = self.dados['Species'].unique()  # Seperar as espeicies
            print("\nEstatísticas calculadas para cada espécie:\n")
            
            for especie in especies:
                print(f"Espécie: {especie}")
                dados_especie = self.dados[self.dados['Species'] == especie]  # Coletar os dados da especie selecionada
                
                for coluna in ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']:
                    media = dados_especie[coluna].mean()
                    maximo = dados_especie[coluna].max()
                    minimo = dados_especie[coluna].min()
                    
                    print(f"  {coluna}:")
                    print(f"    Média: {media:.2f}")
                    print(f"    Máximo: {maximo:.2f}")
                    print(f"    Mínimo: {minimo:.2f}")
                print()
        else:
            print("Dados não carregados.")

if __name__ == "__main__":
    analise = AnaliseIrisSemAgg('iris.data')  
    analise.Load() 
    analise.Estatisticas()
