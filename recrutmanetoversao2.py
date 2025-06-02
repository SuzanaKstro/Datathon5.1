# %% [markdown]
# # Ferramneto para Recrutamento Decision - Datathon Pós Tech Fiap
# 
# Este notebook contém todo o processo de leitura, tratamento, análise e modelagem
# utilizando dados da empresa Decision visando otimizar o processo de recrutamento.

# %% [markdown]
# ## 1. Importações e Caminhos dos Arquivos

# %%
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# Caminhos dos arquivos JSON (ajuste se necessário)
caminho_vagas = r'C:\Users\bruno\OneDrive\Área de Trabalho\DESAFIO FASE 05\vagas.json'
caminho_prospects = r'C:\Users\bruno\OneDrive\Área de Trabalho\DESAFIO FASE 05\prospects.json'
caminho_applicants = r'C:\Users\bruno\OneDrive\Área de Trabalho\DESAFIO FASE 05\applicants.json'

# %% [markdown]
# ## 2. Leitura e Normalização de `vagas.json`

# %%
with open(caminho_vagas, encoding='utf-8') as f:
    raw_vagas = json.load(f)

lista_vagas = []
for id_vaga, dados in raw_vagas.items():
    dados['id_vaga'] = id_vaga
    lista_vagas.append(dados)

df_vagas = pd.json_normalize(lista_vagas)
print(f'Vagas carregadas: {df_vagas.shape}')

# %% [markdown]
# ## 3. Leitura e Normalização de `prospects.json` (com estrutura mista)

# %%
with open(caminho_prospects, encoding='utf-8') as f:
    raw_prospects = json.load(f)

lista_prospects = []
for id_vaga, conteudo in raw_prospects.items():
    if isinstance(conteudo, list):
        for candidato in conteudo:
            if isinstance(candidato, dict):
                candidato['id_vaga'] = id_vaga
                lista_prospects.append(candidato)
    elif isinstance(conteudo, dict) and 'prospects' in conteudo:
        for candidato in conteudo['prospects']:
            if isinstance(candidato, dict):
                candidato['id_vaga'] = id_vaga
                lista_prospects.append(candidato)

df_prospeccoes = pd.json_normalize(lista_prospects)
print(f'Prospecções carregadas: {df_prospeccoes.shape}')

# %% [markdown]
# ## 4. Leitura e Normalização de `applicants.json`

# %%
with open(caminho_applicants, encoding='utf-8') as f:
    raw_candidatos = json.load(f)

lista_candidatos = []
for id_candidato, dados in raw_candidatos.items():
    dados['id_candidato'] = id_candidato
    lista_candidatos.append(dados)

df_candidatos = pd.json_normalize(lista_candidatos)
print(f'Candidatos carregados: {df_candidatos.shape}')

# %% [markdown]
# ## 5. Visualização Inicial

# %%
print("\nVagas:")
print(df_vagas.head())

print("\nProspecções:")
print(df_prospeccoes.head())

print("\nCandidatos:")
print(df_candidatos.head())


# %%
# %% [markdown]
# ## 6. Análise Exploratória (EDA)
# 
# ### 6.1 Estrutura dos DataFrames
# %%
print("Vagas:")
df_vagas.info()
# %%
print("\nProspecções:")
df_prospeccoes.info()
# %%
print("\nCandidatos:")
df_candidatos.info()
# %% [markdown]
# ### 6.2 Verificação dos nomes das colunas
# %%
print("Colunas disponíveis em df_vagas:")
print(df_vagas.columns.tolist())
# %%
print("Colunas disponíveis em df_prospeccoes:")
print(df_prospeccoes.columns.tolist())
# %%
print("Colunas disponíveis em df_candidatos:")
print(df_candidatos.columns.tolist())
# %%
# %% [markdown]
### 6.3.1 Quantidade de registros únicos de vagas
print(f"Total de vagas: {df_vagas['id_vaga'].nunique():,.0f}".replace(',', '.'))
# %%
# %% [markdown]
## 6.3.2 Quantidade de registros únicos de canditados
print(f"Total de candidatos: {df_candidatos['id_candidato'].nunique():,.0f}".replace(',', '.'))
# %%
# %% [markdown]
## 6.3.3 Quantidade de registros únicos de prospecções
print(f"Total de prospecções: {df_prospeccoes.shape[0]:,.0f}".replace(',', '.'))
# %%
# %% [markdown]
# ## Análise item 6.3:
# O número de prospecções é superior ao número de candidatos, pois um candidato pode se inscrever em várias vagas.
# Isso é comum em processos seletivos, onde candidatos podem se candidatar a múltiplas oportunidades de emprego.
# %%
# %% [markdown]
# ### 6.4 Verificar valores nulos
print("Valores nulos em df_vagas:")
print(df_vagas.isnull().sum().sort_values(ascending=False))
# %%
print("\nValores nulos em df_prospeccoes:")
print(df_prospeccoes.isnull().sum().sort_values(ascending=False))
# %%
print("\nValores nulos em df_candidatos:")
print(df_candidatos.isnull().sum().sort_values(ascending=False))
# %%
# %% [markdown]
# ### 6.5 Distribuição de status nas prospecções
# %%
if 'situacao_candidado' in df_prospeccoes.columns:
    print("Distribuição da situação dos candidatos:")
    print(df_prospeccoes['situacao_candidado'].value_counts(dropna=False))
else:
    print("Coluna 'situacao' não encontrada. Verifique o nome correto.")
# %% [markdown]
# Gráfico da distribuição da situação dos candidatos
if 'situacao_candidado' in df_prospeccoes.columns:
    situacao_counts = df_prospeccoes['situacao_candidado'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=situacao_counts.values, y=situacao_counts.index, palette="viridis")
    plt.title("Distribuição da Situação dos Candidatos")
    plt.xlabel("Quantidade")
    plt.ylabel("Situação")
    plt.tight_layout()
    plt.show()
else:
    print("Coluna 'situacao' não encontrada. Verifique o nome correto.")
    print("Coluna 'situacao' não encontrada. Verifique o nome correto.")
# %%
# Tabela com percentual por situação
percentual = (situacao_counts / situacao_counts.sum() * 100).round(2)
df_percentual = pd.DataFrame({
    'Situação': situacao_counts.index,
    'Quantidade': situacao_counts.values,
    'Percentual (%)': percentual.values
})
print("Tabela com percentuais da situação dos candidatos em relação ao total de candidaturas:")
print(df_percentual)
# %%
# %% [markdown]
# ### 6.6 Junção dos dados para modelagem

# %%
# Renomear a coluna 'codigo' para 'id_candidato' para fazer o merge corretamente
df_prospeccoes = df_prospeccoes.rename(columns={'codigo': 'id_candidato'})

# Juntar prospecções com vagas
df_completo = df_prospeccoes.merge(df_vagas, on='id_vaga', how='left')

# Juntar com candidatos
df_completo = df_completo.merge(df_candidatos, on='id_candidato', how='left')

# Verificar o resultado
print(f"Shape final do DataFrame consolidado: {df_completo.shape}")
print(df_completo[['id_vaga', 'id_candidato', 'situacao_candidado']].head())

# %%
print("Colunas disponíveis em df_completo:")
print(df_completo.columns.tolist())
# %%
print("Registros em df_prospeccoes:", df_prospeccoes.shape[0])
print("Registros após merge (df_completo):", df_completo.shape[0])

# %%
print("Vagas ausentes no merge:", df_completo['informacoes_basicas.titulo_vaga'].isnull().sum())
print("Candidatos ausentes no merge:", df_completo['infos_basicas.nome'].isnull().sum())

# %%
print(df_prospeccoes['id_candidato'].dtype)
print(df_candidatos['id_candidato'].dtype)

# %%
# %% [markdown]
# Conferindo de maneira aleatoria os dados do merge
df_completo[['id_candidato', 'id_vaga', 'situacao_candidado', 
             'perfil_vaga.nivel_ingles', 'formacao_e_idiomas.nivel_ingles', 
             'informacoes_profissionais.area_atuacao']].sample(5)

# %%
# %% [markdown]
# ### 7. Criação da variável alvo: `foi_contratado`

# %%
situacoes_contratado = ['Contratado pela Decision', 'Contratado como Hunting']

df_completo['foi_contratado'] = df_completo['situacao_candidado'].isin(situacoes_contratado).astype(int)

# Verificar distribuição
print("Distribuição da variável `foi_contratado`:")
print(df_completo['foi_contratado'].value_counts())

# Percentual
percentual = (df_completo['foi_contratado'].value_counts(normalize=True) * 100).round(2)
print("\nPercentual:")
print(percentual)

# %%
# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Contagem dos valores
contagem = df_completo['foi_contratado'].value_counts().sort_index()
labels = ['Não contratado', 'Contratado']

# Gráfico de barras
plt.figure(figsize=(8, 4))
sns.barplot(x=contagem.values, y=labels, palette="crest")
plt.title("Distribuição da Variável Alvo: Foi Contratado")
plt.xlabel("Quantidade")
plt.ylabel("Situação")
plt.tight_layout()
plt.show()

# %%
# %%
# Quantidade
contagem = df_completo['foi_contratado'].value_counts().sort_index()
# Percentual
percentual = df_completo['foi_contratado'].value_counts(normalize=True).sort_index() * 100

# Monta DataFrame
df_alvo = pd.DataFrame({
    'Classe': ['Não Contratado', 'Contratado'],
    'Quantidade': contagem.values,
    'Percentual (%)': percentual.round(2).values
})

print("\nTabela da variável alvo (`foi_contratado`):")
print(df_alvo)

# %%
# %% [markdown]
# ### 8.2 Tratamento de valores nulos nas features selecionadas
# %% [markdown]
# # 8. Seleção e Engenharia de Features
# 
# Nesta etapa, selecionamos colunas com potencial explicativo para a contratação, considerando:
# 
# - Relevância para o processo seletivo
# - Presença de dados (poucos valores nulos)
# - Facilidade de tratamento (sem estruturas aninhadas ou texto livre)

# %% [markdown]
# 8.1 Seleção inicial de colunas relevantes

colunas_modelo = [
    'perfil_vaga.estado',
    'perfil_vaga.cidade',
    'perfil_vaga.regiao',
    'perfil_vaga.nivel_academico',
    'perfil_vaga.nivel_ingles',
    'perfil_vaga.nivel_espanhol',
    'perfil_vaga.faixa_etaria',
    'perfil_vaga.horario_trabalho',
    'perfil_vaga.areas_atuacao',
    'perfil_vaga.vaga_especifica_para_pcd',
    'informacoes_basicas.tipo_contratacao',
    'informacoes_basicas.prioridade_vaga',
    'informacoes_profissionais.area_atuacao',
    'informacoes_profissionais.nivel_profissional',
    'formacao_e_idiomas.nivel_academico',
    'formacao_e_idiomas.nivel_ingles',
    'formacao_e_idiomas.nivel_espanhol',
    'formacao_e_idiomas.outro_idioma',
    'formacao_e_idiomas.instituicao_ensino_superior',
    'formacao_e_idiomas.ano_conclusao',
    'foi_contratado'  # variável alvo
]

# %%
# %% [markdown]
# ## 8.2 Tratamento de valores ausentes
# 
# - Colunas categóricas: preenchidas com "Desconhecido"
# - Colunas numéricas: preenchidas com a mediana da variável
# %%
# %%
# Criação do DataFrame para modelagem
df_modelo = df_completo[colunas_modelo].copy()

# Identifica colunas categóricas
colunas_categoricas = df_modelo.select_dtypes(include='object').columns
df_modelo[colunas_categoricas] = df_modelo[colunas_categoricas].fillna('Desconhecido')

# Identifica colunas numéricas e preenche com a mediana
colunas_numericas = df_modelo.select_dtypes(include=['int64', 'float64']).columns
for col in colunas_numericas:
    mediana = df_modelo[col].median()
    df_modelo[col] = df_modelo[col].fillna(mediana)

# Verifica se ainda restam nulos
print("Valores nulos restantes por coluna:")
print(df_modelo.isnull().sum().sort_values(ascending=False))


# %%
# %% [markdown]
# # 9. Pré-processamento para Modelagem
# 
# Nesta etapa, preparamos os dados para alimentar algoritmos de classificação:
# 
# - Separação entre features (X) e variável-alvo (y)
# - Codificação de variáveis categóricas via One-Hot Encoding


# %%
# %%
# Separa variável alvo
y = df_modelo['foi_contratado']
X = df_modelo.drop(columns='foi_contratado')

# %%
# %% [markdown]
# ## 9.1 Codificação de Variáveis Categóricas
# 
# Aplicamos One-Hot Encoding para transformar colunas categóricas em variáveis numéricas binárias.

# %%
# %%
# Codifica variáveis categóricas
X_encoded = pd.get_dummies(X, drop_first=True)

# Verifica o novo formato das variáveis
print(f"Formato final das features codificadas: {X_encoded.shape}")

# %%
# %% [markdown]
# # 10. Modelagem com Random Forest
# 
# Iniciamos com um modelo base de Random Forest para prever a variável `foi_contratado`.
# 
# Etapas:
# - Divisão dos dados em treino e teste
# - Treinamento do modelo
# - Avaliação da acurácia inicial

# %%
# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Divisão dos dados (80% treino / 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Criação e treinamento do modelo
modelo_rf = RandomForestClassifier(random_state=42)
modelo_rf.fit(X_train, y_train)

# %%
# %% [markdown]
# ## 10.1 Avaliação do modelo

# %%
# %%
# Predições
y_pred = modelo_rf.predict(X_test)

# Avaliação
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de Confusão
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# %%
# %% [markdown]
# # 11. Reajuste com Balanceamento de Classes
# 
# O modelo anterior teve desempenho ruim para a classe minoritária (`foi_contratado = 1`).
# 
# Agora ajustamos o `RandomForestClassifier` com o parâmetro `class_weight='balanced'`, para dar pesos maiores à classe menos representada.

# %%
# %%
# Modelo com pesos balanceados
modelo_rf_bal = RandomForestClassifier(random_state=42, class_weight='balanced')
modelo_rf_bal.fit(X_train, y_train)

# Predição
y_pred_bal = modelo_rf_bal.predict(X_test)

# Avaliação
print("Acurácia:", accuracy_score(y_test, y_pred_bal))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_bal))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_bal))

# %%
# %% [markdown]
# ## 11.1 Conclusão do Ajuste com Classes Balanceadas
# 
# Após aplicarmos o `class_weight='balanced'`, observamos:
# 
# - **Melhora significativa no recall da classe contratada (`1`)**, de 17% para **45%**.
# - **F1-score da classe 1 também melhorou**, indicando maior equilíbrio entre precisão e sensibilidade.
# - Como esperado, a **acurácia geral caiu** (de 94% para 87%), pois o modelo passou a errar mais na classe majoritária.
# 
# **Resumo:**
# 
# | Métrica      | Modelo Padrão | Modelo Balanceado |
# |--------------|----------------|--------------------|
# | Recall (1)   | 17%            | **45%**            |
# | F1-score (1) | 27%            | **28%**            |
# | Acurácia     | **94,9%**      | 87,3%              |
# 
# Essa troca mostra que o modelo está mais sensível a identificar candidatos contratados, o que é importante em contextos onde essa classe é o foco.

# %%
# %% [markdown]
# ## 12. Importância das Variáveis
# 
# Após o treinamento, avaliamos quais colunas mais influenciam a previsão de contratação (`foi_contratado = 1`).
# 
# Abaixo, listamos as 20 variáveis mais importantes segundo o modelo Random Forest.

# %%
# %%
# Verificar se modelo já foi treinado
importances = modelo_rf_bal.feature_importances_
features = X_train.columns

# DataFrame com importâncias
df_importancia = pd.DataFrame({
    'Feature': features,
    'Importância': importances
}).sort_values(by='Importância', ascending=False)

# Exibir top 20
top_n = 20
plt.figure(figsize=(10, 8))
sns.barplot(data=df_importancia.head(top_n), x='Importância', y='Feature', palette='viridis')
plt.title(f'Top {top_n} Variáveis mais Importantes')
plt.xlabel('Importância')
plt.ylabel('Variável')
plt.tight_layout()
plt.show()

# %%
# %% [markdown]
# ## 13. Avaliação com Curva ROC e AUC
# 
# A Curva ROC nos ajuda a visualizar o desempenho do modelo em separar as classes. A AUC resume essa performance.

# %%
# %%
from sklearn.metrics import roc_curve, roc_auc_score

# Probabilidades preditas
y_probs = modelo_rf_bal.predict_proba(X_test)[:, 1]  # Prob da classe 1

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc = roc_auc_score(y_test, y_probs)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("Taxa de Falsos Positivos (FPR)")
plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
plt.title("Curva ROC")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 14. Otimização do Modelo Random Forest com GridSearchCV (versão reduzida)
# 
# Nesta etapa, vamos buscar os melhores hiperparâmetros para o modelo RandomForestClassifier, 
# utilizando uma grade reduzida para agilizar o processo. O foco permanece em otimizar a AUC 
# e melhorar a identificação de candidatos contratados.

# %%
from sklearn.model_selection import GridSearchCV

# Grade reduzida de hiperparâmetros para testes mais rápidos
param_grid = {
    'n_estimators': [100],
    'max_depth': [None, 10],
    'max_features': ['sqrt']
}

# Configurando o GridSearchCV com validação cruzada simplificada
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=1
)

# Executando o treinamento com validação cruzada
grid_search.fit(X_train, y_train)

# %% [markdown]
# ### Resultados da busca reduzida

# %%
print("Melhores hiperparâmetros (versão reduzida):", grid_search.best_params_)
print("Melhor AUC (validação cruzada):", grid_search.best_score_)


# %%
# %% [markdown]
# ## 15. Avaliação Final com os Melhores Hiperparâmetros
# 
# Com os melhores parâmetros encontrados, treinamos novamente o modelo Random Forest e avaliamos seu desempenho 
# em termos de acurácia, métricas de classificação e curva ROC/AUC.

# %%
# Treinando modelo final com os melhores parâmetros
modelo_final = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',
    n_estimators=100,
    max_depth=10,
    max_features='sqrt'
)

modelo_final.fit(X_train, y_train)

# Predição
y_pred_final = modelo_final.predict(X_test)
y_proba_final = modelo_final.predict_proba(X_test)[:, 1]

# Avaliação
print("Acurácia:", accuracy_score(y_test, y_pred_final))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_final))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_final))

# %%
# %% [markdown]
# ### Curva ROC - Modelo Final com Parâmetros Otimizados
# 
# Abaixo está a curva ROC gerada a partir do modelo final treinado com os melhores hiperparâmetros
# encontrados no GridSearchCV. A área sob a curva (AUC) representa a capacidade do modelo de
# distinguir entre as classes (contratado vs não contratado).

# %%
fpr, tpr, _ = roc_curve(y_test, y_proba_final)
auc = roc_auc_score(y_test, y_proba_final)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Falso Positivo')
plt.ylabel('Verdadeiro Positivo')
plt.title('Curva ROC - Modelo Final')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# %% [markdown]
# ## 16. Modelo com XGBoost
# 
# Nesta etapa, aplicamos o algoritmo XGBoost para prever a contratação de candidatos. 
# O modelo será treinado com os mesmos dados e avaliado com as mesmas métricas usadas anteriormente.

# %%
from xgboost import XGBClassifier

# Criando o modelo com configuração básica (balanceamento automático das classes)
modelo_xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # compensando desbalanceamento
)

# Treinamento
modelo_xgb.fit(X_train, y_train)

# Predição
y_pred_xgb = modelo_xgb.predict(X_test)
y_proba_xgb = modelo_xgb.predict_proba(X_test)[:, 1]

# Avaliação
print("Acurácia:", accuracy_score(y_test, y_pred_xgb))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_xgb))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_xgb))

# %%
# %% [markdown]
# ### Curva ROC - Modelo XGBoost
# 
# A curva ROC abaixo mostra a performance do modelo XGBoost na separação das classes. 
# A AUC reflete a capacidade de distinguir entre candidatos contratados e não contratados.

# %%
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
auc_xgb = roc_auc_score(y_test, y_proba_xgb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, label=f'AUC = {auc_xgb:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Falso Positivo')
plt.ylabel('Verdadeiro Positivo')
plt.title('Curva ROC - Modelo XGBoost')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%

# %% [markdown]
# ## 18. Comparativo de Modelos
#
# Abaixo está a comparação entre os dois modelos testados: Random Forest (com hiperparâmetros otimizados via GridSearchCV) e XGBoost.
#
# | Métrica                  | Random Forest (com GridSearch) | XGBoost               |
# |--------------------------|-------------------------------|------------------------|
# | Acurácia                 | 0.872                          | 0.825                  |
# | Recall (Classe 1)        | 0.45                           | 0.60                   |
# | Precisão (Classe 1)      | 0.21                           | 0.18                   |
# | F1-Score (Classe 1)      | 0.28                           | 0.28                   |
# | AUC (Curva ROC)          | 0.767                          | 0.788                  |
# | Matriz de Confusão       | [[9118, 1037], [329, 268]]     | [[8516, 1639], [239, 358]] |
#
# **Recomendação final**  
# Caso o objetivo seja **identificar o maior número de candidatos que efetivamente serão contratados** (maior recall na classe 1), o **XGBoost** é mais indicado.  
# Se a prioridade for **acurácia geral** e **redução de falsos positivos**, o **Random Forest** apresenta melhor desempenho.
# Como buscamos identificar os candidatos que serão contratados vamos utilizar o modelo XGBoost


# %%
# %% [markdown]
# ## 19. Deploy Simulado: Inferência com Modelo XGBoost
# 
# Esta etapa simula a aplicação do modelo em produção, recebendo dados de um novo candidato e retornando a probabilidade de contratação.

# %%
def prever_contratacao(dados_dict, modelo, colunas_modelo):
    """
    Recebe um dicionário com os dados do candidato e retorna a predição do modelo treinado.
    """
    import pandas as pd
    import numpy as np

    # Cria o DataFrame e garante todas as colunas, na ordem correta
    df_input = pd.DataFrame([dados_dict])
    df_input = df_input.reindex(columns=colunas_modelo, fill_value=0)

    # Predição
    proba = modelo.predict_proba(df_input)[0][1]
    classe = int(proba >= 0.5)

    return classe, round(proba, 3)

# %%
# Exemplo de candidato (ajuste os campos com base no X_train.columns)
exemplo_candidato = {
    'formacao_e_idiomas.nivel_ingles': 3,
    'formacao_e_idiomas.nivel_espanhol': 1,
    'formacao_e_idiomas.nivel_academico': 4,
    'informacoes_profissionais.nivel_profissional': 2,
    'perfil_vaga.estado': 'SP',
    # Adicione mais campos se necessário...
}

# %%
# Executa a previsão com o modelo XGBoost
classe_predita, probabilidade = prever_contratacao(
    dados_dict=exemplo_candidato,
    modelo=modelo_xgb,
    colunas_modelo=X_train.columns
)

print(f"Classe prevista: {classe_predita} (0 = Não contratado, 1 = Contratado)")
print(f"Probabilidade de contratação: {probabilidade}")

# %%
import joblib

# Salva o modelo XGBoost treinado
joblib.dump(modelo_xgb, 'modelo_xgb.pkl')

# %%
joblib.dump(X_train.columns.tolist(), 'colunas_modelo.pkl')

# %%
# Exportar dados do X_train para CSV
X_train.to_csv('dados_treinamento.csv', index=False)

# %%
