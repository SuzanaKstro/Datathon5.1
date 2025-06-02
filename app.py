import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# Carrega o modelo treinado
modelo = joblib.load('modelo_xgb.pkl')

# Define os campos que o modelo espera e seus nomes amigáveis
campos = {
    'perfil_vaga.estado': 'Estado da vaga',
    'perfil_vaga.cidade': 'Cidade da vaga',
    'perfil_vaga.regiao': 'Região da vaga',
    'perfil_vaga.nivel_academico': 'Nível acadêmico da vaga',
    'perfil_vaga.nivel_ingles': 'Nível de inglês da vaga',
    'perfil_vaga.nivel_espanhol': 'Nível de espanhol da vaga',
    'perfil_vaga.faixa_etaria': 'Faixa etária da vaga',
    'perfil_vaga.horario_trabalho': 'Horário de trabalho',
    'perfil_vaga.areas_atuacao': 'Área de atuação da vaga',
    'perfil_vaga.vaga_especifica_para_pcd': 'Vaga específica para PCD?',
    'informacoes_basicas.tipo_contratacao': 'Tipo de contratação',
    'informacoes_basicas.prioridade_vaga': 'Prioridade da vaga',
    'informacoes_profissionais.area_atuacao': 'Área de atuação do candidato',
    'informacoes_profissionais.nivel_profissional': 'Nível profissional do candidato',
    'formacao_e_idiomas.nivel_academico': 'Nível acadêmico do candidato',
    'formacao_e_idiomas.nivel_ingles': 'Nível de inglês do candidato',
    'formacao_e_idiomas.nivel_espanhol': 'Nível de espanhol do candidato',
    'formacao_e_idiomas.outro_idioma': 'Outro idioma',
    'formacao_e_idiomas.instituicao_ensino_superior': 'Instituição de ensino superior',
    'formacao_e_idiomas.ano_conclusao': 'Ano de conclusão da formação'
}

# Título
st.title('🧠 Previsão de Contratação - Decision AI')

# Inputs do usuário
st.markdown("### Preencha os dados da vaga e do candidato")
entrada = {}

for campo, nome_amigavel in campos.items():
    if campo == 'perfil_vaga.vaga_especifica_para_pcd':
        valor = st.selectbox(nome_amigavel, ['Não', 'Sim'])
        entrada[campo] = True if valor == 'Sim' else False
    elif campo == 'formacao_e_idiomas.ano_conclusao':
        entrada[campo] = st.number_input(nome_amigavel, min_value=1950, max_value=2050, step=1)
    else:
        entrada[campo] = st.text_input(nome_amigavel)

# Botão para prever
if st.button('Fazer Previsão'):
    df_input = pd.DataFrame([entrada])

    # Codificação one-hot
    df_input = pd.get_dummies(df_input)

    # Garante que todas as colunas esperadas pelo modelo estejam presentes
    colunas_esperadas = modelo.get_booster().feature_names
    colunas_faltantes = [col for col in colunas_esperadas if col not in df_input.columns]

    # Cria colunas faltantes com valor 0
    df_faltantes = pd.DataFrame(0, index=[0], columns=colunas_faltantes)
    df_input = pd.concat([df_input, df_faltantes], axis=1)

    # Reordena as colunas conforme esperado pelo modelo
    df_input = df_input[colunas_esperadas]

    # Previsão
    proba = modelo.predict_proba(df_input)[0][1]
    classe = modelo.predict(df_input)[0]

    st.markdown(f"### Resultado:")
    st.write(f"**Classe prevista:** {'Contratado' if classe == 1 else 'Não contratado'}")
    st.write(f"**Probabilidade de contratação:** {proba:.2%}")
    st.balloons()  # Animação de balões para celebrar a previsão
    st.success("Previsão realizada com sucesso!")


