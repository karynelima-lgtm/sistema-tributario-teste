import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ---------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------

st.set_page_config(
    page_title="Protótipo de Inteligência Tributária",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# Geração de dados sintéticos
# ---------------------------------------------------------

@st.cache_data
def gerar_dados_sinteticos(quantidade=80, semente=42):
    """
    Gera dados fictícios para fins educacionais.
    Nenhum registro representa uma pessoa ou empresa real.
    """

    rng = np.random.default_rng(semente)

    tributos = ["IPTU", "ISS", "ITBI", "Taxa"]
    setores = ["Centro", "Taquaralto", "Aureny", "Plano Diretor"]
    faixas_renda = ["Baixa", "Média", "Alta"]

    dados = pd.DataFrame({
        "id_anonimo": [
            f"REG-{numero:04d}" for numero in range(1, quantidade + 1)
        ],
        "tributo": rng.choice(tributos, quantidade),
        "setor": rng.choice(setores, quantidade),
        "faixa_renda": rng.choice(
            faixas_renda,
            quantidade,
            p=[0.40, 0.40, 0.20]
        ),
        "valor_divida": rng.uniform(300, 15000, quantidade).round(2),
        "meses_atraso": rng.integers(1, 49, quantidade),
        "pagamentos_anteriores": rng.integers(0, 13, quantidade),
        "protegido_iptusocial": rng.choice(
            [0, 1],
            quantidade,
            p=[0.90, 0.10]
        )
    })

    # Criação de uma variável auxiliar apenas para gerar
    # um resultado fictício e controlado.
    influencia = (
        0.08 * dados["meses_atraso"]
        - 0.12 * dados["pagamentos_anteriores"]
        + 0.00004 * dados["valor_divida"]
    )

    probabilidade = 1 / (1 + np.exp(-influencia + 1.8))

    dados["inadimplente"] = rng.binomial(1, probabilidade)

    return dados


# ---------------------------------------------------------
# Treinamento do modelo
# ---------------------------------------------------------

@st.cache_resource
def treinar_modelo(dados):
    """
    Treina um modelo simples de regressão logística.
    Registros protegidos são retirados do treinamento
    para fins demonstrativos de governança.
    """

    dados_treinamento = dados[
        dados["protegido_iptusocial"] == 0
    ].copy()

    colunas_entrada = [
        "tributo",
        "setor",
        "faixa_renda",
        "valor_divida",
        "meses_atraso",
        "pagamentos_anteriores"
    ]

    X = dados_treinamento[colunas_entrada]
    y = dados_treinamento["inadimplente"]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    colunas_categoricas = [
        "tributo",
        "setor",
        "faixa_renda"
    ]

    colunas_numericas = [
        "valor_divida",
        "meses_atraso",
        "pagamentos_anteriores"
    ]

    pre_processamento = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore"),
                colunas_categoricas
            ),
            (
                "numericas",
                StandardScaler(),
                colunas_numericas
            )
        ]
    )

    modelo = Pipeline(
        steps=[
            ("pre_processamento", pre_processamento),
            (
                "classificador",
                LogisticRegression(max_iter=1000)
            )
        ]
    )

    modelo.fit(X_treino, y_treino)

    previsoes = modelo.predict(X_teste)

    metricas = {
        "acuracia": accuracy_score(y_teste, previsoes),
        "precisao": precision_score(
            y_teste,
            previsoes,
            zero_division=0
        ),
        "recall": recall_score(
            y_teste,
            previsoes,
            zero_division=0
        ),
        "f1": f1_score(
            y_teste,
            previsoes,
            zero_division=0
        ),
        "matriz_confusao": confusion_matrix(
            y_teste,
            previsoes
        )
    }

    return modelo, metricas


# ---------------------------------------------------------
# Cálculo de scores didáticos
# ---------------------------------------------------------

def calcular_scores(dados, modelo):
    """
    Gera risco de inadimplência e score didático de recuperabilidade.
    """

    colunas_entrada = [
        "tributo",
        "setor",
        "faixa_renda",
        "valor_divida",
        "meses_atraso",
        "pagamentos_anteriores"
    ]

    dados_resultado = dados.copy()

    dados_resultado["risco_inadimplencia"] = (
        modelo.predict_proba(
            dados_resultado[colunas_entrada]
        )[:, 1] * 100
    ).round(2)

    # Componentes normalizados para um score didático.
    pagamento_normalizado = (
        dados_resultado["pagamentos_anteriores"] / 12
    ).clip(0, 1)

    atraso_normalizado = (
        1 - dados_resultado["meses_atraso"] / 48
    ).clip(0, 1)

    valor_normalizado = (
        1 - dados_resultado["valor_divida"] / 15000
    ).clip(0, 1)

    dados_resultado["recuperabilidade"] = (
        (
            0.50 * pagamento_normalizado
            + 0.30 * atraso_normalizado
            + 0.20 * valor_normalizado
        ) * 100
    ).round(2)

    # Score pedagógico para ordenar a análise.
    # Ele não é uma decisão administrativa.
    dados_resultado["score_prioridade"] = (
        (
            dados_resultado["risco_inadimplencia"] * 0.50
            + dados_resultado["recuperabilidade"] * 0.50
        )
    ).round(2)

    dados_resultado["faixa_risco"] = pd.cut(
        dados_resultado["risco_inadimplencia"],
        bins=[-1, 33, 66, 100],
        labels=["Baixo", "Médio", "Alto"]
    )

    return dados_resultado


# ---------------------------------------------------------
# Construção da aplicação
# ---------------------------------------------------------

st.title("📊 Protótipo de Inteligência Tributária")

st.warning(
    "Este sistema utiliza dados sintéticos. "
    "Os resultados são apenas educacionais e não representam "
    "contribuintes reais nem decisões de cobrança."
)

dados = gerar_dados_sinteticos()
modelo, metricas = treinar_modelo(dados)
resultados = calcular_scores(dados, modelo)


# ---------------------------------------------------------
# Menu lateral
# ---------------------------------------------------------

st.sidebar.header("Filtros")

setores_selecionados = st.sidebar.multiselect(
    "Setor",
    options=sorted(resultados["setor"].unique()),
    default=sorted(resultados["setor"].unique())
)

tributos_selecionados = st.sidebar.multiselect(
    "Tributo",
    options=sorted(resultados["tributo"].unique()),
    default=sorted(resultados["tributo"].unique())
)

faixas_selecionadas = st.sidebar.multiselect(
    "Faixa de risco",
    options=["Baixo", "Médio", "Alto"],
    default=["Baixo", "Médio", "Alto"]
)

mostrar_protegidos = st.sidebar.checkbox(
    "Mostrar registros protegidos",
    value=False
)


# ---------------------------------------------------------
# Aplicação dos filtros
# ---------------------------------------------------------

filtrados = resultados[
    resultados["setor"].isin(setores_selecionados)
    & resultados["tributo"].isin(tributos_selecionados)
    & resultados["faixa_risco"].astype(str).isin(faixas_selecionadas)
].copy()

if not mostrar_protegidos:
    filtrados = filtrados[
        filtrados["protegido_iptusocial"] == 0
    ].copy()


# ---------------------------------------------------------
# Indicadores
# ---------------------------------------------------------

coluna1, coluna2, coluna3, coluna4 = st.columns(4)

coluna1.metric(
    "Registros totais",
    len(resultados)
)

coluna2.metric(
    "Registros exibidos",
    len(filtrados)
)

coluna3.metric(
    "Alto risco",
    int((filtrados["faixa_risco"] == "Alto").sum())
)

coluna4.metric(
    "Valor total da dívida",
    f"R$ {filtrados['valor_divida'].sum():,.2f}"
)


# ---------------------------------------------------------
# Abas principais
# ---------------------------------------------------------

aba1, aba2, aba3 = st.tabs(
    [
        "📋 Priorização",
        "📈 Métricas do modelo",
        "ℹ️ Explicação"
    ]
)


with aba1:
    st.subheader("Registros priorizados para análise humana")

    st.info(
        "A ordem abaixo é apenas uma priorização didática. "
        "Ela não inicia cobrança e não representa uma decisão automática."
    )

    tabela = filtrados.sort_values(
        by="score_prioridade",
        ascending=False
    ).copy()

    tabela["protegido_iptusocial"] = tabela[
        "protegido_iptusocial"
    ].map({
        0: "Não",
        1: "Sim"
    })

    colunas_exibicao = [
        "id_anonimo",
        "tributo",
        "setor",
        "faixa_renda",
        "valor_divida",
        "meses_atraso",
        "risco_inadimplencia",
        "recuperabilidade",
        "score_prioridade",
        "faixa_risco",
        "protegido_iptusocial"
    ]

    st.dataframe(
        tabela[colunas_exibicao],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Distribuição por faixa de risco")

    distribuicao = (
        filtrados["faixa_risco"]
        .value_counts()
        .rename_axis("faixa_risco")
        .reset_index(name="quantidade")
    )

    st.bar_chart(
        distribuicao.set_index("faixa_risco")
    )


with aba2:
    st.subheader("Métricas do modelo")

    metrica1, metrica2, metrica3, metrica4 = st.columns(4)

    metrica1.metric(
        "Acurácia",
        f"{metricas['acuracia']:.2%}"
    )

    metrica2.metric(
        "Precisão",
        f"{metricas['precisao']:.2%}"
    )

    metrica3.metric(
        "Recall",
        f"{metricas['recall']:.2%}"
    )

    metrica4.metric(
        "F1-score",
        f"{metricas['f1']:.2%}"
    )

    st.write("Matriz de confusão:")

    matriz = pd.DataFrame(
        metricas["matriz_confusao"],
        index=["Real: 0", "Real: 1"],
        columns=["Predito: 0", "Predito: 1"]
    )

    st.dataframe(
        matriz,
        use_container_width=False
    )

    st.caption(
        "Essas métricas são calculadas sobre uma base sintética "
        "e não devem ser interpretadas como desempenho real."
    )


with aba3:
    st.subheader("Como o sistema produz os resultados")

    st.markdown(
        """
        **Variáveis utilizadas pelo modelo:**

        - tipo de tributo;
        - setor;
        - faixa de renda agregada;
        - valor da dívida;
        - quantidade de meses em atraso;
        - quantidade de pagamentos anteriores.

        **Modelo utilizado:**

        Foi utilizada uma Regressão Logística, escolhida neste protótipo
        por ser simples, rápida e relativamente interpretável.

        **Risco de inadimplência:**

        Representa a probabilidade estimada pelo modelo de o registro
        pertencer à classe de inadimplência.

        **Recuperabilidade:**

        É um cálculo didático baseado em pagamentos anteriores,
        tempo de atraso e valor da dívida.

        **Score de prioridade:**

        É a combinação entre risco de inadimplência e recuperabilidade.
        Ele serve apenas para ordenar registros para análise humana.

        **Registros protegidos:**

        Registros marcados como protegidos pelo IPTU Social não aparecem
        na priorização padrão do protótipo.

        **Limitações:**

        - os dados são fictícios;
        - os critérios de recuperabilidade são ilustrativos;
        - não há integração com a Prefeitura;
        - o modelo não deve ser usado em produção;
        - o sistema não identifica fraude;
        - o sistema não toma decisão de cobrança.
        """
    )