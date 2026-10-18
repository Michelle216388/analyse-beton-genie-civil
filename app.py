import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Analyse du béton | Génie civil", page_icon="🏗️", layout="wide")

COLUMN_ALIASES = {
    "cement": "Ciment (kg/m³)",
    "cement_component": "Ciment (kg/m³)",
    "blast furnace slag": "Laitier de haut fourneau (kg/m³)",
    "slag": "Laitier de haut fourneau (kg/m³)",
    "fly ash": "Cendres volantes (kg/m³)",
    "water": "Eau (kg/m³)",
    "superplasticizer": "Superplastifiant (kg/m³)",
    "coarse aggregate": "Granulats grossiers (kg/m³)",
    "fine aggregate": "Granulats fins (kg/m³)",
    "age": "Âge du béton (jours)",
    "concrete compressive strength": "Résistance à la compression (MPa)",
    "strength": "Résistance à la compression (MPa)",
}

def normalize_columns(df):
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower().replace("_", " ").replace("-", " ")
        renamed[col] = COLUMN_ALIASES.get(key, str(col).strip())
    df = df.rename(columns=renamed)
    # Some versions use the original UCI column headings.
    replacements = {
        "Cement (component 1)(kg in a m^3 mixture)": "Ciment (kg/m³)",
        "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": "Laitier de haut fourneau (kg/m³)",
        "Fly Ash (component 3)(kg in a m^3 mixture)": "Cendres volantes (kg/m³)",
        "Water  (component 4)(kg in a m^3 mixture)": "Eau (kg/m³)",
        "Superplasticizer (component 5)(kg in a m^3 mixture)": "Superplastifiant (kg/m³)",
        "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": "Granulats grossiers (kg/m³)",
        "Fine Aggregate (component 7)(kg in a m^3 mixture)": "Granulats fins (kg/m³)",
        "Age (day)": "Âge du béton (jours)",
        "Concrete compressive strength(MPa, megapascals)": "Résistance à la compression (MPa)",
    }
    return df.rename(columns=replacements)

@st.cache_data(show_spinner=False)
def load_online_data():
    """Charge le jeu de données public officiel depuis UCI."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=165)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    df = pd.concat([features, targets], axis=1)
    if df.empty or len(df.columns) < 5:
        raise ValueError("Le jeu de données récupéré ne ressemble pas au jeu attendu.")
    return normalize_columns(df), "UCI Machine Learning Repository — Concrete Compressive Strength (ID 165)"

st.title("🏗️ Analyse des données de résistance du béton")
st.caption("Application d'analyse de données en génie civil — jeu de données public sur le béton")

with st.sidebar:
    st.header("Source des données")
    st.write("L'application essaie de charger automatiquement un jeu de données public en ligne.")
    uploaded = st.file_uploader("Ou importer un fichier CSV", type=["csv"])
    st.divider()
    st.caption("Sujet : analyse statistique et estimation de la résistance à la compression du béton.")

try:
    if uploaded is not None:
        df = normalize_columns(pd.read_csv(uploaded))
        source_label = "Fichier CSV importé par l'utilisateur"
    else:
        with st.spinner("Récupération des données publiques en ligne..."):
            df, source_url = load_online_data()
        source_label = f"Données en ligne : {source_url}"
except Exception as exc:
    st.error("Le chargement automatique des données a échoué.")
    st.warning("Tu peux quand même utiliser l'application en important un fichier CSV dans la barre latérale.")
    st.code(str(exc))
    st.stop()

# Ensure numeric columns and clean the dataset
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
df = df.drop_duplicates()
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

target_candidates = [c for c in df.columns if "résistance" in c.lower() or "strength" in c.lower()]
target = target_candidates[0] if target_candidates else (numeric_cols[-1] if numeric_cols else None)
if target is None or len(numeric_cols) < 2:
    st.error("Le fichier doit contenir au moins deux colonnes numériques, dont la résistance du béton.")
    st.write("Colonnes détectées :", list(df.columns))
    st.stop()

# Data overview
st.success(f"Données chargées. Source : {source_label}")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Observations", f"{len(df):,}".replace(",", " "))
k2.metric("Variables numériques", len(numeric_cols))
k3.metric("Valeurs manquantes", int(df.isna().sum().sum()))
k4.metric("Doublons restants", int(df.duplicated().sum()))

tab_data, tab_stats, tab_graphs, tab_corr, tab_model = st.tabs([
    "📋 Données", "📊 Statistiques", "📈 Graphiques", "🔗 Corrélations", "🤖 Prédiction"
])

with tab_data:
    st.subheader("Exploration du jeu de données")
    st.write("Les données représentent les constituants du béton, son âge et sa résistance à la compression.")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.write("**Filtrer les observations**")
        filter_col = st.selectbox("Variable de filtre", ["Aucun"] + numeric_cols)
        filtered = df.copy()
        if filter_col != "Aucun":
            low = float(df[filter_col].min())
            high = float(df[filter_col].max())
            if low == high:
                st.info("Cette variable ne varie pas.")
            else:
                selected_range = st.slider("Intervalle", min_value=low, max_value=high, value=(low, high))
                filtered = filtered[filtered[filter_col].between(selected_range[0], selected_range[1])]
        st.metric("Lignes affichées", len(filtered))
    with c2:
        st.dataframe(filtered, use_container_width=True, height=390)
        st.download_button(
            "⬇️ Télécharger les données affichées (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8-sig"),
            file_name="donnees_beton_analyse.csv",
            mime="text/csv",
        )
    st.subheader("Aperçu du nettoyage")
    st.write("Les lignes entièrement vides et les doublons ont été retirés. Les cellules non numériques ont été converties en valeurs manquantes.")
    st.dataframe(pd.DataFrame({
        "Variable": df.columns,
        "Type": [str(df[c].dtype) for c in df.columns],
        "Valeurs manquantes": [int(df[c].isna().sum()) for c in df.columns],
        "Valeurs distinctes": [int(df[c].nunique(dropna=True)) for c in df.columns],
    }), use_container_width=True)

with tab_stats:
    st.subheader("Statistiques descriptives")
    st.dataframe(df[numeric_cols].describe().T.rename(columns={
        "count": "Effectif", "mean": "Moyenne", "std": "Écart-type",
        "min": "Minimum", "25%": "Q1", "50%": "Médiane", "75%": "Q3", "max": "Maximum"
    }).round(3), use_container_width=True)
    st.subheader(f"Indicateurs de la variable cible : {target}")
    series = df[target].dropna()
    a, b, c, d = st.columns(4)
    a.metric("Moyenne", f"{series.mean():.2f}")
    b.metric("Médiane", f"{series.median():.2f}")
    c.metric("Écart-type", f"{series.std():.2f}")
    d.metric("Étendue", f"{series.max() - series.min():.2f}")
    st.info("Ces indicateurs décrivent l'échantillon étudié. Ils ne garantissent pas les mêmes valeurs pour tous les bétons ou chantiers.")

with tab_graphs:
    st.subheader("Visualisations interactives")
    left, right = st.columns(2)
    with left:
        hist_var = st.selectbox("Variable à distribuer", numeric_cols, index=numeric_cols.index(target))
        fig = px.histogram(df, x=hist_var, nbins=30, marginal="box",
                           title=f"Distribution de {hist_var}")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        x_options = [c for c in numeric_cols if c != target]
        x_var = st.selectbox("Variable à comparer à la résistance", x_options, index=0)
        fig2 = px.scatter(df, x=x_var, y=target, trendline="ols",
                          title=f"{x_var} et {target}",
                          labels={x_var: x_var, target: target},
                          hover_data=[c for c in numeric_cols if c not in [x_var, target]][:3])
        st.plotly_chart(fig2, use_container_width=True)
    if "Âge du béton (jours)" in df.columns:
        age_summary = df.groupby("Âge du béton (jours)", as_index=False)[target].mean()
        fig3 = px.line(age_summary, x="Âge du béton (jours)", y=target, markers=True,
                       title="Résistance moyenne selon l'âge du béton")
        st.plotly_chart(fig3, use_container_width=True)
    st.caption("Les droites de tendance décrivent les relations observées dans les données ; elles ne prouvent pas à elles seules une causalité.")

with tab_corr:
    st.subheader("Matrice de corrélation")
    corr = df[numeric_cols].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title="Corrélations entre variables numériques")
    st.plotly_chart(fig, use_container_width=True)
    corr_target = corr[target].drop(labels=[target]).sort_values(key=lambda s: s.abs(), ascending=False)
    st.write(f"**Corrélation avec {target}**")
    st.dataframe(corr_target.rename("Coefficient de corrélation").to_frame().round(3), use_container_width=True)
    st.caption("Une corrélation proche de +1 indique une association linéaire positive, proche de -1 une association négative, et proche de 0 une association linéaire faible.")

with tab_model:
    st.subheader("Estimation de la résistance à la compression")
    st.write("Un modèle Random Forest est entraîné sur une partie des données et évalué sur des observations mises de côté.")
    feature_cols = [c for c in numeric_cols if c != target]
    model_df = df[feature_cols + [target]].dropna()
    if len(model_df) < 30:
        st.warning("Il faut au moins 30 lignes complètes pour tenter une évaluation raisonnable du modèle.")
    else:
        test_size = st.slider("Part des données réservée au test", min_value=0.15, max_value=0.40, value=0.20, step=0.05)
        X = model_df[feature_cols]
        y = model_df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model = RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE (MPa)", f"{mean_absolute_error(y_test, predictions):.2f}")
        m2.metric("RMSE (MPa)", f"{np.sqrt(mean_squared_error(y_test, predictions)):.2f}")
        m3.metric("R²", f"{r2_score(y_test, predictions):.3f}")
        result_df = pd.DataFrame({"Résistance réelle (MPa)": y_test, "Résistance estimée (MPa)": predictions})
        fig = px.scatter(result_df, x="Résistance réelle (MPa)", y="Résistance estimée (MPa)",
                         title="Valeurs réelles et estimées")
        low = float(min(result_df.min()))
        high = float(max(result_df.max()))
        fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line=dict(dash="dash"))
        st.plotly_chart(fig, use_container_width=True)
        importance = pd.DataFrame({"Variable": feature_cols, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
        st.plotly_chart(px.bar(importance, x="Importance", y="Variable", orientation="h",
                               title="Importance relative des variables"), use_container_width=True)
        st.write("**Tester une composition**")
        user_values = {}
        input_cols = st.columns(3)
        for i, col in enumerate(feature_cols):
            with input_cols[i % 3]:
                user_values[col] = st.number_input(
                    col, min_value=0.0, value=float(model_df[col].median()), step=1.0, key=f"input_{i}"
                )
        if st.button("Estimer la résistance", type="primary"):
            estimate = model.predict(pd.DataFrame([user_values], columns=feature_cols))[0]
            st.success(f"Résistance estimée : **{estimate:.2f} MPa**")
            st.caption("Cette estimation est indicative, dépend du jeu de données et ne remplace pas les essais normalisés ni la validation d'un ingénieur.")

st.divider()
st.caption("Projet pédagogique — Analyse de données publiques. Vérifier la source, les unités et les limites avant toute décision technique.")
