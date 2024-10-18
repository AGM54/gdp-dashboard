import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


data = pd.read_csv('risk_factors_cervical_cancer.csv')


data.replace('?', pd.NA, inplace=True)
data = data.apply(pd.to_numeric, errors='coerce')


st.title("Tablero de Control: Factores de Riesgo para Cáncer Cervical")


age_filter = st.slider('Filtrar por Edad', int(data['Age'].min()), int(data['Age'].max()), (20, 40))
smokes_filter = st.checkbox('Mostrar solo personas que fuman')

filtered_data = data[(data['Age'] >= age_filter[0]) & (data['Age'] <= age_filter[1])]
if smokes_filter:
    filtered_data = filtered_data[filtered_data['Smokes'] == 1]

# Visualización 1: Histograma de edades
st.subheader("Distribución de Edades")
fig_age = px.histogram(filtered_data, x='Age', title="Distribución de Edades")
st.plotly_chart(fig_age)

# Visualización 2: Dispersión Edad vs Número de parejas sexuales
st.subheader("Relación entre Edad y Número de Parejas Sexuales")
fig_sexual_partners = px.scatter(filtered_data, x='Age', y='Number of sexual partners', color='Dx:Cancer',
                                 title="Edad vs Número de Parejas Sexuales")
st.plotly_chart(fig_sexual_partners)

# Visualización 3: Distribución del uso de anticonceptivos
st.subheader("Uso de Anticonceptivos Hormonales")
fig_contraceptives = px.histogram(filtered_data, x='Hormonal Contraceptives', title="Uso de Anticonceptivos Hormonales")
st.plotly_chart(fig_contraceptives)

# Visualización 4: Relación entre Fumar y Diagnóstico de Cáncer
st.subheader("Relación entre Fumar y Diagnóstico de Cáncer")
fig_smokes_cancer = px.histogram(filtered_data, x='Smokes', color='Dx:Cancer', title="Relación entre Fumar y Cáncer")
st.plotly_chart(fig_smokes_cancer)


X = filtered_data[['Age', 'Number of sexual partners', 'Smokes', 'Hormonal Contraceptives']].fillna(0)
y = filtered_data['Dx:Cancer'].fillna(0)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelos de clasificación
model_choice = st.selectbox("Seleccione el modelo para predecir cáncer", ["Random Forest", "Regresión Logística", "K-Nearest Neighbors"])

# Visualización 5: Desempeño del modelo seleccionado
st.subheader(f"Resultados del modelo: {model_choice}")
if model_choice == "Random Forest":
    model = RandomForestClassifier()
elif model_choice == "Regresión Logística":
    model = LogisticRegression()
else:
    model = KNeighborsClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Matriz de confusión
st.subheader("Matriz de Confusión")
conf_matrix = confusion_matrix(y_test, y_pred)
st.write(conf_matrix)

# Reporte de clasificación
st.subheader("Reporte de Clasificación")
st.text(classification_report(y_test, y_pred, zero_division=1))

# Visualización 6: Importancia de características (solo para Random Forest)
if model_choice == "Random Forest":
    st.subheader("Importancia de Características")
    importances = pd.Series(model.feature_importances_, index=X.columns)
    fig_importances = px.bar(importances, title="Importancia de Características en el Modelo")
    st.plotly_chart(fig_importances)

# Comparación de modelos
st.subheader("Comparativa de Modelos")
# Aplicar 3 modelos para la comparativa
models = {"Random Forest": RandomForestClassifier(), "Regresión Logística": LogisticRegression(), "K-Nearest Neighbors": KNeighborsClassifier()}
comparison_data = []
for model_name, model_instance in models.items():
    model_instance.fit(X_train, y_train)
    y_pred_model = model_instance.predict(X_test)
    report = classification_report(y_test, y_pred_model, output_dict=True, zero_division=1)
    accuracy = report['accuracy']
    comparison_data.append([model_name, accuracy])

# Visualización 7: Tabla comparativa
comparison_df = pd.DataFrame(comparison_data, columns=['Modelo', 'Precisión'])
st.write(comparison_df)

# Visualización 8: Gráfica comparativa de precisión
st.subheader("Gráfica de Precisión de los Modelos")
fig_comparison = px.bar(comparison_df, x='Modelo', y='Precisión', title="Precisión Comparativa de los Modelos")
st.plotly_chart(fig_comparison)


st.subheader("Relación entre Edad y Fumar")
fig_linked = px.scatter(filtered_data, x='Age', y='Smokes', color='Dx:Cancer', title="Edad vs Fumar y Diagnóstico de Cáncer")
st.plotly_chart(fig_linked)
