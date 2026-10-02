print("Estoy ejecutando este archivo")
import pandas as pd
datos = pd.read_csv("39._Enfermedades_crónicas_en_el_municipio_de_Bucaramanga_20260814 (1).csv",sep=";")
datos.columns = datos.columns.str.strip("'\"")
print("LIMPIEZA REALIZADA")
print(datos.shape) 
print(datos["Diabetes"].value_counts(dropna=False)) 
print(datos["Hipertensión"].value_counts(dropna=False)) 
print(datos["Diabetes"].unique()) 
print(datos["Hipertensión"].unique()) 
datos["COEXISTENCIA_DM_HTA"] = ((datos["Diabetes"] == 1) & (datos["Hipertensión"] == 1)).astype(int) 
print(datos["COEXISTENCIA_DM_HTA"].value_counts()) 
print(datos.groupby("CICLO_DE_VIDA")["COEXISTENCIA_DM_HTA"].sum()) 
print(datos.columns[4])
print(datos["Sexo"].value_counts(dropna=False))
print(datos["GRUPO_ ETAREO"].value_counts(dropna=False))
print(datos.groupby("GRUPO_ ETAREO")["COEXISTENCIA_DM_HTA"].agg(["count","sum","mean"]))
print(datos.groupby("CICLO_DE_VIDA")["COEXISTENCIA_DM_HTA"].agg(["count","sum","mean"]))
print(datos.groupby("Sexo")["COEXISTENCIA_DM_HTA"].agg(["count","sum","mean"]))
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LogisticRegression 
from sklearn.calibration import CalibratedClassifierCV 
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report

X = datos[["CICLO_DE_VIDA", "Sexo"]]
y = datos["COEXISTENCIA_DM_HTA"]

print(X.head())
print(y.head())

#Separar los datos para entrenar y evaluar el modelo

X = datos[["CICLO_DE_VIDA","Sexo"]]
y = datos["COEXISTENCIA_DM_HTA"] 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
print("Datos para entrenamiento:", len(X_train))
print("Datos para prueba:", len(X_test))

#Variables categóricas 

variables_categoricas = ["CICLO_DE_VIDA","Sexo"]

#Transformar las variables categoricas 

preprocesamiento = ColumnTransformer(transformers=[("categoricas",OneHotEncoder(handle_unknown="ignore"), variables_categoricas)])

#Crear un modelo

modelo = Pipeline(steps=[("preprocesamiento", preprocesamiento),("clasificador",LogisticRegression(max_iter=1000, class_weight= "balanced"))])

print("Modelo preparado correctamente")

#Entrenar modelo con los datos de entrenamiento 

modelo.fit(X_train,y_train)
print("Modelo entrenado correctamente")

modelo_calibrado = CalibratedClassifierCV(
    estimator = modelo, 
    method="sigmoid", 
    cv=5
    ) 
modelo_calibrado.fit(X_train, y_train)
print("Modelo calibrado correctamente")

#Entrenar modelo con los datos de prueba
precision = modelo.score(X_test, y_test)
print("Precisión del modelo:", precision)

#Evaluación del modelo 
predicciones = modelo.predict(X_test)

print("Exactitud (accuracy):",accuracy_score(y_test,predicciones))
print("Precision:",precision_score(y_test,predicciones))
print("Sensibilidad:",recall_score(y_test,predicciones))
print("F1:",f1_score(y_test,predicciones))

probabilidades_test = modelo.predict_proba(X_test)[:,1]
print("AUC:",roc_auc_score(y_test,probabilidades_test)) 
print("Matriz de confusión:")
print("Reporte de clasificacion:")
print(classification_report(y_test,predicciones))

#Obtener la probabilidad estimada de coexistencia 
probabilidades = modelo.predict_proba(X_test)[:, 1]

print("Primeras 10 probabilidades:")
print(probabilidades[:10])

# FUNCIÓN PARA PREDECIR LA COEXISTENCIA DE UNA PERSONA

ciclos_vida = {"Primera infancia (0 a 6 años)": "PRIMERA INFANCIA",
               "Infancia (7 a 11 años)": "INFANCIA",
                "Adolescencia (12 a 17 años)": "ADOLESCENCIA",
                "Jóvenes (18 a 28 años)": "JOVENES",
                "Adultez (29 a 59)": "ADULTEZ",
                "Persona mayor (60 años y más)": "PERSONA MAYOR"} 
def predecir_coexistencia(sexo, ciclo_de_vida):
    persona = pd.DataFrame({
        "CICLO_DE_VIDA": [ciclo_de_vida],
        "Sexo": [sexo]
    })

    probabilidad = modelo_calibrado.predict_proba(persona)[0, 1]

    return probabilidad


# EJEMPLO DE PREDICCIÓN

probabilidad = predecir_coexistencia("FEMENINO", "PERSONA MAYOR")

print("Probabilidad estimada de coexistencia:", probabilidad)

print("Porcentaje estimado:", round(probabilidad * 100, 2), "%")
print("\nPROBABILIDADES SEGÚN CICLO DE VIDA Y SEXO")

for ciclo in [
    "PRIMERA INFANCIA",
    "INFANCIA",
    "ADOLESCENCIA",
    "JOVENES",
    "ADULTEZ",
    "PERSONA MAYOR"
]:
    for sexo in ["FEMENINO", "MASCULINO"]:
        prob = predecir_coexistencia(sexo, ciclo)
        print(sexo, "-", ciclo, ":", round(prob * 100, 2), "%")

#Clasificar la probabilidad estimada 

def clasificar_probabilidad(probabilidad):
    porcentaje = probabilidad * 100

    if porcentaje >= 20:
        nivel = "ALTO"
        prioridad = "ALTA"

    elif porcentaje > 15:
        nivel = "MEDIO-ALTO"
        prioridad = "ALTA"

    elif porcentaje >= 10:
        nivel = "MEDIO"
        prioridad = "MEDIA"

    else: 
        nivel = "BAJO"
        prioridad = "BAJA"

    return porcentaje, nivel, prioridad 

#Ejemplo para una persona mayor
porcentaje, nivel, prioridad = clasificar_probabilidad(probabilidad)

print("Porcentaje estimado:", round(porcentaje, 2), "%")
print("Nivel de coexistencia:", nivel)
print("Prioridad:", prioridad)

import joblib
joblib.dump(modelo_calibrado,"modelo_coesistencia.pkl")

print("Modelo guardado correctamente")

    
