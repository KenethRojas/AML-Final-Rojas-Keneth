# Modelo de Propensión para la Adquisición de un Seguro Vehicular (Advanced ML)

Proyecto de **clasificación binaria** para predecir la probabilidad de que un cliente adquiera un **seguro vehicular**, usando un dataset tabular anonimizado (30,000 registros, 10 variables predictoras + ID + target). El objetivo es mejorar la eficiencia de campañas comerciales priorizando a los clientes con mayor propensión.

---

## 🧠 Problema y Objetivo

**Problema:** Una empresa de seguros desea anticipar qué clientes tienen mayor probabilidad de adquirir un seguro vehicular para optimizar sus campañas comerciales y mejorar la conversión.

**Objetivo:** Entrenar un modelo predictivo (clasificación binaria) que estime la **propensión de compra** y permita:
- Priorizar clientes con mayor probabilidad o leads (ej. top deciles).
- Reducir costo por contacto.
- Mejorar conversión enfocando esfuerzos en perfiles con mayor intención.

---

## 📦 Dataset

- **Tamaño:** 30,000 filas × 12 columnas  
- **Target:** `Flag_Vehicular` (0 = no compra, 1 = compra)  
- **Desbalance:** ~94% clase 0 vs ~6% clase 1 (ratio ~15.7:1)  
- **Variables:** `Variable1` a `Variable10` (anonimizadas), `cliente` como identificador técnico.

> Nota: El dataset fue provisto internamente por la empresa aseguradora como parte del caso de estudio. Por motivos de confidencialidad, la información se entrega anonimizada: no contiene datos personales identificables (por ejemplo, nombres, documentos, teléfonos, direcciones) y las variables han sido enmascaradas con nombres genéricos (Variable1 a Variable10). Asimismo, el identificador cliente funciona únicamente como un ID técnico para trazabilidad y validaciones, sin permitir la identificación real de un individuo. Esta anonimización asegura el cumplimiento de buenas prácticas de privacidad y permite realizar el modelamiento sin exponer información sensible del negocio.

---

## 🧰 Metodología (resumen)

1. **EDA**: Distribución del target, correlaciones, análisis de outliers.
2. **Preprocesamiento**:
   - No hubo missing values.
   - Tratamiento de outliers con winsorización (1%–99%) en variables continuas.
   - Escalado con `RobustScaler`.
   - Split estratificado train/val/test (70/15/15).
3. **Modelos**:
   - **Deep Learning (PyTorch)**: MLP densa (64 → 32 → 16) con BatchNorm + ReLU + Dropout.
   - **Baselines (sklearn)**: Logistic Regression y Random Forest.
4. **Evaluación**:
   - Métricas de clasificación (Accuracy / Precision / Recall / F1 + matriz de confusión).
   - Comparación contra baselines.

---

## ✅ Resultados y métricas principales

- El dataset está **altamente desbalanceado**, por lo que **Accuracy puede ser engañosa**.
- En la comparación de modelos, los baselines (especialmente Random Forest) obtuvieron mejor desempeño global que la red neuronal en la corrida final.
- La matriz de confusión muestra que el reto principal está en **capturar adecuadamente la clase minoritaria (compradores)** sin disparar falsos positivos.

### Métrica principal usada (clasificación)
- **Accuracy** (y reporte de clasificación con Precision/Recall/F1)

### Comparación de modelos (Test)
| Modelo | Métrica (Accuracy) |
|-------|---------------------|
| Random Forest | **0.9416** |
| Logistic Regression | **0.9402** |
| Deep Learning (MLP) | **0.7044** |

> Recomendación: para una evaluación más justa en desbalance, priorizar **Recall/F1 de la clase 1**, además de **PR-AUC/ROC-AUC** y ajuste de umbral.

---

## 🎯 Conclusiones y Valor para la Toma de Decisiones

- El proyecto demuestra un pipeline completo de **modelado de propensión** (end-to-end).
- Para este dataset, **Random Forest** y **Logistic Regression** alcanzan mejor desempeño global que el DL en accuracy.
- A nivel negocio, el output del modelo permite:
  - Identificar clientes con mayor probabilidad de compra.
  - Priorizar esfuerzos comerciales (contacto, ofertas, seguimiento).
  - Mejorar el ROI de campañas al reducir el gasto en segmentos con baja propensión.

---

## 📁 Estructura del repositorio
* `notebooks/` → Notebook principal del proyecto
* `src/` → Código auxiliar (opcional)
* `data/` → Solo instrucciones del dataset (no subir datos grandes)
* `results/` → Resultados y métricas
* `figures/` → Gráficos generados
* `report/` → Reporte final (PDF o Markdown)

---

## 🚀 Cómo ejecutar

### Opción A: Google Colab (recomendado)
1. Sube el notebook a Colab.
2. Monta tu Google Drive.
3. Coloca el dataset en la ruta esperada.
4. Ejecuta todas las celdas en orden.

### Opción B: Local (Jupyter)
1. Crea un entorno virtual.
2. Instala dependencias.
3. Ejecuta el notebook.

---

## 🔧 Dependencias

Principales librerías usadas:
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `torch` (PyTorch)
- (opcional) `shap`

---

## 📌 Próximas mejoras

- Manejo explícito del desbalance:
- `class_weight` / `pos_weight`,
- oversampling (SMOTE) o undersampling con cuidado de leakage.
- Ajuste de **umbral de decisión** según capacidad comercial.
- Métricas adicionales: **PR-AUC, ROC-AUC**, curva Precision-Recall, calibración.
- Interpretabilidad:
- SHAP con el pipeline correcto (y sample representativo),
- importancia por permutación en baselines.

---

## 👤 Autor

**Keneth Anderson Rojas Cadillo**  
Capstone Project – Advanced Machine Learning

---

## 📚 Referencias (base)

- Scikit-learn documentation (modelos y métricas).
- PyTorch documentation (arquitectura y entrenamiento).
- SHAP documentation (interpretabilidad).

---

## 📄 Licencia

Uso académico / educativo.
Este proyecto está bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.

