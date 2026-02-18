# Modelo de Propensión para la Adquisición de un Seguro Vehicular (Advanced ML)

Proyecto de **clasificación binaria** para predecir la probabilidad de que un cliente adquiera un **seguro vehicular**, usando un dataset tabular anonimizado (30,000 registros, 10 variables predictoras + ID + target). El objetivo es mejorar la eficiencia de campañas comerciales priorizando a los clientes con mayor propensión.

---

## 🧠 Contexto y objetivo de negocio

En seguros, contactar clientes de forma masiva incrementa costos y reduce ROI. Este proyecto construye un modelo que **estima propensión de compra** para:
- priorizar leads (ej. top deciles),
- reducir costo por contacto,
- mejorar conversión enfocando esfuerzos en perfiles con mayor intención.

---

## 📦 Dataset

- **Tamaño:** 30,000 filas × 12 columnas  
- **Target:** `Flag_Vehicular` (0 = no compra, 1 = compra)  
- **Desbalance:** ~94% clase 0 vs ~6% clase 1 (ratio ~15.7:1)  
- **Variables:** `Variable1` a `Variable10` (anonimizadas), `cliente` como identificador técnico.

> Nota: El dataset fue provisto internamente por la empresa aseguradora como parte del caso de estudio. Por motivos de confidencialidad, la información se entrega anonimizada: no contiene datos personales identificables (por ejemplo, nombres, documentos, teléfonos, direcciones) y las variables han sido enmascaradas con nombres genéricos (Variable1 a Variable10). Asimismo, el identificador cliente funciona únicamente como un ID técnico para trazabilidad y validaciones, sin permitir la identificación real de un individuo. Esta anonimización asegura el cumplimiento de buenas prácticas de privacidad y permite realizar el modelamiento sin exponer información sensible del negocio.

---

## 🧰 Metodología (resumen)

1. **EDA**: distribución del target, correlaciones, análisis de outliers.
2. **Preprocesamiento**:
   - no hubo missing values,
   - tratamiento de outliers con winsorización (1%–99%) en variables continuas,
   - escalado con `RobustScaler`,
   - split estratificado train/val/test (70/15/15).
3. **Modelos**:
   - **Deep Learning (PyTorch)**: MLP densa (64 → 32 → 16) con BatchNorm + ReLU + Dropout.
   - **Baselines (sklearn)**: Logistic Regression y Random Forest.
4. **Evaluación**:
   - métricas de clasificación (Accuracy / Precision / Recall / F1 + matriz de confusión),
   - comparación contra baselines.

---

## ✅ Resultados (alto nivel)

- El dataset está **altamente desbalanceado**, por lo que **Accuracy puede ser engañosa**.
- En la comparación de modelos, los baselines (especialmente Random Forest) obtuvieron mejor desempeño global que la red neuronal en la corrida final.
- La matriz de confusión muestra que el reto principal está en **capturar adecuadamente la clase minoritaria (compradores)** sin disparar falsos positivos.

> Recomendación: para una evaluación más justa en desbalance, priorizar **Recall/F1 de la clase 1**, además de **PR-AUC/ROC-AUC** y ajuste de umbral.

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
3. Coloca el dataset en la ruta esperada
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

