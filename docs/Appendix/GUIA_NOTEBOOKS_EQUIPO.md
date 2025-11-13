# Guía de Notebooks - Epic 4: Clustering de Wallets

**Para:** Equipo de Tesis
**Fecha:** 26 de Octubre, 2025
**Autor:** Txelu Sanchez

---

## 📋 Resumen Ejecutivo

Esta carpeta contiene **4 notebooks de Jupyter** que documentan el análisis completo de clustering de wallets del Epic 4 de mi tesis. Cada notebook se puede ejecutar de forma independiente si tienes los archivos de datos necesarios.

**Total de notebooks:** 4
**Total de archivos de datos necesarios:** 11 (mínimo) - 18 (completo)
**Formato:** Jupyter Notebook (.ipynb)
**Tiempo total de ejecución:** ~15-20 minutos (todos)

---

## 📚 Los 4 Notebooks Explicados

### 1️⃣ Story_4.3_Wallet_Clustering_Analysis.ipynb

**¿Qué hace?**
Aplica algoritmos de clustering (HDBSCAN y K-Means) para identificar grupos de wallets con comportamientos similares.

**¿Por qué es importante?**
Es la base del análisis. Sin clustering, no podemos identificar arquetipos de wallets ni patrones de comportamiento.

**Objetivo principal:**
Responder a la pregunta: *"¿Existen grupos distintos de wallets con estrategias diferentes?"*

**Resultado clave:**
- Identifica 13 clusters + 48% de wallets "únicos" (noise)
- Silhouette score: 0.4078 (mejor resultado)
- Validación cruzada entre HDBSCAN y K-Means

**Duración:** ~5 minutos
**Secciones:** 12 pasos desde configuración hasta validación
**Archivos que genera:** 8 archivos (CSVs, JSONs, PNGs)

---

### 2️⃣ Story_4.4_Cluster_Interpretation.ipynb

**¿Qué hace?**
Interpreta los clusters identificados creando "personas" (arquetipos narrativos) y analizando qué hace único a cada grupo.

**¿Por qué es importante?**
Los números de cluster (0, 1, 2...) no significan nada por sí solos. Este notebook los convierte en insights accionables con nombres descriptivos y características claras.

**Objetivo principal:**
Responder a: *"¿Qué diferencia a cada grupo de wallets?"* y *"¿Qué estrategias emplean?"*

**Resultado clave:**
- 14 personas de cluster con descripciones detalladas
- Identifica wallets representativos de cada grupo
- Descubre que 48% de wallets tienen estrategias únicas (finding importante)
- 90-100% de overlap entre algoritmos (validación fuerte)

**Duración:** ~3 minutos
**Secciones:** 12 pasos desde carga hasta exportación
**Archivos que genera:** 7 archivos (profiles, personas, insights)

---

### 3️⃣ Story_4.5_Comprehensive_Evaluation.ipynb

**¿Qué hace?**
Valida estadísticamente todo el análisis anterior usando tests de hipótesis y comparaciones algorítmicas.

**¿Por qué es importante?**
Proporciona rigor estadístico. Demuestra que los clusters no son producto del azar sino patrones reales con significancia estadística.

**Objetivo principal:**
Responder a: *"¿Son los clusters estadísticamente significativos?"* y *"¿Son robustos los resultados?"*

**Resultado clave:**
- Todos los métricas muestran p < 0.05 (significativo)
- ARI > 0.3 (acuerdo moderado-fuerte entre algoritmos)
- Effect sizes de medio a grande
- Valida las 4 hipótesis planteadas

**Duración:** ~5 minutos
**Secciones:** 9 pasos desde métricas hasta síntesis
**Archivos que genera:** 5+ archivos (profiles, visualizaciones, reporte)

---

### 4️⃣ Epic_4_Research_Presentation.ipynb

**¿Qué hace?**
Presenta todo el Epic 4 en formato de defensa académica de 10-15 minutos, sintetizando hallazgos de los 3 notebooks anteriores.

**¿Por qué es importante?**
Es la historia completa contada de forma coherente para el comité académico. Combina metodología, resultados y conclusiones en una narrativa clara.

**Objetivo principal:**
Comunicar: *"¿Qué descubrimos sobre el comportamiento de wallets smart money y por qué importa?"*

**Resultado clave:**
- 3 hallazgos contra-intuitivos principales:
  1. 48% de heterogeneidad (diversidad es la norma)
  2. Portfolios concentrados ganan (HHI > 7,500)
  3. Trading pasivo domina (1-2 trades/mes)
- Las 4 hipótesis validadas con evidencia estadística
- Recomendaciones para investigadores, traders y desarrolladores

**Duración:** 10-15 minutos (presentación)
**Secciones:** 21 partes estructuradas
**Archivos que genera:** Ninguno (solo presenta resultados existentes)

---

## 📁 Archivos Necesarios por Notebook

### ✅ Archivos Mínimos Requeridos (11 archivos)

**Para que TODOS los notebooks funcionen, necesitas estos archivos en Google Drive:**

#### 1. Dataset Base (1 archivo)
```
outputs/features/
└── wallet_features_cleaned_20251025_121221.csv (912 KB)
```

#### 2. Resultados de Clustering (4 archivos)
```
outputs/clustering/
├── wallet_features_with_clusters_optimized_20251025_172729.csv
├── wallet_features_with_clusters_final_20251025_172855.csv
├── cluster_profiles_optimized_20251025_172729.csv
└── cluster_profiles_final_20251025_172855.csv
```

#### 3. Interpretación de Clusters (3 archivos)
```
outputs/cluster_interpretation/
├── cluster_personas_20251025_195003.json (19 KB)
├── cluster_insights_20251025_195003.json (8 KB)
└── representative_wallets_20251025_195003.json (5.6 KB)
```

#### 4. Visualizaciones (3 archivos)
```
outputs/clustering/visualizations/
├── tsne_optimized_20251025_172729.png (1.3 MB)
├── silhouette_final_20251025_172855.png
└── cluster_sizes_optimized_20251025_172729.png
```

**Total mínimo:** ~3-4 MB

---

### 🔍 Qué Notebook Necesita Qué

| Notebook | Archivos Requeridos | Archivos Opcionales |
|----------|---------------------|---------------------|
| **Story 4.3** | `wallet_features_cleaned_*.csv` | Ninguno |
| **Story 4.4** | `wallet_features_with_clusters_optimized_*.csv`<br>`wallet_features_with_clusters_final_*.csv` | Ninguno |
| **Story 4.5** | `wallet_features_with_clusters_optimized_*.csv`<br>`wallet_features_with_clusters_final_*.csv` | `cluster_personas_*.json` |
| **Presentation** | TODOS los 11 archivos mínimos | Ninguno |

---

## 📦 Paquete Completo Recomendado (18 archivos)

Para máxima reproducibilidad, comparte también:

```
outputs/cluster_interpretation/
├── cluster_profiles_detailed_20251025_195003.csv (5.6 KB)
├── hdbscan_kmeans_comparison_20251025_195003.csv (304 B)
├── cluster_overlap_analysis_20251025_195003.csv (396 B)
└── feature_validation_report_20251025_195003.txt (319 B)

outputs/clustering/
├── clustering_metadata_optimized_20251025_172729.json
└── clustering_metadata_final_20251025_172855.json

outputs/clustering/visualizations/
└── (Todas las imágenes PNG adicionales)
```

**Total completo:** ~5-6 MB

---

## 🚀 Cómo Ejecutar los Notebooks

### Opción 1: Google Colab (Recomendado para el equipo)

1. **Sube los notebooks a Google Drive**
2. **Crea una carpeta `outputs/` con la estructura:**
   ```
   Mi Drive/
   └── Epic4_Notebooks/
       ├── Story_4.3_Wallet_Clustering_Analysis.ipynb
       ├── Story_4.4_Cluster_Interpretation.ipynb
       ├── Story_4.5_Comprehensive_Evaluation.ipynb
       ├── Epic_4_Research_Presentation.ipynb
       └── outputs/
           ├── features/
           │   └── wallet_features_cleaned_20251025_121221.csv
           ├── clustering/
           │   ├── (archivos CSV y JSON)
           │   └── visualizations/
           │       └── (archivos PNG)
           └── cluster_interpretation/
               └── (archivos JSON y CSV)
   ```

3. **Abre cada notebook con Google Colab**
   - Click derecho → Abrir con → Google Colaboratory

4. **Monta Google Drive en la primera celda:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   # Ajusta las rutas
   BASE_DIR = Path("/content/drive/MyDrive/Epic4_Notebooks")
   ```

5. **Ejecuta todas las celdas**
   - Runtime → Run all

### Opción 2: Jupyter Local

1. **Instala dependencias:**
   ```bash
   pip install jupyter numpy pandas matplotlib seaborn scikit-learn scipy pillow
   ```

2. **Lanza Jupyter:**
   ```bash
   jupyter notebook
   ```

3. **Abre el notebook y ejecuta todas las celdas**

---

## ⚠️ Problemas Comunes y Soluciones

### Error: "FileNotFoundError: No such file"
**Causa:** Los archivos de datos no están en la ruta esperada
**Solución:** Verifica que la estructura de carpetas `outputs/` esté correctamente creada

### Error: "ModuleNotFoundError: No module named 'hdbscan'"
**Causa:** Librería faltante
**Solución:** `pip install hdbscan` (solo necesario para Story 4.3)

### Las imágenes no se muestran
**Causa:** Archivos PNG no están en `outputs/clustering/visualizations/`
**Solución:** Asegúrate de copiar la carpeta `visualizations/` completa

### Los archivos tienen nombres diferentes
**Causa:** Cada ejecución genera timestamps únicos
**Solución:** Los notebooks buscan el archivo más reciente automáticamente con `glob("*pattern*.csv")`

---

## 📊 Estructura de la Carpeta Compartida de Drive

**Organización recomendada:**

```
📁 Epic4_Notebooks_Compartido/
│
├── 📓 NOTEBOOKS (4 archivos)
│   ├── Story_4.3_Wallet_Clustering_Analysis.ipynb
│   ├── Story_4.4_Cluster_Interpretation.ipynb
│   ├── Story_4.5_Comprehensive_Evaluation.ipynb
│   └── Epic_4_Research_Presentation.ipynb
│
├── 📁 outputs/
│   ├── 📁 features/
│   │   └── wallet_features_cleaned_20251025_121221.csv
│   │
│   ├── 📁 clustering/
│   │   ├── wallet_features_with_clusters_optimized_20251025_172729.csv
│   │   ├── wallet_features_with_clusters_final_20251025_172855.csv
│   │   ├── cluster_profiles_optimized_20251025_172729.csv
│   │   ├── cluster_profiles_final_20251025_172855.csv
│   │   ├── clustering_metadata_optimized_20251025_172729.json
│   │   ├── clustering_metadata_final_20251025_172855.json
│   │   └── 📁 visualizations/
│   │       ├── tsne_optimized_20251025_172729.png
│   │       ├── silhouette_final_20251025_172855.png
│   │       └── cluster_sizes_optimized_20251025_172729.png
│   │
│   └── 📁 cluster_interpretation/
│       ├── cluster_personas_20251025_195003.json
│       ├── cluster_insights_20251025_195003.json
│       ├── representative_wallets_20251025_195003.json
│       ├── cluster_profiles_detailed_20251025_195003.csv
│       ├── hdbscan_kmeans_comparison_20251025_195003.csv
│       ├── cluster_overlap_analysis_20251025_195003.csv
│       └── feature_validation_report_20251025_195003.txt
│
├── 📄 GUIA_NOTEBOOKS_EQUIPO.md (este documento)
├── 📄 PRESENTATION_GUIDE.md (guía de presentación en inglés)
└── 📄 README.md (documentación completa del proyecto)
```

---

## 🎯 Orden Recomendado de Lectura/Ejecución

### Para entender el análisis completo:

1. **Primero: Story 4.3** (Clustering Analysis)
   - Entender qué son los clusters y cómo se identificaron
   - Ver las visualizaciones t-SNE y silhouette

2. **Segundo: Story 4.4** (Cluster Interpretation)
   - Conocer las personas de cada cluster
   - Entender qué hace único a cada grupo

3. **Tercero: Story 4.5** (Comprehensive Evaluation)
   - Ver la validación estadística
   - Confirmar que los resultados son significativos

4. **Cuarto: Epic 4 Presentation**
   - Ver la historia completa sintetizada
   - Entender las conclusiones principales

### Para presentación rápida al equipo:

1. **Solo Epic 4 Presentation** (10-15 minutos)
   - Tiene todo lo importante resumido
   - Perfecto para overview del proyecto

2. **Luego profundizar en los otros 3 según interés**

---

## 📝 Checklist Antes de Compartir

Verifica que tienes:

- [ ] Los 4 notebooks (.ipynb)
- [ ] Carpeta `outputs/features/` con el CSV de features
- [ ] Carpeta `outputs/clustering/` con CSVs de clustering
- [ ] Carpeta `outputs/clustering/visualizations/` con PNGs
- [ ] Carpeta `outputs/cluster_interpretation/` con JSONs
- [ ] Este documento (GUIA_NOTEBOOKS_EQUIPO.md)
- [ ] README.md del proyecto (opcional pero útil)
- [ ] PRESENTATION_GUIDE.md (si van a presentar)

---

## 🔗 Documentación Adicional

Si quieren profundizar más, estos documentos tienen análisis detallados:

- **STORY_4.3_CLUSTERING_COMPLETE.md** - Análisis completo de clustering (7,500+ palabras)
- **STORY_4.4_CLUSTER_INTERPRETATION_COMPLETE.md** - Interpretación detallada (15,000+ palabras)
- **STORY_4.5_EVALUATION_COMPLETE.md** - Evaluación comprehensiva (11,000+ palabras)
- **PRESENTATION_GUIDE.md** - Guía para presentación académica (20 KB)

---

## 💡 Consejos para el Equipo

### Al revisar los notebooks:

1. **Lean primero los markdown cells** - Explican qué hace cada paso
2. **No necesitan ejecutar todos los notebooks** - Si solo quieren ver resultados, pueden leer las celdas sin ejecutar
3. **Las visualizaciones están guardadas** - No hace falta regenerarlas
4. **Los timestamps en nombres de archivo son automáticos** - Los notebooks buscan el archivo más reciente

### Al ejecutar:

1. **Ejecución secuencial recomendada** - Story 4.3 → 4.4 → 4.5
2. **Story 4.3 toma más tiempo** - ~5 minutos por el t-SNE
3. **Los demás son rápidos** - 2-3 minutos cada uno
4. **Presentation no genera archivos nuevos** - Solo muestra resultados

### Para presentaciones:

1. **Use el Presentation notebook** para defender el trabajo
2. **Consulte PRESENTATION_GUIDE.md** para talking points
3. **Practique el timing** - Apunte a 12-14 minutos
4. **Prepare respuestas de Q&A** - La guía tiene preguntas anticipadas

---

## 📞 Soporte

**Si tienen problemas:**

1. **Revisen la sección "Problemas Comunes"** arriba
2. **Verifiquen que tienen todos los archivos** con el checklist
3. **Comprueben las rutas de archivos** en el código
4. **Contacten a Txelu** si algo no funciona

**Archivos de contacto:**
- Email: [Tu email]
- GitHub: [Tu repositorio si aplica]

---

## ✅ Resumen Final

**Lo que necesitas compartir:**
- 4 notebooks (.ipynb)
- 11 archivos de datos (mínimo) o 18 (completo)
- Esta guía (GUIA_NOTEBOOKS_EQUIPO.md)

**Lo que el equipo puede hacer:**
- Ejecutar los notebooks en Colab o Jupyter local
- Ver los resultados del análisis
- Entender la metodología completa
- Usar el Presentation notebook para su propia presentación

**Tiempo total de setup:**
- Subir archivos a Drive: ~5 minutos
- Configurar estructura de carpetas: ~3 minutos
- Ejecutar primer notebook: ~5 minutos
- **Total: ~15 minutos para estar operativo**

---

**Creado:** 26 de Octubre, 2025
**Última actualización:** 26 de Octubre, 2025
**Versión:** 1.0
**Estado:** ✅ Listo para compartir con el equipo

---

**¡Buena suerte con la revisión en equipo! 🚀**
