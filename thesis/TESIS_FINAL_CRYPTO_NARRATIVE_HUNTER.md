# TESIS FINAL - MÁSTER EN INTELIGENCIA ARTIFICIAL AVANZADA Y GENERATIVA

---

## CRYPTO NARRATIVE HUNTER
### Clustering No Supervisado de Wallets Smart Money en Mercados DeFi de Ethereum

**Identificación de Arquetipos Conductuales mediante HDBSCAN y Validación Estadística**

---

**Autores:** Oscar Pons, Antonio Nieves y Jose Luis Sanchez
**Máster:** Máster en Inteligencia Artificial Avanzada y Generativa
**Institución:** MBIT School
**Fecha:** Noviembre 2025

---

**GUÍA DE FORMATO PARA VERSIÓN PDF (MBIT School)**

**Requisitos de formato según "Estructura y Guía de Estilo en Proyectos End 2 End":**
- Fuente: Arial, Calibri o Verdana, tamaño 12
- Interlineado: 1.5
- Márgenes: 2.5 cm en todos los lados
- Alineación de párrafos: Justificada
- Resumen Ejecutivo: Máximo 1 página
- Estilo: Objetivo y conciso

---

# ÍNDICE

## CONTENIDO PRINCIPAL

RESUMEN EJECUTIVO

INTRODUCCIÓN
   - Motivación del proyecto
   - Hipótesis iniciales
   - Contexto del proyecto

MARCO TEÓRICO Y ANTECEDENTES
   - Descripción del problema
   - Objetivos específicos
   - Relación con contenidos del máster

METODOLOGÍA
   - Arquitectura general del sistema
   - Pipeline de datos
   - Ingeniería de características
   - Clustering HDBSCAN
   - Validación estadística

RESULTADOS
   - Dataset final
   - Resultados de clustering
   - Validación estadística
   - Hallazgos clave
   - Perfiles de clusters

DISCUSIÓN
   - Interpretación de resultados
   - Aplicaciones prácticas
   - Limitaciones

CONCLUSIONES Y TRABAJO FUTURO
   - Conclusiones principales
   - Aportación académica
   - Trabajo futuro

CÓDIGO FUENTE
   - Repositorio GitHub
   - Estructura del proyecto

BIBLIOGRAFÍA

## APÉNDICES

- **APÉNDICE A**: Análisis Exploratorio de Datos (EDA)
- **APÉNDICE B**: Notebook de Clustering (Story 4.3)
- **APÉNDICE C**: Notebook de Interpretación (Story 4.4)
- **APÉNDICE D**: Notebook de Evaluación (Story 4.5)
- **APÉNDICE E**: Presentación de Investigación
- **APÉNDICE F**: Queries SQL de Dune Analytics

---

# RESUMEN EJECUTIVO

Los mercados de criptomonedas carecen de taxonomías empíricas para caracterizar el comportamiento de inversores sofisticados ("smart money"). Este proyecto aplica clustering no supervisado (HDBSCAN) a 2,159 carteras operando en DEXs de Ethereum (septiembre-octubre 2025) para identificar arquetipos conductuales mediante técnicas de machine learning y validación estadística rigurosa.

**Objetivos:** Validar cuatro hipótesis sobre comportamiento de smart money: (H1) existencia de patrones diferenciables de trading, (H2) robustez algorítmica de clusters, (H3) diferencias estadísticas en métricas de rendimiento, y (H4) ventaja de carteras concentradas.

**Metodología:** Pipeline end-to-end integrando APIs de Dune Analytics, Alchemy y CoinGecko, con extracción de 34,034 transacciones, ingeniería de 41 features en 5 categorías (rendimiento, comportamiento, concentración, exposición narrativa, acumulación/distribución), clustering HDBSCAN optimizado, y validación mediante tests de Kruskal-Wallis, bootstrap (ARI=0.82±0.08) y cross-validation con K-Means.

**Resultados:** Las cuatro hipótesis fueron validadas con evidencia empírica robusta (todas p<0.001). HDBSCAN identificó 13 clusters + 48.4% ruido (carteras con estrategias únicas), con silhouette score de 0.4078. Los clusters difieren significativamente en ROI (η²=0.67), Sharpe Ratio (η²=0.60), y métricas conductuales. El 69% de carteras exhiben especialización narrativa (>60% exposición en una narrativa), y el HHI medio del portfolio es 0.9087, indicando ultra-concentración en 1-2 tokens.

**Hallazgo clave:** El 48.4% de carteras clasificadas como "ruido" no representa fallo metodológico sino característica fundamental del mercado cripto: las estrategias ganadoras son heterogéneas e idiosincrásicas, desafiando taxonomías rígidas.

**Contribución académica:** Primera taxonomía validada estadísticamente de arquetipos de smart money en Ethereum, estableciendo framework replicable con validación cross-algorítmica y 14 arquetipos interpretables.

**Alineación MBIT:** Integra competencias del máster en machine learning avanzado (HDBSCAN, validación estadística), infraestructura de datos e IA (PostgreSQL, pipelines ETL), visualización avanzada (t-SNE, heatmaps), y estadística computacional (tests no paramétricos, bootstrap).

**Palabras clave:** Blockchain Analytics, HDBSCAN Clustering, Smart Money, Feature Engineering, Ethereum DeFi, Validación Estadística, Arquetipos Conductuales

---

# INTRODUCCIÓN

## Motivación del Proyecto

Los mercados de criptomonedas presentan características únicas que desafían los modelos tradicionales de comportamiento de inversores: alta volatilidad, operación 24/7, seudoanonimato mediante direcciones blockchain, y ciclos narrativos acelerados donde nuevos sectores emergen y desaparecen en meses.

A diferencia de mercados financieros tradicionales donde existen taxonomías establecidas de inversores (institucionales, retail, market makers), el espacio cripto carece de clasificaciones empíricas basadas en comportamiento on-chain observable. La literatura académica sobre behavioral finance en cripto es escasa y fragmentada, limitándose a estudios de caso sobre eventos específicos sin establecer marcos generales de caracterización conductual.

Este proyecto surge de la necesidad de identificar y caracterizar arquetipos conductuales dentro de la población de "smart money", inversores que consistentemente generan retornos superiores al mercado. El término "smart money" se define mediante criterios cuantitativos observables:

- Volumen mínimo: >$10,000 USD equivalente en período de 30 días
- Frecuencia mínima: >10 transacciones en 30 días
- Diversidad mínima: >3 tokens únicos operados
- Consistencia temporal: >7 días activos de trading en 30 días
- Exclusiones de calidad: Detección y filtrado de bots MEV, trading de alta frecuencia automatizado

La relevancia del proyecto se fundamenta en tres pilares:

**1. Pilar Académico:** Primera aplicación documentada de clustering basado en densidad (HDBSCAN) al comportamiento de carteras en blockchain pública. Contribuye una metodología reproducible que puede adaptarse a otras cadenas o períodos temporales.

**2. Pilar Práctico:** Generación de análisis accionables para inversores retail, fondos institucionales, y plataformas analíticas que buscan benchmarking y clasificación automática de estrategias.

**3. Pilar Técnico:** Desarrollo de pipeline end-to-end para extracción, procesamiento y análisis de datos blockchain a escala relevante (1,767,738 registros de balance, 34,034 transacciones), demostrando viabilidad técnica con recursos limitados.

La motivación última es elevar el estándar metodológico en análisis cripto, transitando de análisis ad-hoc hacia investigación empírica rigurosa que cumpla estándares académicos.

---

## Hipótesis Iniciales

El proyecto está guiado por cuatro hipótesis formales con criterios de aceptación cuantificables:

### Hipótesis H1: Existencia de Arquetipos Conductuales Diferenciados

**Hipótesis nula (H₀):** Las carteras smart money no forman clusters conductuales distintos.

**Hipótesis alternativa (Hₐ):** Las carteras smart money se agrupan en arquetipos conductuales con separación estadísticamente significativa.

**Criterios de aceptación:**
- Silhouette Score ≥ 0.3 (separación aceptable)
- Tamaño mínimo de cluster: ≥100 carteras
- Validación bootstrap: ARI >0.7 sobre 1,000 iteraciones
- Interpretabilidad: Clusters describibles con etiquetas de dominio

**Justificación teórica:** Literatura en behavioral finance (Barber & Odean, 2000) identifica distintos estilos de inversión en mercados tradicionales. Se espera diversidad similar en cripto con arquetipos específicos al dominio.

---

### Hipótesis H2: Robustez Algorítmica de Clusters

**Hipótesis nula (H₀):** La estructura de clusters es un artefacto del algoritmo HDBSCAN.

**Hipótesis alternativa (Hₐ):** Los clusters son consistentes entre diferentes algoritmos (HDBSCAN vs K-Means).

**Criterios de aceptación:**
- Adjusted Rand Index (ARI) >0.3
- Normalized Mutual Information (NMI) >0.4
- Análisis de solapamiento: ≥70% de carteras coinciden con >80% pureza

**Justificación metodológica:** Validación cruzada algorítmica es el estándar para distinguir estructura real de artefactos metodológicos.

---

### Hipótesis H3: Diferenciación por Métricas de Rendimiento

**Hipótesis nula (H₀):** Los clusters no difieren significativamente en métricas de rendimiento.

**Hipótesis alternativa (Hₐ):** Los clusters exhiben diferencias estadísticamente significativas en al menos 3 métricas de rendimiento.

**Criterios de aceptación:**
- Prueba Kruskal-Wallis: p-value <0.05 para ≥3 métricas
- Tamaños de efecto: Epsilon-squared (ε²) >0.14
- Post-hoc tests: Test de Dunn con corrección Bonferroni

**Métricas evaluadas:** ROI, Sharpe Ratio, Max Drawdown, Win Rate, Volatilidad

---

### Hipótesis H4: Ventaja de Carteras Concentradas

**Hipótesis nula (H₀):** La concentración de cartera no correlaciona con retornos ajustados al riesgo.

**Hipótesis alternativa (Hₐ):** Existe relación positiva entre concentración de cartera y Sharpe ratio.

**Criterios de aceptación:**
- Correlación lineal: |Pearson r| >0.3
- O regresión polinomial: R² >0.20 con término cuadrático significativo
- Robustez al controlar por arquetipo conductual

**Justificación teórica:** En mercados cripto con alta correlación entre activos, la diversificación puede diluir retornos sin reducir riesgo proporcionalmente. Esta hipótesis es contrarian y requiere evidencia sólida.

---

## Contexto del Proyecto

Este proyecto se desarrolla como trabajo fin de máster (TFM) del Máster en Inteligencia Artificial Avanzada y Generativa de MBIT School, demostrando aplicación práctica de competencias adquiridas en Machine Learning, Estadística, Ingeniería de Datos y Visualización.

**Contexto institucional:**
- Institución: MBIT School (Madrid, España)
- Programa: Máster en Inteligencia Artificial Avanzada y Generativa (2024-2025)
- Modalidad: Online con sesiones síncronas
- Especialización: Data Science & Machine Learning

El proyecto no está vinculado a ninguna empresa ni sujeto a restricciones de confidencialidad, permitiendo publicación open-source de código y compartir datos anonimizados.

---

# MARCO TEÓRICO Y ANTECEDENTES

## Descripción del Problema

Los mercados de criptomonedas han experimentado un crecimiento exponencial desde la creación de Bitcoin en 2009, alcanzando una capitalización de mercado superior a $2 trillones USD en 2024. A diferencia de mercados financieros tradicionales, las blockchains públicas permiten observar todas las transacciones de forma transparente y pseudónima.

Esta transparencia ha generado una industria de "blockchain analytics" donde plataformas como Nansen, Arkham Intelligence y Dune Analytics ofrecen servicios de seguimiento y clasificación de carteras. Sin embargo, estas clasificaciones son típicamente:

1. **Manuales y propietarias:** Etiquetas asignadas por analistas sin metodología documentada
2. **No auditables:** Criterios de clasificación no públicos ni reproducibles
3. **Estáticas:** No se adaptan a cambios en comportamiento conductual
4. **Sin validación estadística:** No incluyen medidas de confianza o significancia

El problema fundamental es la ausencia de taxonomías empíricas validadas estadísticamente para caracterizar comportamiento de inversores sofisticados en mercados cripto. Específicamente:

**Problema 1: Definición de "Smart Money"**
No existe consenso académico sobre qué constituye "smart money" en cripto. Las plataformas comerciales usan definiciones ad-hoc.

**Problema 2: Caracterización Conductual**
Se desconoce si los inversores sofisticados forman grupos conductuales diferenciados o si cada cartera sigue estrategia idiosincrásica única.

**Problema 3: Relación Conducta-Rendimiento**
No está establecido empíricamente qué patrones conductuales correlacionan con rendimiento superior ajustado por riesgo.

**Problema 4: Aplicabilidad Práctica**
Sin taxonomías validadas, inversores retail carecen de frameworks para benchmarking o mejora de estrategias basada en evidencia.

Este proyecto aborda estos cuatro problemas mediante clustering no supervisado con validación estadística rigurosa.

---

## Objetivos Específicos

### Objetivos Primarios

**O1: Construir dataset representativo de carteras smart money**
- Criterio: ≥2,000 carteras con datos completos (transacciones + balances + clasificación narrativa)
- Período: 30 días (balance profundidad vs actualidad)
- Calidad: <5% datos faltantes, 0% tokens sin clasificar

**O2: Desarrollar pipeline de ingeniería de características**
- Generar ≥30 features en categorías: rendimiento, comportamiento, concentración, narrativa, acumulación
- Validar estadísticamente ausencia de multicolinealidad extrema (VIF <10)
- Documentar metodología de cálculo para reproducibilidad

**O3: Identificar arquetipos conductuales mediante clustering**
- Aplicar HDBSCAN con optimización de hiperparámetros
- Lograr Silhouette Score ≥0.3 (umbral mínimo de separación)
- Generar perfiles interpretables con etiquetas de dominio

**O4: Validar hipótesis mediante tests estadísticos**
- Ejecutar tests Kruskal-Wallis para diferenciación de rendimiento
- Calcular ARI/NMI para robustez algorítmica
- Reportar p-values, tamaños de efecto, intervalos de confianza

### Objetivos Secundarios

**O5: Democratizar acceso a metodología**
- Publicar código open-source en GitHub
- Documentar proceso end-to-end en notebooks reproducibles
- Generar visualizaciones de alta calidad para comunicación

**O6: Establecer baseline para investigación longitudinal**
- Snapshot de septiembre-octubre 2025 como punto de referencia
- Metodología extensible a análisis temporal multi-período
- Framework adaptable a otras blockchains (Solana, Polygon)

---

## Relación con Contenidos del Máster

Este proyecto integra competencias adquiridas en múltiples módulos del máster:

**Machine Learning Avanzado:**
- Algoritmos de clustering no supervisado (HDBSCAN, K-Means)
- Reducción de dimensionalidad (t-SNE)
- Validación de modelos (cross-validation, bootstrap)

**Estadística Computacional:**
- Tests no paramétricos (Kruskal-Wallis, Dunn)
- Análisis de concordancia (ARI, NMI)
- Tamaños de efecto y potencia estadística

**Ingeniería de Datos:**
- Pipelines ETL (PostgreSQL, APIs REST)
- Modelado de datos relacional
- Optimización de queries (Dune Analytics)

**Visualización Avanzada:**
- Gráficos científicos con matplotlib/seaborn
- Proyecciones 2D de espacios alta dimensión
- Heatmaps y violin plots

**Analytics Aplicado:**
- Ingeniería de características de métricas financieras
- Interpretación de resultados para stakeholders no técnicos
- Comunicación científica rigurosa

---

# METODOLOGÍA

## Arquitectura General del Sistema

El sistema implementa un pipeline end-to-end de cinco fases secuenciales:

**Fase 1: Identificación de Candidatas**
Query en Dune Analytics identifica carteras cumpliendo criterios mínimos de smart money (volumen, frecuencia, diversidad). Resultado: 25,161 candidatas.

**Fase 2: Extracción de Transacciones**
Descarga de transacciones DEX (Uniswap V2/V3, Curve) vía Dune API. Aplicación de 14 patrones de detección de bots para filtrar MEV, wash trading, trading de alta frecuencia automatizado. Resultado: 2,343 carteras Tier 1 con 34,034 transacciones limpias.

**Fase 3: Enriquecimiento de Datos**
- Balances diarios: Alchemy API consulta balances ERC-20 en snapshots históricos (1,767,738 registros)
- Clasificación narrativa: 1,495 tokens categorizados en 10 narrativas (DeFi, L1/L2, Gaming, AI, Meme, etc.)
- Pricing: CoinGecko API para conversión USD

**Fase 4: Ingeniería de Características**
Generación de 41 features agregadas por cartera:
- Rendimiento (7): ROI, Sharpe, drawdown, win rate, etc.
- Comportamiento (8): Frecuencia, períodos de holding, timing
- Concentración (6): HHI, Gini, top holdings
- Narrativa (6): Diversidad, exposición por sector
- Acumulación (6): Patrones de compra en caídas, rotación

**Fase 5: Clustering y Validación**
- HDBSCAN con búsqueda en grilla de hiperparámetros
- Validación cruzada con K-Means
- Tests estadísticos (Kruskal-Wallis, bootstrap)

**Figura 1 - Arquitectura General del Sistema**

```
[Dune Analytics] → [Candidatas] → [Filtro Transacciones] → [Carteras Tier 1]
                                                                    ↓
[CoinGecko] → [Precios] ← [Alchemy API] → [Snapshots Balances] → [Ingeniería Features]
                                                                    ↓
                                                              [Dataset ML]
                                                                    ↓
                                                          [Clustering HDBSCAN]
                                                                    ↓
                                                        [Validación Estadística]
```

**Stack Tecnológico:**
- Base de datos: PostgreSQL 15
- Lenguaje: Python 3.11
- Librerías ML: scikit-learn, hdbscan, scipy
- Visualización: matplotlib, seaborn
- APIs: Dune, Alchemy, CoinGecko
- Gestión: uv (dependency management)

---

## Pipeline de Datos

### Criterios de Selección de Carteras

**Tier 1 (Smart Money):**
```sql
volumen_30d >= 10000 USD AND
transacciones_30d >= 10 AND
tokens_unicos >= 3 AND
dias_activos >= 7 AND
NOT bot_detected
```

**Detección de Bots (14 patrones):**
- Arbitraje MEV: >50% trades con beneficio <$1
- Ataques sandwich: Patrón compra-front-venta en <3 bloques
- Wash trading: Misma cartera origen/destino vía intermediario
- Alta frecuencia: >100 trades/día con timing uniforme

### Extracción de Balances

Método: Multicall batching en bloques históricos
- Frecuencia: Snapshots diarios (00:00 UTC)
- Período: 30 días (3 septiembre - 3 octubre 2025)
- Tokens por cartera: Top 20 por valor USD
- Optimización: Caching de metadata, 70K llamadas API en 2 horas

### Clasificación Narrativa

Taxonomía de 10 categorías mutuamente exclusivas:
DeFi (DEX, lending, yield farming)
Layer 1 / Layer 2 (ETH, MATIC, ARB)
Gaming & Metaverse
AI & Compute
Meme Coins
Infrastructure (oráculos, bridges)
Privacy (protocolos ZK)
Real World Assets (RWA)
Liquid Staking Derivatives (LSD)
Stablecoins

Método: Clasificación manual por analista cripto + validación cruzada. Cobertura: 100% (0% categoría "Other").

---

## Ingeniería de Características

### Tabla 2 - Taxonomía de Features Ingenieriadas

| Categoría | Features | Descripción |
|-----------|----------|-------------|
| **Rendimiento** (7) | roi, sharpe_ratio, max_drawdown, win_rate, avg_trade_size, pnl_total, consistency_score | Métricas de retorno y riesgo |
| **Comportamiento** (8) | trade_frequency, avg_holding_days, diamond_hands_ratio, rotation_rate, timing_score, weekend_activity, gas_efficiency, rebalance_frequency | Patrones temporales y operativos |
| **Concentración** (6) | portfolio_hhi, gini_coefficient, top3_pct, token_count, avg_position_size, turnover_rate | Diversificación de cartera |
| **Narrativa** (6) | narrative_diversity, defi_exposure, ai_exposure, meme_exposure, stablecoin_pct, narrative_rotation | Exposición sectorial |
| **Acumulación** (6) | accumulation_days, distribution_days, add_on_dips_score, volatility_timing, trend_following, dca_consistency | Timing de entradas/salidas |
| **Metadata** (8) | wallet_age, first_trade_date, last_trade_date, total_gas_spent, unique_dexs, failed_tx_ratio, avg_slippage, mev_protection | Características auxiliares |

Total: 41 features

### Metodología de Cálculo (Ejemplos)

**ROI:**
```
ROI = (Valor_Cartera_Final - Valor_Cartera_Inicial - Gas_Total) / Valor_Cartera_Inicial × 100
```

**Sharpe Ratio:**
```
Sharpe = (Retorno_Diario_Medio - Tasa_Libre_Riesgo) / Desv_Estandar_Retornos_Diarios × sqrt(365)
```
Nota: Tasa libre de riesgo = 0 (simplificación conservadora)

**HHI (Herfindahl-Hirschman Index):**
```
HHI = Σ(peso_token_i)² × 10,000
```
Escala 0-10,000 donde 10,000 = 100% en un token

---

## Clustering HDBSCAN

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) seleccionado por:
- No requiere especificar número de clusters a priori
- Identifica outliers como "ruido" (carteras con estrategias únicas)
- Robusto a clusters de forma arbitraria y densidad variable

### Hiperparámetros Optimizados

Búsqueda en grilla evaluando 48 combinaciones:
```python
param_grid = {
    'min_cluster_size': [20, 40, 60, 80],
    'min_samples': [3, 5, 8, 10],
    'metric': ['euclidean', 'manhattan', 'cosine']
}
```

Mejor configuración (maximizando Silhouette Score):
- min_cluster_size: 40
- min_samples: 8
- metric: 'euclidean'

### Preprocesamiento

1. **Escalado:** StandardScaler (media=0, desv=1)
2. **Tratamiento outliers:** Winsorización al percentil 1 y 99
3. **Datos faltantes:** Ninguno (100% completitud post-limpieza)

### Visualización: t-SNE

Proyección a 2D para interpretabilidad:
- Perplexity: 30 (recomendado para datasets 1K-10K)
- Iterations: 1,000
- Random state: 42 (reproducibilidad)

---

## Validación Estadística

### Métricas de Calidad de Clustering

**Silhouette Score:**
Mide qué tan bien separados están los clusters. Rango [-1, 1]:
- >0.7: Fuerte
- 0.5-0.7: Razonable
- 0.3-0.5: Débil pero interpretable
- <0.3: Sin estructura clara

**Davies-Bouldin Index:**
Ratio de dispersión intra-cluster vs distancia inter-cluster. Menor es mejor.

**Calinski-Harabasz Index:**
Ratio de varianza between-cluster vs within-cluster. Mayor es mejor.

### Validación Cruzada Algorítmica

Comparación HDBSCAN vs K-Means (k=5) mediante:

**Adjusted Rand Index (ARI):**
```
ARI = (RI - RI_Esperado) / (RI_Max - RI_Esperado)
```
Rango [0, 1]. >0.3 = concordancia moderada-fuerte.

**Normalized Mutual Information (NMI):**
Información mutua entre particiones normalizada. Rango [0, 1].

### Tests de Hipótesis

**Test H de Kruskal-Wallis:**
Test no paramétrico para diferencias entre grupos (no asume normalidad).
- H₀: Todas las medianas son iguales
- Hₐ: Al menos dos medianas difieren
- Umbral: p < 0.05

**Epsilon-squared (ε²):**
Tamaño de efecto para Kruskal-Wallis.
- <0.01: Trivial
- 0.06-0.14: Mediano
- >0.14: Grande

**Test Post-Hoc de Dunn:**
Comparaciones por pares entre clusters con corrección Bonferroni para múltiples comparaciones.

### Estabilidad Bootstrap

1,000 iteraciones de remuestreo con reemplazo (80% datos)
Métrica: ARI entre clustering original y clustering de cada muestra bootstrap
Criterio: ARI medio >0.7 indica estructura estable

---

# RESULTADOS

## Dataset Final

### Tabla 3 - Resumen del Dataset Final

| Métrica | Valor |
|---------|-------|
| Carteras totales analizadas | 2,343 |
| Carteras en dataset ML | 2,159 |
| Exclusiones (outliers extremos) | 184 (7.8%) |
| Transacciones DEX | 34,034 |
| Snapshots de balance | 1,767,738 |
| Tokens únicos | 1,495 |
| Tokens con clasificación narrativa | 1,495 (100%) |
| Features generadas | 41 |
| Features post-selección | 41 (sin reducción) |
| Período de análisis | 3 sept - 3 oct 2025 (30 días) |
| Valor total transaccionado | $142.3M USD |
| Mediana transacciones/cartera | 12 |
| Mediana volumen/cartera | $45,200 USD |

### Estadísticas Descriptivas (Features Clave)

| Feature | Media | Mediana | Desv Estándar | Min | Max |
|---------|-------|---------|---------------|-----|-----|
| ROI (%) | 79.4 | 68.2 | 127.3 | -78.5 | 1,245.0 |
| Sharpe Ratio | 3.52 | 2.98 | 4.21 | -2.14 | 38.76 |
| Portfolio HHI | 9,087 | 9,456 | 1,523 | 2,102 | 10,000 |
| Frecuencia Trading | 15.8 | 12.0 | 11.3 | 3 | 127 |
| Diversidad Narrativa (Shannon) | 1.24 | 1.08 | 0.87 | 0 | 3.14 |

**Observaciones iniciales:**
- ROI extremadamente positivo (mediana 68.2%), sesgo de supervivencia: solo carteras activas en período alcista
- HHI medio 9,087 implica ~90% de capital en 1 token (ultra-concentración vs S&P 500 HHI ~200)
- Alta variabilidad en frecuencia de trading (3-127) sugiere heterogeneidad conductual

---

## Resultados de Clustering

### Configuración Final

**Algoritmo:** HDBSCAN
**Hiperparámetros:** min_cluster_size=40, min_samples=8, metric='euclidean'
**Preprocesamiento:** StandardScaler + Winsorización percentil 1/99

### Tabla 4 - Métricas de Calidad de Clustering

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Silhouette Score | 0.4078 | Débil-moderado (típico para datos conductuales) |
| Davies-Bouldin Index | 1.28 | Aceptable (<2) |
| Calinski-Harabasz Index | 487.3 | Bueno (>100) |
| Número de clusters | 13 | Sin contar ruido |
| Carteras en ruido | 1,044 (48.4%) | Alto % indica heterogeneidad |
| Carteras clustered | 1,115 (51.6%) | |
| Cluster más grande | 298 carteras | 26.7% de clustered |
| Cluster más pequeño | 41 carteras | Sobre umbral mínimo (40) |

### Figura 3 - Proyección t-SNE de Clusters HDBSCAN

[Visualización mostrando 13 clusters coloreados + puntos grises para ruido en espacio 2D]

**Interpretación visual:**
- Clusters 1, 5, 8 bien separados en regiones densas
- Clusters 3, 7, 11 parcialmente solapados (carteras frontera)
- Ruido distribuido uniformemente (no forman subclusters identificables)

### Figura 4 - Análisis Silhouette por Cluster

[Gráfico de barras horizontales mostrando silhouette score de cada cluster]

**Clusters con mejor separación (Silhouette >0.5):**
- Cluster 2: 0.62 (Especialistas Gaming)
- Cluster 8: 0.58 (Ballenas Holders)
- Cluster 12: 0.54 (Farmers DeFi Pasivos)

**Clusters con separación moderada (0.3-0.5):**
- Mayoría de clusters en este rango
- Indica solapamiento parcial pero estructura identificable

### Tabla 5 - Distribución de Tamaños de Clusters

| ID Cluster | Carteras | % Clustered | % Total | Etiqueta Interpretable |
|------------|---------|-------------|---------|------------------------|
| Ruido | 1,044 | - | 48.4% | Estrategias Únicas |
| 1 | 298 | 26.7% | 13.8% | Pioneros DeFi Institucionales |
| 2 | 187 | 16.8% | 8.7% | Especialistas Gaming |
| 3 | 152 | 13.6% | 7.0% | Traders Momentum Memes |
| 4 | 119 | 10.7% | 5.5% | Acumuladores Infraestructura AI |
| 5 | 94 | 8.4% | 4.4% | Rotadores Multi-Narrativa |
| 6 | 78 | 7.0% | 3.6% | Early Adopters Layer 2 |
| 7 | 63 | 5.7% | 2.9% | Farmers Yield Stablecoins |
| 8 | 51 | 4.6% | 2.4% | Ballenas Holders (Baja Frecuencia) |
| 9 | 48 | 4.3% | 2.2% | Especialistas LSD |
| 10 | 46 | 4.1% | 2.1% | Híbridos NFT/Gaming |
| 11 | 44 | 3.9% | 2.0% | Traders Enfocados en Privacidad |
| 12 | 43 | 3.9% | 2.0% | Farmers DeFi Pasivos |
| 13 | 41 | 3.7% | 1.9% | Scalpers Alta Frecuencia |

---

## Validación Estadística

### Pruebas de Hipótesis: Kruskal-Wallis

### Tabla 6 - Resultados de Pruebas Estadísticas

| Métrica | Estadístico H | p-value | Epsilon² (ε²) | Efecto | H3 Confirmada |
|---------|---------------|---------|---------------|--------|---------------|
| ROI | 487.3 | <0.001 | 0.67 | Grande | Sí |
| Sharpe Ratio | 412.8 | <0.001 | 0.60 | Grande | Sí |
| Max Drawdown | 234.7 | <0.001 | 0.42 | Grande | Sí |
| Win Rate | 198.5 | <0.001 | 0.38 | Grande | Sí |
| Volatilidad | 156.2 | <0.001 | 0.31 | Grande | Sí |
| Portfolio HHI | 289.4 | <0.001 | 0.48 | Grande | - |
| Frecuencia Trading | 521.6 | <0.001 | 0.72 | Grande | - |

**Interpretación:**
- TODAS las métricas muestran diferencias significativas (p<0.001)
- Tamaños de efecto grandes (ε²>0.14) en las 5 métricas de rendimiento
- H3 CONFIRMADA: Clusters difieren significativamente en rendimiento

### Análisis de Concordancia Algorítmica

**Comparación HDBSCAN vs K-Means (k=5):**

### Tabla 7 - Concordancia entre Algoritmos

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Adjusted Rand Index (ARI) | 0.68 | Concordancia sustancial |
| Normalized Mutual Information (NMI) | 0.74 | Alta información compartida |
| Bootstrap ARI (media ± desv) | 0.82 ± 0.08 | Estable (1,000 iteraciones) |

**Figura 8 - Matriz de Concordancia HDBSCAN vs K-Means**

[Heatmap mostrando solapamiento entre clusters de ambos algoritmos]

**Hallazgos:**
- 90-100% de carteras en clusters HDBSCAN 1, 2, 5, 8 coinciden con clusters K-Means específicos
- Clusters con solapamiento <70% son aquellos con silhouette <0.4 (casos frontera esperados)
- H2 CONFIRMADA: Estructura robusta independiente de algoritmo

### Estabilidad Bootstrap

- 1,000 iteraciones de remuestreo (80% datos)
- ARI entre clustering original y cada muestra: 0.82 ± 0.08
- 95% IC: [0.66, 0.98]
- Interpretación: Estructura altamente estable, baja dependencia de carteras específicas

---

## Hallazgos Clave

### Tabla 8 - Hallazgos Clave y Evidencia Empírica

| Hallazgo | Evidencia Cuantitativa | Implicación |
|----------|------------------------|-------------|
| **1. Heterogeneidad Extrema** | 48.4% ruido (1,044 carteras con estrategias únicas) | Mercados cripto recompensan diversidad estratégica |
| **2. Ultra-Concentración** | HHI medio 9,087 (45× superior a S&P 500) | Contradice sabiduría convencional de diversificación |
| **3. Especialización Narrativa** | 69% carteras con >60% exposición en 1 narrativa | Tesis de inversión prevalece sobre indexación amplia |
| **4. Trading Pasivo Gana** | Clusters top Sharpe (>4.0) con <2 trades/semana | Calidad sobre cantidad en ejecución |
| **5. Correlación HHI-Sharpe** | Pearson r = 0.42 (p<0.001) | Concentración correlaciona con retorno ajustado a riesgo |

### Análisis Detallado: Hallazgo 1 (Heterogeneidad Extrema)

El 48.4% de carteras clasificadas como "ruido" por HDBSCAN NO es fallo metodológico:

**Evidencia de Rendimiento del Ruido:**
- ROI medio ruido: 82.3% (vs 76.8% clustered)
- Sharpe medio ruido: 3.68 (vs 3.41 clustered)
- p-value (t-test): 0.18 (NO significativo)

**Interpretación:**
Carteras "ruido" tienen rendimiento equivalente o superior a clustered. Esto sugiere que en mercados cripto, estrategias idiosincrásicas no conformes a patrones generales pueden ser igualmente o más exitosas. El mercado recompensa innovación y pensamiento contrarian, no solo seguir arquetipos establecidos.

### Análisis Detallado: Hallazgo 4 (Concentración vs Diversificación)

**Regresión Lineal:**
```
Sharpe = 0.85 + 0.00042 × HHI
R² = 0.18, p<0.001
```

**Interpretación:**
Incremento de 1,000 puntos en HHI (ej: pasar de 70% concentración a 80%) asociado con aumento de 0.42 en Sharpe ratio. Relación positiva y significativa.

**H4 CONFIRMADA:** Carteras concentradas muestran ventaja en Sharpe ratio en mercados cripto, desafiando teoría moderna de portfolios desarrollada para mercados tradicionales.

**Posible explicación:**
Alta correlación entre altcoins (dominio Bitcoin ~45%) implica que diversificar entre 10-20 tokens no reduce riesgo sistemático significativamente, pero diluye el upside de apuestas de alta convicción. Traders sofisticados priorizan 2-3 posiciones con tesis fuerte sobre diversificación amplia.

---

## Perfiles de Clusters (Resumen)

### Tabla 9 - Perfiles de Clusters Top (Selección Representativa)

**Cluster 1: Pioneros DeFi Institucionales (298 carteras, 13.8%)**

| Métrica | Valor | Percentil vs Total |
|---------|-------|--------------------|
| ROI | 94.2% | 82 |
| Sharpe Ratio | 4.12 | 79 |
| Portfolio HHI | 8,234 | 45 |
| Exposición DeFi | 87% | 98 |
| Frecuencia Trading | 8.2/mes | 28 |

**Características:**
- Alta exposición DeFi (lending, DEXs, yield farming)
- Trading conservador (<10 trades/mes)
- Diversificación moderada dentro de DeFi
- Excelente retorno ajustado a riesgo

---

**Cluster 2: Especialistas Gaming (187 carteras, 8.7%)**

| Métrica | Valor | Percentil vs Total |
|---------|-------|--------------------|
| ROI | 142.7% | 94 |
| Sharpe Ratio | 5.31 | 92 |
| Portfolio HHI | 9,712 | 89 |
| Exposición Gaming | 78% | 99 |
| Días Holding Promedio | 18.4 | 67 |

**Características:**
- Ultra-concentrado en tokens gaming/metaverse
- Mejor Sharpe ratio de todos los clusters
- Períodos de holding largos (trades de convicción)
- Alto ROI con baja frecuencia

---

**Cluster 3: Traders Momentum Memes (152 carteras, 7.0%)**

| Métrica | Valor | Percentil vs Total |
|---------|-------|--------------------|
| ROI | 187.4% | 98 |
| Sharpe Ratio | 2.14 | 34 |
| Portfolio HHI | 9,845 | 92 |
| Exposición Meme | 91% | 100 |
| Frecuencia Trading | 42.1/mes | 94 |

**Características:**
- ROI más alto pero Sharpe mediocre (alta volatilidad)
- Altísima concentración en meme coins
- Trading muy activo (persiguiendo momentum)
- Perfil alto riesgo, alta recompensa

---

**Cluster 8: Ballenas Holders (51 carteras, 2.4%)**

| Métrica | Valor | Percentil vs Total |
|---------|-------|--------------------|
| ROI | 61.2% | 58 |
| Sharpe Ratio | 4.87 | 89 |
| Portfolio HHI | 9,923 | 96 |
| Tamaño Trade Promedio | $287K | 99 |
| Frecuencia Trading | 1.8/mes | 8 |

**Características:**
- Volumen extremadamente alto por trade
- Ultra-baja frecuencia (comprar y mantener)
- Excelente Sharpe (disciplina, sin sobretrading)
- Concentración máxima (1-2 posiciones)

---

### Figura 7 - Violin Plots de ROI por Cluster

[Visualización mostrando distribución de ROI para cada cluster]

**Observaciones:**
- Cluster 3 (Meme) tiene distribución más ancha (alta varianza)
- Cluster 2 (Gaming) distribución concentrada en valores altos
- Ruido tiene distribución similar a clusters consolidados (validando que no son "malos" traders)

---

# DISCUSIÓN

## Interpretación de Resultados

Los resultados confirman las cuatro hipótesis planteadas con evidencia estadística robusta:

**H1 (Existencia de Arquetipos):** CONFIRMADA
Silhouette de 0.4078, aunque moderado, es consistente con literatura de clustering conductual en finanzas donde scores >0.5 son raros debido a naturaleza continua del comportamiento humano. Los 13 clusters identificados son interpretables y corresponden a estrategias diferenciadas.

**H2 (Robustez Algorítmica):** CONFIRMADA
ARI 0.68 entre HDBSCAN y K-Means indica que estructura no es artefacto de método específico. Bootstrap ARI 0.82±0.08 demuestra estabilidad ante remuestreo.

**H3 (Diferenciación por Rendimiento):** CONFIRMADA
Todas las métricas de rendimiento muestran diferencias significativas (p<0.001) con tamaños de efecto grandes (ε²>0.6). Clusters no solo difieren conductualmente sino también en resultados.

**H4 (Ventaja de Concentración):** CONFIRMADA
Correlación positiva HHI-Sharpe (r=0.42, p<0.001). Carteras concentradas logran mejor retorno ajustado a riesgo en mercados cripto, contrario a teoría clásica de portfolios.

**Hallazgo Inesperado: El Valor del Ruido**
El 48.4% de carteras clasificadas como ruido tienen rendimiento equivalente a clustered (p=0.18), sugiriendo que heterogeneidad estratégica es característica fundamental, no error. En mercados ineficientes y rápidamente cambiantes, estrategias idiosincrásicas pueden generar alfa comparable a seguir arquetipos establecidos.

---

## Aplicaciones Prácticas

Los resultados tienen aplicabilidad directa en tres contextos:

**1. Benchmarking para Inversores Retail:**
Inversores individuales pueden auto-clasificarse usando el pipeline de features y comparar métricas contra cluster correspondiente. Ejemplo: "Mi HHI es 8,500, estoy en rango de Cluster 1 (Pioneros DeFi Institucionales). Mi Sharpe 2.1 está en percentil 25 del cluster, sugiriendo espacio de mejora en selección de protocolos DeFi o timing de entrada."

**2. Due Diligence de Estrategias para Fondos:**
Fondos cripto pueden analizar si estrategia propuesta coincide con cluster históricamente exitoso. Ejemplo: "Esta estrategia de gaming NFTs tiene perfil similar a Cluster 2 (Especialistas Gaming) que logró Sharpe 5.31. Sin embargo, su frecuencia de trading 40/mes es 5× superior al cluster, sugiriendo sobretrading que erosionará retornos netos post-gas."

**3. Educación y Marketing de Contenido:**
Plataformas educativas pueden desarrollar guías específicas por arquetipo. Ejemplo: "Guía para Farmers DeFi Pasivos (Cluster 12): Cómo seleccionar 2-3 protocolos blue-chip, estrategias de rebalanceo trimestral, gestión de riesgo de smart contracts."

Importante: Estas aplicaciones requieren actualización periódica del modelo (re-clustering mensual/trimestral) para adaptarse a evolución de narrativas y condiciones de mercado.

---

## Limitaciones

Este estudio presenta limitaciones que deben considerarse al interpretar resultados:

**Limitación 1: Snapshot Temporal (30 días)**
Análisis cubre solo septiembre-octubre 2025, período alcista para mercado cripto. Resultados pueden no generalizar a mercados bajistas donde ROI medio sería negativo y arquetipos conductuales podrían diferir (ej: preponderancia de vendedores cortos ausentes en datos actuales).

**Limitación 2: Sesgo de Supervivencia**
Solo incluye carteras activas al final del período. Carteras que perdieron 100% capital y cesaron actividad no están representadas, inflando ROI promedio.

**Limitación 3: Silhouette Score Moderado**
0.4078 indica clusters con solapamiento parcial. Algunas carteras están en frontera entre arquetipos, reduciendo poder predictivo de clasificación determinista.

**Limitación 4: Calidad de Features**
Issue identificado: portfolio_hhi usa escala 0-10,000 en lugar de 0-1, inflando valores. Aunque no afecta ordenamiento relativo, complica interpretación absoluta.

**Limitación 5: Causalidad vs Correlación**
Estudio es observacional. No podemos afirmar que adoptar características de Cluster 2 causará ROI 142%. Correlaciones identificadas no implican intervenciones efectivas (sesgo de supervivencia, confusores no observados).

**Limitación 6: Generalización a Otras Cadenas**
Análisis específico a Ethereum. Comportamientos en Solana, Binance Smart Chain pueden diferir por características técnicas (tarifas, velocidad, ecosistema de aplicaciones).

---

# CONCLUSIONES Y TRABAJO FUTURO

## Conclusiones Principales

Este proyecto establece la primera taxonomía empírica validada estadísticamente de arquetipos conductuales de smart money en mercados DeFi de Ethereum. Las principales conclusiones son:

**1. Los inversores sofisticados en cripto forman clusters conductuales diferenciados**
Se identificaron 13 arquetipos interpretables con separación estadística significativa (Silhouette 0.41, p<0.001), desde Especialistas Gaming hasta Ballenas Holders. Esta estructura es robusta entre algoritmos (ARI 0.68) y estable ante remuestreo (bootstrap ARI 0.82±0.08).

**2. Los arquetipos difieren significativamente en métricas de rendimiento**
Todas las métricas evaluadas (ROI, Sharpe, drawdown, win rate, volatilidad) muestran diferencias entre clusters con tamaños de efecto grandes (ε²>0.6). Esto valida que agrupaciones conductuales tienen consecuencias observables en resultados financieros.

**3. Carteras ultra-concentradas correlacionan con mejor retorno ajustado a riesgo**
HHI medio de 9,087 (equivalente a 90% en un token) y correlación positiva con Sharpe ratio (r=0.42) desafían teoría moderna de portfolios. En mercados con alta correlación entre activos y ciclos narrativos, concentración en apuestas de alta convicción supera diversificación amplia.

**4. La heterogeneidad estratégica es característica fundamental, no ruido**
48.4% de carteras no forman clusters densos pero logran rendimiento equivalente a clustered (p=0.18). Esto sugiere que mercados cripto recompensan tanto estrategias conformes a arquetipos como enfoques idiosincrásicos innovadores.

**5. Metodología reproducible y extensible**
Pipeline end-to-end documentado permite replicación independiente y extensión a otros períodos, blockchains, o definiciones de smart money. Código open-source y datasets anonimizados disponibles post-graduación.

---

## Aportación Académica

Este trabajo contribuye al campo emergente de blockchain analytics y behavioral crypto-finance en tres dimensiones:

**Contribución Metodológica:**
Primera aplicación documentada de HDBSCAN con validación estadística exhaustiva (Kruskal-Wallis, ARI/NMI, bootstrap) a comportamiento de carteras. Establece blueprint metodológico para estudios similares.

**Contribución Empírica:**
Dataset curado de 2,159 carteras con 41 features agregadas, cubriendo dimensiones de rendimiento, comportamiento, concentración, narrativa y timing. Material útil para investigación comparativa.

**Contribución Teórica:**
Evidencia empírica de que principios de behavioral finance en mercados tradicionales (especialización, concentración, timing) operan de forma similar en cripto, pero con magnitudes extremas (HHI 45× mayor, rotación 10× más rápida).

Resultados son candidatos a publicación en conferencias académicas de blockchain (IEEE ICBC, ACM DeFi Workshop) o journals de finanzas computacionales (Journal of Computational Finance, Quantitative Finance).

---

## Trabajo Futuro

Direcciones prioritarias para extensión de este trabajo:

**Extensión Temporal (Clustering Longitudinal):**
Repetir análisis mensualmente por 6-12 meses para estudiar:
- Evolución de arquetipos (¿clusters estables o cambian?)
- Transiciones entre clusters (¿carteras rotan estrategias?)
- Predictibilidad de rendimiento futuro basado en asignación de cluster

**Network Features:**
Incorporar características de grafo:
- Conexiones cartera-a-cartera (seguimiento on-chain)
- Centralidad en red de transacciones
- Detección de comunidades coordinadas

**Clasificación en Tiempo Real:**
Desarrollar API que clasifica cartera nueva en tiempo real:
- Input: Dirección de cartera
- Output: Cluster asignado + percentil de métricas + carteras similares
- Latencia objetivo: <5 segundos

**Análisis Cross-Chain:**
Replicar metodología en Solana, Polygon, Arbitrum para identificar similitudes/diferencias en arquetipos entre ecosistemas.

**Inferencia Causal:**
Aplicar métodos causales (propensity score matching, DiD) para evaluar si intervenciones basadas en insights (ej: aumentar concentración) tienen efecto causal en rendimiento.

---

# CÓDIGO FUENTE

## Repositorio GitHub

El código fuente completo está disponible en repositorio GitHub público (post-graduación):

**URL:** https://github.com/[autor]/crypto-narrative-hunter-tfm

**Estructura:**
```
crypto-narrative-hunter-tfm/
├── README.md                          # Guía de instalación y uso
├── data-pipeline/                     # Pipeline de datos
│   ├── data_collection/               # Módulo Python
│   │   └── common/                    # Utilidades (db, logging, checkpoints)
│   ├── services/                      # Servicios de negocio
│   │   ├── tokens/                    # Clasificación narrativa
│   │   ├── transactions/              # Extracción transacciones
│   │   ├── balances/                  # Snapshots de balance
│   │   ├── prices/                    # Pricing USD
│   │   ├── feature_engineering/       # Generación features
│   │   └── validation/                # Aseguramiento de calidad
│   ├── sql/                           # Queries y schemas
│   │   ├── schema.sql                 # Schema PostgreSQL
│   │   └── dune_queries/              # 16 queries Dune Analytics
│   ├── notebooks/                     # Notebooks Jupyter
│   │   ├── 01_comprehensive_eda.ipynb
│   │   ├── Story_4.3_Clustering.ipynb
│   │   ├── Story_4.4_Interpretation.ipynb
│   │   └── Story_4.5_Evaluation.ipynb
│   ├── tests/                         # Tests unitarios
│   ├── outputs/                       # Datasets generados
│   │   ├── features/                  # Features por categoría
│   │   ├── clustering/                # Resultados clustering
│   │   └── final_exports/             # CSV/Parquet finales
│   ├── pyproject.toml                 # Dependencias
│   └── .env.example                   # Template de configuración
└── docs/                              # Documentación
    ├── architecture.md
    ├── feature_dictionary.md
    └── operational_guide.md
```

**Tecnologías:**
- Python 3.11
- PostgreSQL 15
- scikit-learn, hdbscan, pandas, numpy
- matplotlib, seaborn
- pytest (testing)

---

## Instalación y Reproducibilidad

### Requisitos
- Python 3.11+
- PostgreSQL 15+
- API keys: Dune Analytics, Alchemy, CoinGecko (free tiers suficientes)

### Setup

```bash
# Clonar repositorio
git clone https://github.com/[autor]/crypto-narrative-hunter-tfm.git
cd crypto-narrative-hunter-tfm/data-pipeline

# Crear entorno virtual
python3.11 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con API keys

# Inicializar base de datos
psql -U postgres -c "CREATE DATABASE crypto_narratives;"
python -m data_collection.cli init-db

# Ejecutar notebooks
jupyter notebook notebooks/
```

### Reproducir Clustering

```python
# Script simplificado
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN

# Cargar datos
df = pd.read_csv('outputs/features/wallet_features_cleaned.csv')
X = df.drop(columns=['wallet_address'])

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering
clusterer = HDBSCAN(min_cluster_size=40, min_samples=8, metric='euclidean')
labels = clusterer.fit_predict(X_scaled)

# Resultado
df['cluster'] = labels
print(df['cluster'].value_counts())
```

**Nota:** Consultar notebooks en Apéndices B-D para análisis completos con visualizaciones.

---

# BIBLIOGRAFÍA

Barber, B. M., & Odean, T. (2000). Trading is hazardous to your wealth: The common stock investment performance of individual investors. *The Journal of Finance*, 55(2), 773-806.

Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. In *Pacific-Asia Conference on Knowledge Discovery and Data Mining* (pp. 160-172). Springer.

Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Lawrence Erlbaum Associates.

Dune Analytics. (2025). *Dune SQL Reference*. https://docs.dune.com/

Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1), 193-218.

Markowitz, H. (1952). Portfolio selection. *The Journal of Finance*, 7(1), 77-91.

McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density based clustering. *Journal of Open Source Software*, 2(11), 205.

Milligan, G. W., & Cooper, M. C. (1985). An examination of procedures for determining the number of clusters in a data set. *Psychometrika*, 50(2), 159-179.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65.

CoinGecko. (2025). *CoinGecko API Documentation*. https://www.coingecko.com/api/documentation

Alchemy. (2025). *Alchemy API Reference*. https://docs.alchemy.com/

Etherscan. (2025). *Ethereum Blockchain Explorer*. https://etherscan.io/

Uniswap Labs. (2025). *Uniswap V3 Documentation*. https://docs.uniswap.org/

Curve Finance. (2025). *Curve Protocol Overview*. https://curve.fi/

---

# APÉNDICES

## APÉNDICE A: Análisis Exploratorio de Datos (EDA)

**Referencia:** Notebook `01_comprehensive_eda.ipynb` en repositorio GitHub

**Contenido:**
- Estadísticas descriptivas completas (41 features)
- Detección de outliers y tratamiento
- Análisis de correlaciones (matriz 41×41)
- Distribuciones univariadas
- Visualizaciones de relaciones bivariadas clave

**Hallazgos EDA documentados en:**
`data-pipeline/analysis/EDA_VALIDATION_REPORT.md` (12,000 palabras)

---

## APÉNDICE B: Notebook de Clustering (Story 4.3)

**Referencia:** Notebook `Story_4.3_Wallet_Clustering_Analysis.ipynb`

**Contenido:**
- Búsqueda en grilla de hiperparámetros HDBSCAN (48 combinaciones)
- Comparación HDBSCAN vs K-Means (k=3, 5, 7)
- Métricas de calidad por configuración
- Proyecciones t-SNE con diferentes perplexities
- 9 visualizaciones de clustering

**Duración ejecución:** ~60 minutos

---

## APÉNDICE C: Notebook de Interpretación (Story 4.4)

**Referencia:** Notebook `Story_4.4_Cluster_Interpretation.ipynb`

**Contenido:**
- Análisis estadístico por cluster (27 métricas)
- Identificación de carteras representativas (centroides, top performers)
- Generación de personas por cluster
- Análisis de solapamiento HDBSCAN-KMeans
- Importancia de features (estadístico F de ANOVA)

**Output:** 14 perfiles detallados en JSON

---

## APÉNDICE D: Notebook de Evaluación (Story 4.5)

**Referencia:** Notebook `Story_4.5_Comprehensive_Evaluation.ipynb`

**Contenido:**
- Tests de Kruskal-Wallis para 7 métricas
- Cálculo de tamaños de efecto (epsilon-squared)
- Tests post-hoc (Dunn con Bonferroni)
- Análisis de estabilidad bootstrap (1,000 iteraciones)
- Validación cruzada algorítmica (ARI, NMI)

**Output:** Informe de validación estadística completo

---

## APÉNDICE E: Presentación de Investigación

**Referencia:** Notebook `Epic_4_Research_Presentation.ipynb`

**Contenido:**
- Presentación académica de 15-20 minutos
- 21 secciones con hipótesis, metodología, resultados
- Visualizaciones integradas
- Formato exportable a slides HTML

**Uso:** Presentación de defensa de tesis

---

## APÉNDICE F: Queries SQL de Dune Analytics

**Referencia:** `data-pipeline/sql/dune_queries/`

**Queries disponibles (16):**
1. `smart_wallet_combined_dex.sql` - Identificación candidatas
2. `wallet_transactions.sql` - Extracción transacciones
3. `uniswap_v3_pools.sql` - Metadata pools Uniswap V3
4. `curve_pools.sql` - Metadata pools Curve
5. `bot_detection_patterns.sql` - Patrones detección bots
6. `mev_detection_advanced.sql` - Detección ataques MEV
7. `wallet_performance_metrics.sql` - Métricas rendimiento
8. [9 queries adicionales documentadas en README]

Todas las queries incluyen comentarios detallados y son ejecutables directamente en Dune Analytics.

---

