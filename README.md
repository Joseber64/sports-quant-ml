# 🧠 Sports Quant ML — Automated Football Prediction Pipeline  
### *Machine Learning + Live Odds API + Poisson + Kelly + Telegram Alerts*

---

## 📌 Resumen  
Sports Quant ML es un pipeline **profesional y completamente automatizado** para predicción de fútbol.  
Integra datos en vivo, modelos de Machine Learning, métricas avanzadas y alertas automáticas a Telegram.

Incluye:

- Descarga de **cuotas en vivo** desde *The Odds API*  
- Entrenamiento automático con **xG, xGA, xG_diff**  
- Predicciones con **probabilidades reales**, **Poisson**, **Kelly Criterion**  
- Envío automático de picks a **Telegram**  
- CI/CD completo con **GitHub Actions**  
- Versionado automático de modelos con timestamp  

---

# 🚀 Características principales

### ✔️ Datos en vivo (Odds API)
El pipeline descarga automáticamente las cuotas más recientes de partidos de fútbol.

### ✔️ Entrenamiento ML profesional
- Regresión logística con:
  - Goles
  - Tiros
  - Tiros a puerta
  - Tarjetas
  - Corners
  - **xG**
  - **xGA**
  - **xG_diff**
- Imputación automática de valores faltantes
- Guardado de modelos con timestamp para evitar conflictos

### ✔️ Predicciones avanzadas
Cada partido incluye:
- Probabilidad de victoria local/visitante  
- Modelo Poisson para goles esperados  
- Kelly Criterion para gestión de banca  
- Exportación automática a CSV  

### ✔️ Alertas automáticas a Telegram
Cada ejecución envía los picks generados directamente a tu cuenta de Telegram.

### ✔️ CI/CD profesional
GitHub Actions ejecuta automáticamente:
1. Descarga de datos  
2. Entrenamiento  
3. Predicción  
4. Envío a Telegram  
5. Commit automático de outputs  

---

# 📂 Estructura del proyecto

