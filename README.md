# AirPrice

Sistema inteligente para la clasificación del rango de precio de alojamientos Airbnb mediante técnicas de Machine Learning.

## Descripción

AirPrice utiliza datos oficiales de Inside Airbnb correspondientes a Londres para analizar las características de los alojamientos y clasificarlos en cuatro rangos de precio:

- Económico
- Medio
- Alto
- Premium

## Dataset

Fuente: Inside Airbnb - London.

### Dataset original

- 92.638 registros
- 90 variables
- 92.638 IDs únicos

### Dataset limpio

- 61.617 registros
- 32 variables
- 0 registros duplicados
- 0 valores faltantes

### Distribución de las clases

- Económico: 25,14 %
- Medio: 24,93 %
- Alto: 24,94 %
- Premium: 24,99 %

## Configuración del entorno

```bash
# 1. Crear el entorno virtual
python -m venv .venv

# 2. Activarlo (Windows / PowerShell)
.\.venv\Scripts\Activate.ps1

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Registrar el kernel en Jupyter/VS Code
python -m ipykernel install --user --name airprice
```

En VS Code, selecciona el kernel `airprice` en la esquina superior derecha de cada notebook.

## Estructura del proyecto

```text
AirPrice/
├── app/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── src/
├── .gitignore
└── README.md