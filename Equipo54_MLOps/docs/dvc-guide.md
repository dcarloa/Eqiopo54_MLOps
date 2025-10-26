# 📘 Data Version Control (DVC) + AWS S3 Setup

## 🎯 Descripción General

Este proyecto utiliza **[DVC](https://dvc.org/)** para el versionado de archivos de datos grandes y su sincronización con **AWS S3**. DVC funciona junto con **Git** para hacer nuestro pipeline de ML/datos reproducible, sin inflar el repositorio.

---

## 🚀 Inicio Rápido para Nuevos Miembros del Equipo

### 1️⃣ Prerrequisitos

* [Git](https://git-scm.com/)
* [DVC](https://dvc.org/doc/install)
* [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

Confirma que están instalados:

```bash
git --version
dvc --version
aws --version
```

---

### 2️⃣ Configurar Credenciales de AWS

Necesitarás acceso al bucket S3 del proyecto.
Usa una de las siguientes opciones:

#### **Opción A – Recomendada para Desarrollo Local**

Pide a tu administrador **credenciales temporales** (Access key ID + Secret access key), luego crea un archivo local `.env`:

```
# .env (no se commit a git)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxx
AWS_DEFAULT_REGION=us-east-1
```

Luego ejecuta el script de configuración:

- **Usuarios Linux**
```bash
./scripts/setup_aws.sh
```

- **Usuarios Windows**
```bash
.\scripts\setup_aws.ps1
```

Esto configurará AWS CLI automáticamente.

#### **Opción B – Usando `aws configure` manualmente**

```bash
aws configure
```

Luego ingresa:

```
AWS Access Key ID: ...
AWS Secret Access Key: ...
Default region name: us-east-1
```

---

### 3️⃣ Clonar e Inicializar el Proyecto

```bash
git clone <url-de-tu-repo>
cd <carpeta-del-proyecto>

# Inicializar DVC (si no se ha hecho)
dvc init

# Configurar remote de DVC (solo una vez por proyecto, no es necesario hacerlo ya que ya esta configurado)
dvc remote add -d myremote s3://my-dvc-study-entry-performance-data
dvc remote modify myremote region us-east-1
```

---

### 4️⃣ Descargar Datos y Reproducir el Pipeline

```bash
# Descargar código último y metadatos de DVC
git pull

# Descargar todos los archivos de datos y artefactos del modelo
dvc pull

# Reproducir todo el pipeline de ML
dvc repro
```

> 🔍 **¿Qué pasa?** DVC lee `dvc.lock` para descargar versiones exactas de todos los archivos de datos, modelos y artefactos, luego ejecuta el pipeline completo.

---

## 📁 Entendiendo los Archivos de DVC

| Archivo | Propósito | ¿Va a Git? |
|---------|-----------|------------|
| `dvc.yaml` | Definición del pipeline (etapas, comandos) | ✅ Sí |
| `dvc.lock` | **Versiones exactas** de todos los inputs/outputs | ✅ Sí |
| `params.yaml` | Hiperparámetros y configuración | ✅ Sí |
| `*.dvc` | Archivos puntero para datos raw | ✅ Sí |
| Archivos de datos raw | Archivos CSV, PKL reales | ❌ No (DVC) |
| Archivos de modelo | Modelos entrenados (.pkl) | ❌ No (DVC) |

---

## 🔄 Flujo de Trabajo Diario

### Descargar Cambios Más Recientes
```bash
git pull
dvc pull
dvc repro  # si el pipeline cambió
```

### Agregar Nuevos Datos Raw
En caso de que sean archivos que no se generen desde el pipeline es necesario manualmente agregarlos a DVC

```bash
# Para nuevos archivos de datos raw (no generados por pipeline)
dvc add ruta/a/nuevos_datos.csv
git add ruta/a/nuevos_datos.csv.dvc
git commit -m "feat: agregar nuevo dataset"
dvc push
git push
```

### Modificar Pipeline y Parámetros
```bash
# Editar params.yaml o código
dvc repro  # ejecuta solo las etapas cambiadas
git add dvc.lock params.yaml
git commit -m "feat: actualizar parámetros del modelo"
dvc push
git push
```

### Verificar wué Está Trackeado
```bash
# Ver solo archivos manejados por DVC
dvc ls --dvc-only

# Verificar estado del pipeline
dvc status

# Mostrar visualización del pipeline
dvc dag
```

---

## 🧪 Flujo de Experimentación (Mejora Futura)

### 🔬 Usando DVC Experiments para Ajuste de Hiperparámetros

```bash
# Ejecutar experimentos con diferentes parámetros
dvc exp run --set-param model.max_depth=5
dvc exp run --set-param model.max_depth=10
dvc exp run --set-param model.min_samples_split=2

# Comparar todos los experimentos
dvc exp show

# Aplicar el experimento con mejor desempeño
dvc exp apply <id-experimento>

# Commit del mejor modelo
git add .
git commit -m "feat: promover mejor modelo de experimentos"
dvc push
git push
```

### 🎯 Características de Experimentación Planeadas
- Búsqueda automática de hiperparámetros
- Dashboard de comparación de métricas
- Integración con seguimiento de experimentos
- Umbrales de desempeño del modelo

---

## 🚨 Solución de Problemas

### "Datos raw no encontrados" después de clonar
```bash
# Asegurarse que los datos raw estén seguidos con DVC
dvc add Equipo54_MLOps/src/data/raw/tus_datos.csv
git add Equipo54_MLOps/src/data/raw/tus_datos.csv.dvc
dvc push
git push
```

### Pipeline no se reproduce
```bash
# Verificar si todas las dependencias están seguidas
dvc status

# Forzar reproducción de todas las etapas
dvc repro --force
```

### Verificar qué archivos maneja DVC
```bash
# Ver SOLO archivos seguidos por DVC (no todo en el repo)
dvc ls --dvc-only
```

---

## 📋 Estructura de Carpetas y Convenciones

```
Equipo54_MLOps/
├── src/                    # Código fuente (✅ Git)
├── data/
│   ├── raw/               # Datos raw (❌ Git, ✅ DVC)
│   └── processed/         # Datos procesados (❌ Git, ✅ DVC)
├── models/                # Modelos entrenados (❌ Git, ✅ DVC)
├── reports/               # Métricas & predicciones (mixto)
│   ├── metrics.json       # Métricas pequeñas (❌ Git, ✅ DVC)
│   └── metrics.pkl        # Métricas grandes (❌ Git, ✅ DVC)
├── dvc.yaml               # Pipeline (✅ Git)
├── dvc.lock               # Bloqueo de versiones (✅ Git)
└── params.yaml            # Parámetros (✅ Git)
```

---

## 🎉 Mejores Prácticas

1. **Siempre commitea `dvc.lock`** - asegura la reproducibilidad
2. **Ejecuta `dvc pull` después de `git pull`** - obtén los datos más recientes
3. **Usa `dvc repro` en lugar de ejecutar scripts manualmente** - mantén consistencia
4. **Los datos raw van en `data/raw/`** y deben seguirse con `dvc add`
5. **Los outputs del pipeline se siguen automáticamente** en `dvc.lock`

---

## 🔮 Próximos Pasos & Roadmap

- [ ] Implementar DVC Experiments para ajuste de hiperparámetros
- [ ] Configurar seguimiento automatizado de experimentos
- [ ] Crear pipeline CI/CD con comparación de experimentos
- [ ] Agregar monitoreo de desempeño del modelo
- [ ] Implementar etapas de validación de datos

