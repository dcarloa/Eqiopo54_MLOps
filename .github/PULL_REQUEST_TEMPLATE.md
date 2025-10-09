# 📝 Pull Request

## 🚀 Descripción
<!-- Explica brevemente qué hace este PR y por qué es necesario -->
Ejemplo: "Agrega gráfico de distribución de estudiantes en notebooks/EDA_students.ipynb"

## 🛠 Tipo de Cambio
<!-- Marca todas las opciones que apliquen -->
- [ ] ✨ feat: nueva funcionalidad o script
- [ ] 🐛 fix: corrección de errores
- [ ] 📊 data: limpieza o actualización de dataset
- [ ] 🔧 chore: tarea de mantenimiento
- [ ] ♻️ refactor: mejora de código existente
- [ ] 📚 docs: actualización de documentación

## 📂 Archivos Modificados
<!-- Lista los archivos principales modificados o agregados -->
- notebooks/EDA_students.ipynb
- scripts/modelo_base.py
- data/processed/dataset_v1.csv

## ✅ Cómo Probar
<!-- Pasos para probar tus cambios -->
1. Asegúrate de tener la rama `dev` actualizada: `git checkout dev && git pull origin dev`
2. Cambia a tu rama: `git checkout feat/nombre-rama`
3. Ejecuta los scripts o notebooks modificados
4. Verifica que no haya errores y que los resultados sean correctos

## 📝 Checklist Antes del Merge
- [ ] 🔄 Rama `dev` actualizada (`git pull origin dev`)
- [ ] ✅ Código probado y ejecuta sin errores
- [ ] 📂 Archivos grandes NO subidos al repositorio (usar `.gitignore`)
- [ ] ✍️ Commits claros y descriptivos siguiendo el formato `{TusIniciales}/tipo: mensaje`
- [ ] 📖 README actualizado si aplica
- [ ] 🧹 Notebooks sin celdas de salida innecesarias

## 🗒 Notas Adicionales
<!-- Información adicional para los revisores -->
- Revisa la coherencia de los datos y resultados
- Verifica que los cambios sean reproducibles
