# Especificación del pipeline

Etapas:
1. load_data: lectura de CSV de `src/data/raw`
2. clean_data: limpieza y guardado en `src/data/processed`
3. build_features: generar features y guardar en `src/data/features`
4. train_model: entrenar y guardar modelo en `src/models`
5. eval_model: evaluar y guardar métricas en `src/reports/metrics.json`

Entradas:
- CSV raw en `src/data`

Salidas:
- CSV processed
- Modelo serializado
- Métricas JSON

Orden: seguir el listado de etapas.
