"""
Script para generar features desde datos limpios

Uso:
    python src/features/build_features.py data/processed/student_clean.csv data/processed/student_features.csv
"""
import pandas as pd
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) < 3:
        print("Uso: python build_features.py <input_csv> <output_csv>")
        sys.exit(1)
   
    input_path = sys.argv[1]
    output_path = sys.argv[2]
   
    logger.info(f"Cargando datos limpios desde: {input_path}")
    df = pd.read_csv(input_path)
   
    # Aquí puedes añadir ingeniería de features si es necesario
    # Por ahora, simplemente copiamos el DataFrame limpio
    logger.info(f"Generando features (sin transformaciones por ahora)...")
   
    logger.info(f"Guardando features en: {output_path}")
    df.to_csv(output_path, index=False)
   
    logger.info(f"✅ Features guardadas: {df.shape[0]} filas, {df.shape[1]} columnas")

if __name__ == '__main__':
    main()
