"""
Script para procesar y limpiar datos de rendimiento estudiantil

Uso:
    python src/data/make_dataset.py data/raw/student_entry_performance_modified.csv data/processed/student_performance_clean.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_text(text):
    """
    Limpia y estandariza texto
    
    Args:
        text: Texto a limpiar
        
    Returns:
        Texto limpio en mayúsculas sin espacios extra
    """
    if pd.isna(text):
        return text
    return str(text).strip().upper()


def process_data(input_path, output_path):
    """
    Procesa el dataset completo
    
    Args:
        input_path (str): Ruta del archivo CSV raw
        output_path (str): Ruta donde guardar el CSV procesado
    """
    logger.info(f"Cargando datos desde: {input_path}")
    
    # Cargar datos
    df = pd.read_csv(input_path)
    logger.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Limpiar todas las columnas de texto
    logger.info("Limpiando y estandarizando texto...")
    text_columns = df.select_dtypes(include=['object']).columns
    
    for col in text_columns:
        df[col] = df[col].apply(clean_text)
    
    # Estandarizar categorías de Performance
    logger.info("Estandarizando categorías de Performance...")
    performance_mapping = {
        'EXCELLENT': 'Excellent',
        'VG': 'Very Good',
        'GOOD': 'Good',
        'AVERAGE': 'Average'
    }
    
    df['Performance'] = df['Performance'].map(performance_mapping)
    
    # Eliminar columna mixed_type_col si existe (ruido)
    if 'mixed_type_col' in df.columns:
        logger.info("Eliminando columna 'mixed_type_col'")
        df = df.drop('mixed_type_col', axis=1)
    
    # Manejar valores nulos
    null_count_before = df.isnull().sum().sum()
    logger.info(f"Valores nulos antes: {null_count_before}")
    
    df = df.dropna()
    
    null_count_after = df.isnull().sum().sum()
    logger.info(f"Valores nulos después: {null_count_after}")
    logger.info(f"Filas restantes: {len(df)}")
    
    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Guardar datos procesados
    df.to_csv(output_path, index=False)
    logger.info(f"Datos procesados guardados en: {output_path}")
    
    # Resumen
    logger.info("\n=== RESUMEN DEL PROCESAMIENTO ===")
    logger.info(f"Filas procesadas: {len(df)}")
    logger.info(f"Columnas: {df.shape[1]}")
    logger.info(f"Distribución de Performance:")
    for category, count in df['Performance'].value_counts().items():
        logger.info(f"  - {category}: {count}")
    
    return df


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Procesar datos de rendimiento estudiantil'
    )
    parser.add_argument(
        'input_filepath',
        type=str,
        help='Ruta del archivo CSV raw'
    )
    parser.add_argument(
        'output_filepath',
        type=str,
        help='Ruta donde guardar el CSV procesado'
    )
    
    args = parser.parse_args()
    
    # Verificar que el archivo de entrada existe
    if not os.path.exists(args.input_filepath):
        logger.error(f"El archivo {args.input_filepath} no existe")
        return
    
    # Procesar datos
    process_data(args.input_filepath, args.output_filepath)
    logger.info("✅ Procesamiento completado exitosamente")


if __name__ == '__main__':
    main()
