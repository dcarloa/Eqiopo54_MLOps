"""
Script para crear features a partir de datos limpios

Uso:
    python src/features/build_features.py data/processed/student_clean.csv data/processed/student_features.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
import logging
from sklearn.preprocessing import LabelEncoder
import joblib

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_features(input_path, output_path, encoders_path='models/label_encoders.pkl'):
    """
    Transforma datos limpios en features codificadas
    
    Args:
        input_path (str): Ruta del CSV limpio
        output_path (str): Ruta donde guardar CSV con features
        encoders_path (str): Ruta donde guardar los encoders
    """
    logger.info(f"Cargando datos limpios desde: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Separar features y target
    X = df.drop('Performance', axis=1)
    y = df['Performance']
    
    logger.info(f"Features: {list(X.columns)}")
    logger.info(f"Target: {y.name}")
    
    # Codificar variables categóricas
    logger.info("Codificando variables categóricas...")
    label_encoders = {}
    
    for column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        label_encoders[column] = le
        logger.info(f"  {column}: {len(le.classes_)} clases únicas")
    
    # Codificar variable objetivo
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    logger.info(f"Clases objetivo: {list(le_target.classes_)}")
    
    # Reconstruir DataFrame con features codificadas
    df_features = X.copy()
    df_features['Performance'] = y_encoded
    
    # Guardar features
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_features.to_csv(output_path, index=False)
    logger.info(f"✅ Features guardadas en: {output_path}")
    
    # Guardar encoders
    os.makedirs(os.path.dirname(encoders_path), exist_ok=True)
    joblib.dump({
        'feature_encoders': label_encoders,
        'target_encoder': le_target,
        'feature_names': list(X.columns)
    }, encoders_path)
    logger.info(f"✅ Encoders guardados en: {encoders_path}")
    
    # Resumen
    logger.info("\n=== RESUMEN ===")
    logger.info(f"Total de filas: {len(df_features)}")
    logger.info(f"Total de features: {len(X.columns)}")
    logger.info(f"Distribución del target codificado:")
    for idx, class_name in enumerate(le_target.classes_):
        count = (y_encoded == idx).sum()
        logger.info(f"  {class_name} ({idx}): {count}")
    
    return df_features


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Generar features desde datos limpios'
    )
    parser.add_argument(
        'input_filepath',
        type=str,
        help='Ruta del archivo CSV limpio'
    )
    parser.add_argument(
        'output_filepath',
        type=str,
        help='Ruta donde guardar el CSV con features'
    )
    parser.add_argument(
        '--encoders-path',
        type=str,
        default='models/label_encoders.pkl',
        help='Ruta donde guardar los encoders (default: models/label_encoders.pkl)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_filepath):
        logger.error(f"El archivo {args.input_filepath} no existe")
        return
    
    build_features(args.input_filepath, args.output_filepath, args.encoders_path)
    logger.info("✅ Generación de features completada exitosamente")


if __name__ == '__main__':
    main()