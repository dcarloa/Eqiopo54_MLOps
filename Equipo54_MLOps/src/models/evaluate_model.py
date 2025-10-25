"""
Script para evaluar modelo entrenado

Uso:
    python src/models/evaluate_model.py models/decision_tree_model.pkl models/train_test_split.pkl models/label_encoders.pkl reports/metrics/
"""

import pandas as pd
import numpy as np
import argparse
import os
import logging
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_artifacts(model_path, splits_path, encoders_path):
    """
    Carga modelo, splits y encoders
    
    Args:
        model_path: Ruta del modelo
        splits_path: Ruta de los splits
        encoders_path: Ruta de los encoders
        
    Returns:
        tuple: (model, X_train, X_test, y_train, y_test, encoders_data)
    """
    logger.info(f"Cargando modelo desde: {model_path}")
    model = joblib.load(model_path)
    
    logger.info(f"Cargando splits desde: {splits_path}")
    splits = joblib.load(splits_path)
    X_train = splits['X_train']
    X_test = splits['X_test']
    y_train = splits['y_train']
    y_test = splits['y_test']
    
    logger.info(f"Cargando encoders desde: {encoders_path}")
    encoders_data = joblib.load(encoders_path)
    
    logger.info(f"✅ Artefactos cargados correctamente")
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    return model, X_train, X_test, y_train, y_test, encoders_data


def evaluate_model(model, X_train, X_test, y_train, y_test, le_target):
    """
    Evalúa el modelo en train y test
    
    Args:
        model: Modelo entrenado
        X_train, X_test: Features
        y_train, y_test: Target
        le_target: LabelEncoder del target
        
    Returns:
        dict: Métricas del modelo
    """
    logger.info("\n=== EVALUANDO MODELO ===")
    
    # Predicciones
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Métricas
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    logger.info(f"Accuracy (Train): {train_accuracy:.4f}")
    logger.info(f"Accuracy (Test): {test_accuracy:.4f}")
    
    # Classification report
    logger.info("\n=== CLASSIFICATION REPORT (Test) ===")
    report = classification_report(
        y_test, 
        y_pred_test, 
        target_names=le_target.classes_,
        output_dict=True
    )
    print(classification_report(
        y_test, 
        y_pred_test, 
        target_names=le_target.classes_
    ))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred_test)
    logger.info(f"\n=== MATRIZ DE CONFUSIÓN ===")
    logger.info(f"\n{cm}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\n=== TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES ===")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"{row['feature']:20s}: {row['importance']:.4f}")
    
    metrics = {
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'feature_importance': feature_importance.to_dict('records')
    }
    
    return metrics


def save_metrics(metrics, output_dir):
    """
    Guarda las métricas en archivos
    
    Args:
        metrics: Diccionario con las métricas
        output_dir: Directorio de salida
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar métricas completas en PKL
    metrics_pkl_path = os.path.join(output_dir, 'metrics.pkl')
    joblib.dump(metrics, metrics_pkl_path)
    logger.info(f"✅ Métricas PKL guardadas en: {metrics_pkl_path}")
    
    # Guardar métricas en JSON (más legible)
    metrics_json = metrics.copy()
    # Simplificar feature_importance para JSON
    metrics_json['feature_importance_top10'] = {
        row['feature']: row['importance'] 
        for row in metrics['feature_importance'][:10]
    }
    del metrics_json['feature_importance']
    
    metrics_json_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    logger.info(f"✅ Métricas JSON guardadas en: {metrics_json_path}")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Evaluar modelo entrenado'
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='Ruta del modelo entrenado (.pkl)'
    )
    parser.add_argument(
        'splits_path',
        type=str,
        help='Ruta del archivo con train/test splits (.pkl)'
    )
    parser.add_argument(
        'encoders_path',
        type=str,
        help='Ruta de los label encoders (.pkl)'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directorio donde guardar las métricas'
    )
    
    args = parser.parse_args()
    
    # Verificar archivos
    if not os.path.exists(args.model_path):
        logger.error(f"Modelo no encontrado: {args.model_path}")
        return
    if not os.path.exists(args.splits_path):
        logger.error(f"Splits no encontrados: {args.splits_path}")
        return
    if not os.path.exists(args.encoders_path):
        logger.error(f"Encoders no encontrados: {args.encoders_path}")
        return
    
    logger.info("=" * 60)
    logger.info("🚀 INICIANDO EVALUACIÓN DEL MODELO")
    logger.info("=" * 60)
    
    # 1. Cargar artefactos
    model, X_train, X_test, y_train, y_test, encoders_data = load_artifacts(
        args.model_path, args.splits_path, args.encoders_path
    )
    
    le_target = encoders_data['target_encoder']
    
    # 2. Evaluar modelo
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test, le_target)
    
    # 3. Guardar métricas
    save_metrics(metrics, args.output_dir)
    
    # Resumen final
    logger.info("\n" + "=" * 60)
    logger.info("✅ EVALUACIÓN COMPLETADA EXITOSAMENTE")
    logger.info("=" * 60)
    logger.info(f"Métricas guardadas en: {args.output_dir}")


if __name__ == '__main__':
    main()