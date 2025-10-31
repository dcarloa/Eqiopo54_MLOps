"""
Script para entrenar modelo de árbol de decisión

Uso:
    python src/models/train_model.py data/processed/student_performance_clean.csv models/
"""

import pandas as pd
import numpy as np
import argparse
import os
import logging
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json
import yaml

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(data_path):
    """
    Carga y prepara los datos para entrenamiento
    
    Args:
        data_path (str): Ruta del archivo CSV procesado
        
    Returns:
        tuple: (X, y, label_encoders, le_target, feature_names)
    """
    logger.info(f"Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Datos cargados: {df.shape}")
    
    # Separar features y target
    X = df.drop('Performance', axis=1)
    y = df['Performance']
    
    logger.info(f"Features: {X.shape}")
    logger.info(f"Target: {y.shape}")
    
    # Codificar variables categóricas
    logger.info("Codificando variables categóricas...")
    label_encoders = {}
    
    for column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        label_encoders[column] = le
    
    # Codificar variable objetivo
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    logger.info(f"Clases objetivo: {le_target.classes_}")
    
    return X, y_encoded, label_encoders, le_target, list(X.columns)


def train_model(X_train, y_train, optimize=True, cfg=None, random_state=42):
    """
    Entrena un DecisionTreeClassifier leyendo hiperparámetros desde YAML (cfg) si existe.
    Retorna: (modelo, best_params, cv_score)
    """
    cfg = cfg or {}
    model_cfg = cfg.get('model', {})

    class_weight = model_cfg.get('class_weight', 'balanced')

    if optimize:
        logger.info("🔍 Optimizando hiperparámetros con GridSearchCV (desde YAML si existe)...")

        # Estimador base con random_state y class_weight del YAML
        base_est = DecisionTreeClassifier(
            random_state=model_cfg.get('random_state', random_state),
            class_weight=class_weight
        )

        # Param grid desde YAML o fallback
        param_grid = model_cfg.get('hyperparameters', {
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [10, 20, 30],
            'min_samples_leaf': [5, 10, 15],
            'criterion': ['gini', 'entropy']
        })

        gs_cfg = model_cfg.get('grid_search', {})
        cv = gs_cfg.get('cv', 5)
        scoring = gs_cfg.get('scoring', 'accuracy')
        n_jobs = gs_cfg.get('n_jobs', -1)
        verbose = gs_cfg.get('verbose', 1)

        logger.info(f"[CFG] param_grid={param_grid}, cv={cv}, scoring={scoring}, n_jobs={n_jobs}, verbose={verbose}")

        grid_search = GridSearchCV(
            estimator=base_est,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose
        )
        grid_search.fit(X_train, y_train)

        logger.info(f"✅ Mejor CV Score: {grid_search.best_score_:.4f}")
        logger.info(f"✅ Mejores parámetros: {grid_search.best_params_}")
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

    else:
        logger.info("Entrenando modelo con parámetros por defecto (desde YAML si existe)...")

        default_params = model_cfg.get('default_params', {
            "max_depth": 10,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "criterion": "gini"
        })

        logger.info(f"[CFG] default_params={default_params}, class_weight={class_weight}, random_state={model_cfg.get('random_state', random_state)}")

        model = DecisionTreeClassifier(
            random_state=model_cfg.get('random_state', random_state),
            class_weight=class_weight,
            **default_params
        )
        model.fit(X_train, y_train)
        return model, None, None


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
    logger.info(classification_report(
        y_test, 
        y_pred_test, 
        target_names=le_target.classes_
    ))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred_test)
    logger.info(f"\nMatriz de confusión:\n{cm}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\n=== TOP 5 CARACTERÍSTICAS MÁS IMPORTANTES ===")
    for idx, row in feature_importance.head(5).iterrows():
        logger.info(f"{row['feature']}: {row['importance']:.4f}")
    
    metrics = {
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'feature_importance': feature_importance.to_dict('records')
    }
    
    return metrics


def save_artifacts(model, label_encoders, le_target, feature_names, metrics, 
                   best_params, cv_score, output_dir):
    """
    Guarda modelo y artefactos
    
    Args:
        model: Modelo entrenado
        label_encoders: Encoders de features
        le_target: Encoder del target
        feature_names: Nombres de las features
        metrics: Métricas del modelo
        best_params: Mejores parámetros (si GridSearch)
        cv_score: Score de CV (si GridSearch)
        output_dir: Directorio de salida
    """
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar modelo
    model_path = os.path.join(output_dir, 'decision_tree_model.pkl')
    joblib.dump(model, model_path)
    logger.info(f"✅ Modelo guardado en: {model_path}")
    
    # Guardar encoders
    encoders_path = os.path.join(output_dir, 'label_encoders.pkl')
    joblib.dump({
        'feature_encoders': label_encoders,
        'target_encoder': le_target,
        'feature_names': feature_names
    }, encoders_path)
    logger.info(f"✅ Encoders guardados en: {encoders_path}")
    
    # Agregar parámetros a métricas
    if best_params:
        metrics['best_params'] = best_params
        metrics['cv_score'] = float(cv_score)
    
    # Guardar métricas
    metrics_path = os.path.join(output_dir, 'model_metrics.pkl')
    joblib.dump(metrics, metrics_path)
    logger.info(f"✅ Métricas guardadas en: {metrics_path}")
    
    # Guardar métricas también en JSON (más legible)
    metrics_json = metrics.copy()
    # Convertir feature_importance a formato más simple para JSON
    metrics_json['feature_importance_top10'] = {
        row['feature']: row['importance'] 
        for row in metrics['feature_importance'][:10]
    }
    del metrics_json['feature_importance']
    
    metrics_json_path = os.path.join(output_dir, 'model_metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    logger.info(f"✅ Métricas JSON guardadas en: {metrics_json_path}")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Entrenar modelo de árbol de decisión'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='src/pipeline/params.yaml',
        help='Ruta al params.yaml'
    )
    parser.add_argument(
        'data_path',
        type=str,
        help='Ruta del archivo CSV procesado'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directorio donde guardar el modelo y artefactos'
    )
    parser.add_argument(
        '--no-optimize',
        action='store_true',
        help='No optimizar hiperparámetros (más rápido)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proporción del test set (default: 0.2)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state para reproducibilidad (default: 42)'
    )
    
    args = parser.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
    else:
        logger.warning(f"No se encontró archivo de configuración: {args.config}")

    logger.info(f"[CFG] Cargado desde: {args.config}")
    logger.info(f"[CFG] model.test_size={cfg.get('model', {}).get('test_size')}")
    logger.info(f"[CFG] model.random_state={cfg.get('model', {}).get('random_state')}")
    logger.info(f"[CFG] model.stratify={cfg.get('model', {}).get('stratify')}")
    logger.info(f"[CFG] model.class_weight={cfg.get('model', {}).get('class_weight')}")
    logger.info(f"[CFG] model.default_params={cfg.get('model', {}).get('default_params')}")
    logger.info(f"[CFG] model.hyperparameters={cfg.get('model', {}).get('hyperparameters')}")

    
    # Verificar que el archivo existe
    if not os.path.exists(args.data_path):
        logger.error(f"El archivo {args.data_path} no existe")
        return
    
    logger.info("=" * 60)
    logger.info("🚀 INICIANDO ENTRENAMIENTO DEL MODELO")
    logger.info("=" * 60)
    
    # 1. Cargar y preparar datos
    X, y, label_encoders, le_target, feature_names = load_and_prepare_data(args.data_path)
    
    # 2. Split train/test
    # Valores efectivos desde YAML (o fallback a CLI)
    test_size     = cfg.get('model', {}).get('test_size', args.test_size)
    random_state  = cfg.get('model', {}).get('random_state', args.random_state)
    stratify_flag = cfg.get('model', {}).get('stratify', True)
    logger.info(f"\nDividiendo datos (test_size={test_size}, random_state={random_state}, stratify={stratify_flag})...")


    # Sobrescribir parámetros con YAML si existen
    test_size = cfg.get('model', {}).get('test_size', args.test_size)
    random_state = cfg.get('model', {}).get('random_state', args.random_state)
    stratify_flag = cfg.get('model', {}).get('stratify', True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
    test_size=test_size,
    random_state=random_state,
    stratify=y if stratify_flag else None
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 3. Entrenar modelo
    optimize = not args.no_optimize
    model, best_params, cv_score = train_model(
        X_train, y_train,
        optimize=optimize,
        cfg=cfg,
        random_state=random_state
    )
    
    # 4. Evaluar modelo
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test, le_target)
    
    # 5. Guardar artefactos
    save_artifacts(
        model, label_encoders, le_target, feature_names,
        metrics, best_params, cv_score, args.output_dir
    )
    
    # Resumen final
    logger.info("\n" + "=" * 60)
    logger.info("🎯 RESUMEN FINAL")
    logger.info("=" * 60)
    logger.info(f"Accuracy (Train): {metrics['train_accuracy']:.4f}")
    logger.info(f"Accuracy (Test): {metrics['test_accuracy']:.4f}")
    if best_params:
        logger.info(f"CV Score: {cv_score:.4f}")
    logger.info("\n✅ Entrenamiento completado exitosamente")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
