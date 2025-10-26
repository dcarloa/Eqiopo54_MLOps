"""
Script para entrenar modelo de árbol de decisión

Uso:
    python src/models/train_model.py data/processed/student_features.csv models/
"""

import pandas as pd
import argparse
import os
import logging
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import dvc.api

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_features(data_path):
    """
    Carga features ya procesadas y codificadas
    
    Args:
        data_path (str): Ruta del archivo CSV con features
        
    Returns:
        tuple: (X, y)
    """
    logger.info(f"Cargando features desde: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Features cargadas: {df.shape[0]} filas, {df.shape[1]} columnas")

    # Separar features y target
    X = df.drop('Performance', axis=1)
    y = df['Performance']

    logger.info(f"X: {X.shape}, y: {y.shape}")
    logger.info(f"Features: {list(X.columns)}")
    
    return X, y



def train_model(X_train, y_train, params):
    """
    Entrena el modelo de árbol de decisión

    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        params: Parámetros del modelo desde DVC

    Returns:
        tuple: (modelo, best_params, cv_score)
    """
    optimize = params['train']['optimize']

    if optimize:
        logger.info("🔍 Optimizando hiperparámetros con GridSearchCV...")

        # Use param_grid from DVC params
        param_grid = params['model']['param_grid']
        grid_search_config = params['model']['grid_search']

        grid_search = GridSearchCV(
            DecisionTreeClassifier(
                random_state=params['train']['random_state'],
                class_weight=params['model']['class_weight']
            ),
            param_grid,
            cv=grid_search_config['cv'],
            scoring=grid_search_config['scoring'],
            n_jobs=grid_search_config['n_jobs'],
            verbose=grid_search_config['verbose']
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"✅ Mejor CV Score: {grid_search.best_score_:.4f}")
        logger.info(f"✅ Mejores parámetros: {grid_search.best_params_}")

        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

    else:
        logger.info("Entrenando modelo con parámetros por defecto...")

        model_params = params['model']
        model = DecisionTreeClassifier(
            random_state=params['train']['random_state'],
            max_depth=model_params['max_depth'],
            min_samples_split=model_params['min_samples_split'],
            min_samples_leaf=model_params['min_samples_leaf'],
            criterion=model_params['criterion'],
            class_weight=model_params['class_weight']
        )

        model.fit(X_train, y_train)

        return model, None, None


def save_model_and_splits(model, X_train, X_test, y_train, y_test,
                          output_dir, best_params=None, cv_score=None,
                          encoders = None):
    """
    Guarda el modelo y los splits de datos

    Args:
        model: Modelo entrenado
        X_train, X_test, y_train, y_test: Datos de entrenamiento y prueba
        output_dir: Directorio de salida
        best_params: Mejores parámetros del GridSearch (opcional)
        cv_score: Score de validación cruzada (opcional)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Guardar modelo
    model_path = os.path.join(output_dir, 'decision_tree_model.pkl')
    joblib.dump(model, model_path)
    logger.info(f"✅ Modelo guardado en: {model_path}")

    # Guardar splits para evaluate_model.py
    splits_path = os.path.join(output_dir, 'train_test_split.pkl')
    joblib.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }, splits_path)
    logger.info(f"✅ Train/test splits guardados en: {splits_path}")
    
    # Guardar parámetros del modelo (opcional)
    if best_params is not None:
        params_path = os.path.join(output_dir, 'model_params.pkl')
        joblib.dump({
            'best_params': best_params,
            'cv_score': cv_score
        }, params_path)
        logger.info(f"✅ Parámetros guardados en: {params_path}")


def main():
    """Función principal"""

    # Load parameters from DVC
    params = dvc.api.params_show()
    logger.info("📊 DVC Parameters loaded")


    parser = argparse.ArgumentParser(
        description='Entrenar modelo de árbol de decisión'
    )
    parser.add_argument(
        'data_path',
        type=str,
        help='Ruta del archivo CSV con features procesadas'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directorio donde guardar el modelo'
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        logger.error(f"El archivo {args.data_path} no existe")
        return

    logger.info("=" * 60)
    logger.info("🚀 INICIANDO ENTRENAMIENTO DEL MODELO")
    logger.info("=" * 60)

    # 1. Cargar features
    X, y = load_features(args.data_path)

    # 2. Split train/test using DVC parameters
    test_size = params['train']['test_size']
    random_state = params['train']['random_state']

    logger.info(f"\nDividiendo datos (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 3. Entrenar modelo usando parametros de DVC
    model, best_params, cv_score = train_model(
        X_train, y_train,
        params=params
    )

    # 4. Guardar modelo y splits
    save_model_and_splits(model, X_train, X_test, y_train, y_test,
                          args.output_dir, best_params, cv_score)

    # Resumen final
    logger.info("\n" + "=" * 60)
    logger.info("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    logger.info("=" * 60)
    logger.info(f"Modelo guardado en: {args.output_dir}")
    logger.info("Para evaluar el modelo, ejecuta:")
    logger.info(f"  python src/models/evaluate_model.py {args.output_dir}/decision_tree_model.pkl {args.output_dir}/train_test_split.pkl models/label_encoders.pkl reports/metrics/")


if __name__ == '__main__':
    main()