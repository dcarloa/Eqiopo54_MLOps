"""
Script para hacer predicciones con el modelo entrenado

Uso:
    # Predicción de un CSV completo
    python src/models/predict_model.py models/ data/new_students.csv --output predictions.csv
    
    # Predicción individual (interactivo)
    python src/models/predict_model.py models/ --interactive
"""

import pandas as pd
import numpy as np
import argparse
import os
import logging
import joblib

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StudentPerformancePredictor:
    """Clase para hacer predicciones de rendimiento estudiantil"""
    
    def __init__(self, model_dir):
        """
        Inicializa el predictor cargando el modelo y encoders
        
        Args:
            model_dir (str): Directorio donde están los artefactos del modelo
        """
        self.model_dir = model_dir
        self.model = None
        self.encoders_data = None
        self.load_artifacts()
    
    def load_artifacts(self):
        """Carga el modelo y encoders"""
        logger.info(f"Cargando artefactos desde: {self.model_dir}")
        
        # Cargar modelo
        model_path = os.path.join(self.model_dir, 'decision_tree_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        
        self.model = joblib.load(model_path)
        logger.info("✅ Modelo cargado")
        
        # Cargar encoders
        encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
        if not os.path.exists(encoders_path):
            raise FileNotFoundError(f"Encoders no encontrados en: {encoders_path}")
        
        self.encoders_data = joblib.load(encoders_path)
        logger.info("✅ Encoders cargados")
        logger.info(f"Features esperadas: {self.encoders_data['feature_names']}")
    
    def preprocess_input(self, df):
        """
        Preprocesa los datos de entrada
        
        Args:
            df (pd.DataFrame): DataFrame con los datos a predecir
            
        Returns:
            pd.DataFrame: DataFrame preprocesado
        """
        df_processed = df.copy()
        
        # Limpiar texto (mismo proceso que en entrenamiento)
        text_columns = df_processed.select_dtypes(include=['object']).columns
        for col in text_columns:
            df_processed[col] = df_processed[col].apply(
                lambda x: str(x).strip().upper() if pd.notna(x) else x
            )
        
        # Verificar que todas las features necesarias están presentes
        required_features = self.encoders_data['feature_names']
        missing_features = set(required_features) - set(df_processed.columns)
        
        if missing_features:
            raise ValueError(f"Faltan las siguientes features: {missing_features}")
        
        # Seleccionar solo las features necesarias en el orden correcto
        df_processed = df_processed[required_features]
        
        # Codificar variables categóricas
        feature_encoders = self.encoders_data['feature_encoders']
        
        for column in df_processed.columns:
            if column in feature_encoders:
                le = feature_encoders[column]
                # Manejar valores no vistos en entrenamiento
                try:
                    df_processed[column] = le.transform(df_processed[column].astype(str))
                except ValueError as e:
                    logger.warning(f"Valores no vistos en {column}, usando valor por defecto")
                    # Asignar el primer valor del encoder como default
                    df_processed[column] = 0
        
        return df_processed
    
    def predict(self, df):
        """
        Hace predicciones
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            tuple: (predicciones, probabilidades)
        """
        # Preprocesar
        X = self.preprocess_input(df)
        
        # Predecir
        predictions_encoded = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Decodificar predicciones
        le_target = self.encoders_data['target_encoder']
        predictions = le_target.inverse_transform(predictions_encoded)
        
        return predictions, probabilities
    
    def predict_with_details(self, df):
        """
        Hace predicciones con detalles completos
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            pd.DataFrame: DataFrame con predicciones y probabilidades
        """
        predictions, probabilities = self.predict(df)
        
        # Crear DataFrame de resultados
        results = df.copy()
        results['Predicted_Performance'] = predictions
        
        # Agregar probabilidades para cada clase
        le_target = self.encoders_data['target_encoder']
        for idx, class_name in enumerate(le_target.classes_):
            results[f'Prob_{class_name}'] = probabilities[:, idx]
        
        # Agregar confianza (máxima probabilidad)
        results['Confidence'] = probabilities.max(axis=1)
        
        return results


def predict_from_csv(predictor, input_path, output_path):
    """
    Hace predicciones desde un archivo CSV
    
    Args:
        predictor: Instancia de StudentPerformancePredictor
        input_path (str): Ruta del CSV de entrada
        output_path (str): Ruta donde guardar las predicciones
    """
    logger.info(f"Cargando datos desde: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Datos cargados: {df.shape}")
    
    logger.info("Haciendo predicciones...")
    results = predictor.predict_with_details(df)
    
    # Guardar resultados
    results.to_csv(output_path, index=False)
    logger.info(f"✅ Predicciones guardadas en: {output_path}")
    
    # Mostrar resumen
    logger.info("\n=== RESUMEN DE PREDICCIONES ===")
    logger.info(f"Total de predicciones: {len(results)}")
    logger.info("\nDistribución de predicciones:")
    for performance, count in results['Predicted_Performance'].value_counts().items():
        percentage = (count / len(results)) * 100
        logger.info(f"  {performance}: {count} ({percentage:.1f}%)")
    
    logger.info(f"\nConfianza promedio: {results['Confidence'].mean():.4f}")
    logger.info(f"Confianza mínima: {results['Confidence'].min():.4f}")
    logger.info(f"Confianza máxima: {results['Confidence'].max():.4f}")


def interactive_prediction(predictor):
    """
    Modo interactivo para hacer predicciones individuales
    
    Args:
        predictor: Instancia de StudentPerformancePredictor
    """
    logger.info("\n" + "=" * 60)
    logger.info("🎓 MODO INTERACTIVO - PREDICCIÓN DE RENDIMIENTO")
    logger.info("=" * 60)
    
    # Obtener features necesarias
    features = predictor.encoders_data['feature_names']
    
    logger.info(f"\nNecesitas proporcionar {len(features)} características:")
    logger.info(", ".join(features))
    
    # Recolectar datos
    student_data = {}
    
    print("\nIngresa los datos del estudiante:\n")
    
    for feature in features:
        # Mostrar opciones válidas si están disponibles
        if feature in predictor.encoders_data['feature_encoders']:
            encoder = predictor.encoders_data['feature_encoders'][feature]
            valid_options = encoder.classes_[:10]  # Mostrar máximo 10 opciones
            print(f"\n{feature}")
            print(f"  Ejemplos válidos: {', '.join(map(str, valid_options))}")
            if len(encoder.classes_) > 10:
                print(f"  (y {len(encoder.classes_) - 10} opciones más...)")
        
        value = input(f"  → {feature}: ").strip()
        student_data[feature] = value
    
    # Crear DataFrame
    df = pd.DataFrame([student_data])
    
    # Hacer predicción
    logger.info("\n🔮 Haciendo predicción...")
    results = predictor.predict_with_details(df)
    
    # Mostrar resultados
    prediction = results['Predicted_Performance'].values[0]
    confidence = results['Confidence'].values[0]
    
    print("\n" + "=" * 60)
    print("📊 RESULTADO DE LA PREDICCIÓN")
    print("=" * 60)
    print(f"\n🎯 Rendimiento Predicho: {prediction}")
    print(f"📈 Confianza: {confidence:.2%}")
    
    print("\n📊 Probabilidades por clase:")
    le_target = predictor.encoders_data['target_encoder']
    for class_name in le_target.classes_:
        prob = results[f'Prob_{class_name}'].values[0]
        bar = "█" * int(prob * 20)
        print(f"  {class_name:15s}: {prob:.2%} {bar}")
    
    print("\n" + "=" * 60)


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Hacer predicciones de rendimiento estudiantil'
    )
    parser.add_argument(
        'model_dir',
        type=str,
        help='Directorio donde están los artefactos del modelo'
    )
    parser.add_argument(
        'input_path',
        type=str,
        nargs='?',
        help='Ruta del CSV con datos para predecir'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Ruta donde guardar las predicciones (default: predictions.csv)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Modo interactivo para predicción individual'
    )
    
    args = parser.parse_args()
    
    # Verificar que el directorio del modelo existe
    if not os.path.exists(args.model_dir):
        logger.error(f"El directorio {args.model_dir} no existe")
        return
    
    # Crear predictor
    try:
        predictor = StudentPerformancePredictor(args.model_dir)
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        return
    
    # Modo interactivo o desde archivo
    if args.interactive:
        interactive_prediction(predictor)
    elif args.input_path:
        if not os.path.exists(args.input_path):
            logger.error(f"El archivo {args.input_path} no existe")
            return
        predict_from_csv(predictor, args.input_path, args.output)
    else:
        logger.error("Debes proporcionar --interactive o un archivo CSV de entrada")
        parser.print_help()


if __name__ == '__main__':
    main()
