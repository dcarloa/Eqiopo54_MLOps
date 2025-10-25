"""
Pipeline completo de Student Performance Prediction
Orquesta la ejecución de todos los scripts del proyecto

Uso:
    # Pipeline completo (rápido)
    python src/pipeline/run_pipeline.py
   
    # Pipeline con optimización de hiperparámetros
    python src/pipeline/run_pipeline.py --optimize
   
    # Pipeline desde un paso específico
    python src/pipeline/run_pipeline.py --start-from 2
"""

import subprocess
import sys
import argparse
import logging
from pathlib import Path
import os

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineRunner:
    """Ejecuta el pipeline completo llamando a los scripts individuales"""
   
    def __init__(self, optimize=False, test_size=0.2, random_state=42):
        """
        Inicializa el runner del pipeline
       
        Args:
            optimize (bool): Si True, usa GridSearch en entrenamiento
            test_size (float): Proporción del test set
            random_state (int): Semilla para reproducibilidad
        """
        self.optimize = optimize
        self.test_size = test_size
        self.random_state = random_state
       
        # Rutas de datos
        self.raw_path = 'src/data/raw/student_entry_performance_modified.csv'
        self.clean_path = 'src/data/processed/student_clean.csv'
        self.features_path = 'src/data/processed/student_features.csv'
       
        # Rutas de modelos y métricas
        self.models_dir = 'models/'
        self.metrics_dir = 'reports/metrics/'
        self.encoders_path = 'models/label_encoders.pkl'
   
    def _exists(self, script_path: str, step_name: str) -> bool:
        if not Path(script_path).exists():
            logger.warning(f"⚠️ {step_name} omitido: no se encontró {script_path}")
            return False
        return True

    def run_command(self, command, step_name):
        """
        Ejecuta un comando y maneja errores
       
        Args:
            command (list): Comando a ejecutar
            step_name (str): Nombre del paso (para logging)
        """
        logger.info(f"Ejecutando: {' '.join(command)}")
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )
            if result.stdout:
                print(result.stdout)
            logger.info(f"✅ {step_name} completado exitosamente")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error en {step_name}")
            logger.error(f"Código de salida: {e.returncode}")
            if e.stdout:
                logger.error(f"STDOUT:\n{e.stdout}")
            if e.stderr:
                logger.error(f"STDERR:\n{e.stderr}")
            # ← en vez de raise, regresamos False para que el pipeline siga
            return False
        except FileNotFoundError as e:
            logger.error(f"❌ Archivo/Script no encontrado al ejecutar {step_name}: {e}")
            return False
   
    def step_1_clean_data(self):
        """Paso 1: Limpieza de datos"""
        logger.info("=" * 70); logger.info("PASO 1: LIMPIEZA DE DATOS"); logger.info("=" * 70)
        script = 'src/data/make_dataset.py'
        if not self._exists(script, "Limpieza de datos"):
            return False
        command = ['python', script, self.raw_path, self.clean_path]
        return self.run_command(command, "Limpieza de datos")
   
    def step_2_build_features(self):
        """Paso 2: Generación de features"""
        logger.info("\n" + "=" * 70); logger.info("PASO 2: GENERACIÓN DE FEATURES"); logger.info("=" * 70)
        script = 'src/features/build_features.py'
        if not self._exists(script, "Generación de features"):
            return False
        command = ['python', script, self.clean_path, self.features_path]
        return self.run_command(command, "Generación de features")
   
    def step_3_train_model(self):
        """Paso 3: Entrenamiento del modelo"""
        logger.info("\n" + "=" * 70); logger.info("PASO 3: ENTRENAMIENTO DEL MODELO"); logger.info("=" * 70)
        script = 'src/models/train_model.py'
        if not self._exists(script, "Entrenamiento del modelo"):
            return False
        command = [
            'python', script, self.features_path, self.models_dir,
            '--test-size', str(self.test_size), '--random-state', str(self.random_state)
        ]
        if not self.optimize:
            command.append('--no-optimize')
        ok = self.run_command(command, "Entrenamiento del modelo")
        # ➜ tras entrenar, intenta mostrar métricas
        if ok:
            self._print_metrics()
        return ok
   
    def step_4_evaluate_model(self):
        """Paso 4: Evaluación del modelo (puede estar integrada en train_model.py)"""
        logger.info("\n" + "=" * 70); logger.info("PASO 4: EVALUACIÓN DEL MODELO"); logger.info("=" * 70)
        script = 'src/models/evaluate_model.py'
        if not Path(script).exists():
            logger.info("ℹ️ Evaluación integrada en train_model.py. Mostrando métricas ya generadas...")
            self._print_metrics()
            return True
        command = [
            'python', script,
            'models/decision_tree_model.pkl',
            'models/train_test_split.pkl',
            'models/label_encoders.pkl',
            self.metrics_dir
        ]
        ok = self.run_command(command, "Evaluación del modelo")
        if ok:
            self._print_metrics()
        return ok
   
    def run_full_pipeline(self, start_from=1, stop_at=4):
        """
        Ejecuta el pipeline completo o parcial
       
        Args:
            start_from (int): Paso desde el cual iniciar (1-4)
            stop_at (int): Paso en el cual detenerse (1-4)
        """
        steps = {
            1: ('Limpieza de datos', self.step_1_clean_data),
            2: ('Generación de features', self.step_2_build_features),
            3: ('Entrenamiento del modelo', self.step_3_train_model),
            4: ('Evaluación del modelo', self.step_4_evaluate_model),
        }
       
        # Validación
        if start_from < 1 or start_from > 4:
            raise ValueError("start_from debe estar entre 1 y 4")
        if stop_at < 1 or stop_at > 4:
            raise ValueError("stop_at debe estar entre 1 y 4")
        if start_from > stop_at:
            raise ValueError("start_from no puede ser mayor que stop_at")
       
        # Banner inicial
        logger.info("\n" + "=" * 70)
        logger.info("🚀 PIPELINE DE STUDENT PERFORMANCE PREDICTION")
        logger.info("=" * 70)
        logger.info(f"Configuración del pipeline:")
        logger.info(f"  • Pasos a ejecutar: {start_from} → {stop_at}")
        logger.info(f"  • Optimización: {'✅ Activada' if self.optimize else '❌ Desactivada'}")
        logger.info(f"  • Test size: {self.test_size}")
        logger.info(f"  • Random state: {self.random_state}")
        logger.info("=" * 70 + "\n")
       
        # Ejecutar pasos
        any_failed = False
        for step_num in range(start_from, stop_at + 1):
            step_name, step_func = steps[step_num]
            try:
                ok = step_func()
                if ok is False:
                    any_failed = True
                    logger.warning(f"⚠️ Paso omitido o fallido: {step_name}. Se continúa con el siguiente.")
            except Exception as e:
                any_failed = True
                logger.error(f"❌ Excepción en {step_name}: {e}")
                logger.warning("Continuando con el siguiente paso...")

        # Resumen final
        logger.info("\n" + "=" * 70)
        if any_failed:
            logger.info("🏁 PIPELINE FINALIZADO CON ADVERTENCIAS (algunos pasos fallaron/omitidos)")
        else:
            logger.info("🎉 PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("=" * 70 + "\n")

    def _print_metrics(self):
        """
        Imprime en consola las métricas si existen en models/model_metrics.json
        """
        try:
            metrics_path = Path(self.models_dir) / 'model_metrics.json'
            if not metrics_path.exists():
                logger.warning(f"⚠️ No se encontraron métricas en {metrics_path}")
                return False
            import json
            with open(metrics_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info("📊 MÉTRICAS DEL MODELO")
            # Imprime lo esencial; ajusta si quieres más campos:
            train_acc = data.get('train_accuracy')
            test_acc = data.get('test_accuracy') or data.get('accuracy')
            logger.info(f"  • train_accuracy: {train_acc}")
            logger.info(f"  • test_accuracy : {test_acc}")
            # si existe f1_macro o similares:
            f1 = data.get('f1_macro')
            if f1 is not None:
                logger.info(f"  • f1_macro      : {f1}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ No se pudieron imprimir métricas: {e}")
            return False

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Ejecutar pipeline completo de Student Performance Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
 
  # Pipeline completo (rápido, sin optimización)
  python src/pipeline/run_pipeline.py
 
  # Pipeline completo con optimización de hiperparámetros
  python src/pipeline/run_pipeline.py --optimize
 
  # Desde el paso 2 (features) hasta el final
  python src/pipeline/run_pipeline.py --start-from 2
 
  # Solo hasta el paso 3 (training)
  python src/pipeline/run_pipeline.py --stop-at 3
 
  # Solo el paso 4 (evaluación)
  python src/pipeline/run_pipeline.py --start-from 4 --stop-at 4
 
  # Pasos 2 y 3 solamente
  python src/pipeline/run_pipeline.py --start-from 2 --stop-at 3
 
  # Con test size personalizado
  python src/pipeline/run_pipeline.py --test-size 0.3

Pasos del pipeline:
  1. Limpieza de datos (make_dataset.py)
  2. Generación de features (build_features.py)
  3. Entrenamiento del modelo (train_model.py)
  4. Evaluación del modelo (evaluate_model.py)
        """
    )
   
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Activar optimización de hiperparámetros con GridSearchCV (más lento, 2-5 min)'
    )
   
    parser.add_argument(
        '--start-from',
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help='Paso desde el cual iniciar (1=data, 2=features, 3=training, 4=evaluation)'
    )
   
    parser.add_argument(
        '--stop-at',
        type=int,
        default=3,
        choices=[1, 2, 3, 4],
        help='Paso en el cual detenerse (1=data, 2=features, 3=training, 4=evaluation)'
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
   
    # Crear y ejecutar pipeline
    pipeline = PipelineRunner(
        optimize=args.optimize,
        test_size=args.test_size,
        random_state=args.random_state
    )
   
    pipeline.run_full_pipeline(
        start_from=args.start_from,
        stop_at=args.stop_at
    )


if __name__ == '__main__':
    main()
