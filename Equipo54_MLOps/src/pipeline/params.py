"""
Loader ligero para `params.yaml` usado por el pipeline.

Comportamiento:
- Si `params.yaml` existe, lo carga y devuelve un dict.
- Si no existe o falla al parsear, devuelve {} (fallback seguro).
"""
from pathlib import Path
import logging
try:
    import yaml
except Exception:  # yaml puede no estar instalado en casos raros
    yaml = None

LOGGER = logging.getLogger(__name__)


def load_params(path: str | Path | None = None) -> dict:
    """Carga params.yaml desde la ubicación dada o desde el mismo directorio del módulo.

    Returns an empty dict on error or when file is absent.
    """
    base = Path(path) if path else Path(__file__).resolve().parent / "params.yaml"
    if not base.exists():
        return {}

    if yaml is None:
        LOGGER.warning("PyYAML no disponible: no se pueden cargar params desde %s", base)
        return {}

    try:
        with open(base, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or {}
    except Exception as e:
        LOGGER.warning("No se pudo leer params.yaml (%s): %s", base, e)
        return {}
