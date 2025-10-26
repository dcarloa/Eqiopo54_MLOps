from setuptools import find_packages, setup

def load_requirements(path):
    """Load only production dependencies (ignore dev/test lines)."""
    requirements = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comments, empty lines, editable installs, or dev tools
            if not line or line.startswith("#") or line.startswith("-e"):
                continue
            if any(dev_kw in line for dev_kw in [
                "Sphinx", "pytest", "flake8", "black", "jupyter", "ipykernel", "coverage", "plotly"
            ]):
                continue
            requirements.append(line)
    return requirements


setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="Proyecto integrador del equipo 54",
    author="Equipo 54",
    license="MIT",
    install_requires=load_requirements("requirements.txt"),
)
