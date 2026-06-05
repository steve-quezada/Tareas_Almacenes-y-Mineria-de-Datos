# Proyecto final

Proyecto final de Almacenes y Minería de Datos sobre accidentes de tránsito
terrestre en zonas urbanas y suburbanas de México, usando datos de ATUS 2024.

## Requisitos

Para reproducir el proyecto desde cero se necesita:

- Git
- Git LFS
- Python 3.10 o superior
- Quarto


En Linux, Quarto debe estar disponible como comando:

```bash
quarto --version
```

## Git LFS

Este proyecto usa [Git LFS](https://git-lfs.com/) para almacenar
`data/atus_anual_2024.csv` ya que es un
archivo grande. Así se evita aumentar innecesariamente el tamaño del
repositorio.

### Instalación

Instala Git LFS según tu sistema operativo:

```bash
# Fedora
sudo dnf install git-lfs

# Ubuntu o Debian
sudo apt install git-lfs

# macOS con Homebrew
brew install git-lfs

# Windows con Chocolatey
choco install git-lfs
```

Después, ejecuta:

```bash
git lfs install
```

Si ya habías clonado el repositorio, descarga los datasets con:

```bash
git lfs pull
```

## Instalación del entorno de Python con `venv`

Desde la raíz del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En Windows PowerShell, la activación del entorno es:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Renderizar la página interactiva

Desde la raíz del proyecto, con el entorno virtual activado:

```bash
quarto render docs/pagina_interactiva
```

La salida se genera en:

```text
docs/pagina_interactiva/_site/index.html
```

## Renderizar la presentación

Desde la raíz del proyecto:

```bash
quarto render reports/presentacion
```

La salida se genera en:

```text
reports/presentacion/_site/index.html
```

## Limpieza de archivos generados

Los archivos de salida de Quarto y los entornos virtuales están ignorados por
Git.