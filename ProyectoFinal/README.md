# Proyecto final

## Git LFS

Este proyecto usa [Git LFS](https://git-lfs.com/) para almacenar
`data/atus_anual_2024.csv`, ya que el dataset pesa aproximadamente 98 MB. Así se
evita aumentar innecesariamente el tamaño del repositorio.

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

Si ya habías clonado el repositorio, descarga el dataset con:

```bash
git lfs pull
```
