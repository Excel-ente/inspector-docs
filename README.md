# docid-tf

Proyecto de clasificación de documentos, OCR y extracción de campos usando TensorFlow/Keras.

## Instalación

1. Crea un entorno virtual y activa:
   ```bash
   make venv
   ```

2. Instala dependencias del sistema:
   - **Linux (apt):**
     ```bash
     sudo apt-get update && sudo apt-get install -y tesseract-ocr poppler-utils
     ```
   - **macOS (brew):**
     ```bash
     brew install tesseract poppler
     ```
   - **Windows (choco):**
     ```powershell
     choco install tesseract poppler
     ```

Opcionalmente instala PaddleOCR para usar el backend `paddle`.

## Uso

### Entrenamiento

Prepara `data/train` y `data/val` con subcarpetas por clase. Luego ejecuta:

```bash
python -m src.docid.train --epochs 12 --batch_size 16
```

Los modelos y métricas se guardan en `models/` y `out/metrics/`.

### Inferencia sobre ZIPs

Coloca los ZIPs en `zips_in/` y ejecuta:

```bash
python -m src.docid.infer_zip --zips_dir ./zips_in
```

Se generará un JSON por ZIP en `out/`.

## Buenas prácticas de PII

No subas ZIPs reales ni resultados con información sensible al repositorio. Opcionalmente enmascara valores antes de compartir los JSON.
