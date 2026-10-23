# cv_experience_ocr_bert

Sistema de *document AI* que lee la **imagen** de un CV, extrae su texto con **PaddleOCR** y predice los
**años de experiencia laboral** del candidato (0–20) con un modelo **BERT ajustado como regresor**,
comparándolo contra tres baselines (mediana, TF-IDF + Ridge y embeddings + Ridge).

> 🚧 En construcción. Este README se completará con los resultados al final del proyecto.

## Estructura

```
cv_experience_ocr_bert/
├── app.py                  # App de Streamlit (4 pestañas)
├── requirements.txt        # Versiones fijadas (incluye paddlepaddle + paddleocr)
├── notebooks/              # Notebook de Colab para las etapas con GPU (OCR y BERT)
├── src/                    # Pipeline: descarga, OCR, calidad, limpieza, split, modelos, evaluación
├── data/                   # raw/ (no se sube), ocr/, processed/, splits/
├── models/                 # Pesos (no se suben)
├── output/                 # figures/, tables/, metrics/
└── video/link.txt
```

## Instalación

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate

# 1) PaddlePaddle primero, desde el índice oficial (CPU o GPU)
pip install paddlepaddle==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
# 2) Resto de dependencias
pip install -r requirements.txt
```

Las imágenes (`data/raw/`) y los pesos de los modelos (`models/`) no se suben a GitHub; se regeneran
siguiendo los pasos de reproducción (se documentarán aquí).
