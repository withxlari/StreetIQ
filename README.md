# StreetIQ

Plataforma de Computer Vision, Machine Learning e IA para análise inteligente de trânsito.

O objetivo não é apenas detectar veículos e pedestres em um vídeo, mas transformar o que é visto pela câmera em eventos estruturados e interpretações inteligentes sobre o que está acontecendo na cena.

## Status atual

Em desenvolvimento.

- [x] V0.1 — Detecção de objetos (YOLOv8 + OpenCV)
- [x] V0.2 — Tracking de objetos (ByteTrack)
- [x] V0.3 — Movement Intelligence (direção e velocidade aproximada)
- [ ] V0.4 — Event Engine
- [ ] V0.5 — Safety Intelligence
- [ ] V0.6 — Backend (FastAPI + PostgreSQL)
- [ ] V0.7 — Dashboard
- [ ] V1.0 — VisionBrain (interpretação via LLM)
- [ ] V2.0 — Predictive Intelligence

## Stack

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- ByteTrack
- PyTorch
- FastAPI (planejado)
- PostgreSQL (planejado)
- LLM (planejado)

## Como rodar

\`\`\`
pip install -r requirements.txt
python vision/detection/detect.py
\`\`\`