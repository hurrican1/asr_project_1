# ASR Project — WhisperX + Pyannote (Diarization)

Этот проект выполняет:
- автоматическую транскрипцию аудио с помощью WhisperX (large-v2),
- выравнивание слов,
- диаризацию спикеров с помощью pyannote.audio 3.x,
- вывод result.txt и result.json.

Проект работает:
- локально (Mac / Linux / Windows),
- на сервере RunPod (GPU).

---

## Структура проекта

asr_project/
│
├── run_asr.py               # основной пайплайн транскрипции
├── requirements.txt         # версии библиотек
├── README.md                # документация
├── .gitignore               # исключения Git
│
├── audio/                   # сюда помещаются входные аудиофайлы
│   └── .gitkeep
│
└── result.txt / result.json # создаются после запуска

---

## Установка (локально)

### 1. Создать окружение

