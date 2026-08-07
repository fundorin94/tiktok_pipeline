# True Crime TikTok Pipeline

Автоматизированный конвейер, который превращает факты по реальному (закрытому,
широко задокументированному) уголовному делу в серию коротких видео формата
"Part 1, Part 2, ..." для TikTok — оригинальный сценарий, архивные
public-domain фото + AI-генерация как fallback, локальная озвучка, локальный
монтаж, метаданные, публикация в dry-run режиме.

Полный контекст решений и почему проект устроен именно так — см. историю
разработки в чате. Этот файл — снимок текущего состояния для быстрого входа.

## Архитектура: 7 агентов

```
Story Research → Script Writer → Archive Finder → Voiceover → Video Assembly → Metadata → Publisher
```

| Агент | Файл | Что делает | Стоимость |
|---|---|---|---|
| Story Research | `agents/story_research.py` | Claude + web_search/web_fetch: собирает бриф (факты, таймлайн, key_details, источники) | ~$0.15-0.2/кейс |
| Script Writer | `agents/script_writer.py` | Claude: сценарий по частям (2-2.5 мин каждая, клиффхэнгеры, `visual_query` на сцену) | ~$0.05-0.1/кейс |
| Archive Finder | `agents/archive_finder.py` | Wikimedia Commons / Internet Archive поиск PD-фото + AI-генерация (SD-Turbo, локально на GPU) для мест/объектов | $0 |
| Voiceover | `agents/voiceover.py` | Piper TTS, один WAV на сцену | $0 |
| Video Assembly | `agents/video_assembly.py` | ffmpeg: Ken Burns, субтитры по предложениям, заголовок части на первом кадре, склейка | $0 |
| Metadata | `agents/metadata.py` | Claude: заголовок/caption/хэштеги на часть | ~$0.01-0.02/кейс |
| Publisher | `agents/publisher.py` | dry-run лог публикации (реальный постинг не реализован — нет TikTok API доступа) | $0 |

Оркестрация: `orchestrator/pipeline.py` (диспетчер стадий) + `orchestrator/db.py`
(SQLite: статусы кейсов, история стадий, учёт токенов) + `orchestrator/cost.py`
(оценка расходов).

## Setup

1. `python -m venv venv`, затем `venv\Scripts\pip install -r requirements.txt`
2. Скопировать `.env.example` → `.env`, вписать `ANTHROPIC_API_KEY`
   (см. `docs/` — как получить ключ, поставить лимит трат в Console)
3. ffmpeg должен быть в PATH (`ffmpeg -version` для проверки)
4. Голосовая модель Piper — скачивается один раз в `data/voices/`
   (см. `agents/voiceover.py` / `tools/try_voice.py` для смены голоса)
5. Для AI-генерации изображений — CUDA-версия torch + diffusers
   (см. историю установки; нужна NVIDIA GPU, работает на 6GB VRAM)

## Запуск

```
venv\Scripts\python.exe run_pipeline.py --case-id <slug> --topic "<тема>" --stage all
```

Или по одной стадии: `--stage story|script|archive|voiceover|video|metadata|publish`.

После каждого прогона печатается оценка расходов (по кейсу и по всей локальной БД).

## Известные ограничения / что дальше

- **Мугшоты и фото реальных людей не публикуются автоматически** — уходят в
  `review_queue.json` / `manual_sourcing_queue.json` на ручную проверку.
  Это осознанное решение, не баг.
- **Publisher работает только в dry-run** — `PUBLISH_DRY_RUN=false` намеренно
  бросает ошибку, пока не настроен TikTok Content Posting API
  (см. `docs/tiktok_api_setup.md`).
- **Только английский язык** — брифы/сценарии/озвучка сейчас только en-US.
- Известные фиксы, которые уже внесены по ходу разработки: rate-limit на
  Wikimedia API, фильтр DjVu/огромных файлов (вешали ffmpeg), таймаут на
  ffmpeg-вызовы, ASCII-safe временные пути (проект лежит в пути с кириллицей,
  и espeak/ffmpeg как C-бинарники не поддерживают non-ASCII argv на Windows).

### Бэклог (не реализовано)

1. Выбрать финальный голос Piper (сейчас `en_US-lessac-medium` по умолчанию)
2. Больше AI-сгенерированных кадров на сцену (сейчас один кадр на всю сцену)
3. Менять визуал каждые ~3 секунды внутри сцены (сейчас держится всю сцену)
4. Ограничить сценарий до 5-6 частей по 2-3 минуты (сейчас до 13 частей по 2-2.5 мин)

## Структура данных кейса

```
data/cases/<case_id>/
  brief.json               # Story Research
  script.json               # Script Writer
  media_manifest.json       # Archive Finder: статус found/needs_review/ai_generated/unresolved на сцену
  review_queue.json         # мугшоты на ручную проверку
  manual_sourcing_queue.json # сцены с реальными людьми без найденного фото
  media/                     # скачанные/сгенерированные картинки
  audio_manifest.json       # Voiceover
  audio/                     # WAV по сценам
  video/part{N}.mp4         # Video Assembly
  metadata.json              # Metadata
  publish_log.json          # Publisher (dry-run записи)
```
