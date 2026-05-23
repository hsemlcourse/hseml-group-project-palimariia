[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Раннее предсказание сепсиса

**Студент:** Палиенко Мария Алексеевна

**Группа:** БИВ234


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи


**Задача:** Классификация. Раннее прогнозирование сепсиса у пациентов в ICU. Цель проекта — построить модель машинного обучения, предсказывающую риск развития сепсиса за несколько часов до его возникновения на основе временных клинических показателей пациентов, включая жизненные показатели, лабораторные анализы и демографические данные

**Датасет:** "Early Prediction of Sepsis from Clinical Data" Источник: https://physionet.org/content/challenge-2019/1.0.0/training/training_setB/#files-panel

**Целевая метрика:** PR-auc


## Структура репозитория
Опишите структуру проекта, сохранив при этом верхнеуровневые папки. Можно добавить новые при необходимости.
```
.
├── data                        # Исходные и обработанные данные
├── models                      # Сохранённые модели 
├── notebooks
│   ├── main.ipynb              # Использование модели
│   ├── preprocessing.ipynb     # Подготовка данных
│   └── experiments.ipynb       # Эксперименты 
├── report
│   ├── img                     # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── requirements.txt
├── Dockerfile
└── README.md
```
## Запуск

### 1. Клонировать репозиторий
```
git clone <https://github.com/hsemlcourse/hseml-group-project-palimariia>
cd <palimariia>
```

### 2. Создать виртуальное окружение
```
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 3. Установить зависимости
```
pip install -r requirements.txt
```

## Результаты

| Модель | PR-AUC (Test) | F1 (Test) | Примечание |
|--------|---------------|-----------|------------|
| Baseline (LogReg, сырые данные) | 0.022 | 0.051 | Без масштабирования и балансировки |
| Лучшая модель (Weighted Ensemble) | **0.096** | **0.164** | Ансамбль XGBoost + CatBoost + HistGB |

*Основная метрика — PR-AUC. Лучшая модель улучшила показатель более чем в 4 раза относительно baseline.*


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
