# Perceptron Lab

Лабораторная работа 1 по курсу Python (трек Data Science, тема А1).
Однослойный перцептрон для бинарной классификации, реализованный вручную на `numpy`.

## Что делает программа

Перцептрон получает точку с признаками `x`, считает взвешенную сумму
`z = w·x + b` и переводит её в класс ступенчатой функцией: `1`, если `z >= 0`, иначе `0`.
Обучение идёт по правилу перцептрона: при ошибке веса сдвигаются как
`w += lr · (target − pred) · x`, `b += lr · (target − pred)`.

Класс `Perceptron` (`perceptron_lab/model.py`):

| Метод | Назначение |
|---|---|
| `__init__(n_features, learning_rate=0.1, epochs=50)` | создаёт модель с нулевыми весами |
| `activate(z)` | ступенчатая функция активации |
| `predict(x)` | класс одной точки (0 или 1) |
| `train(X, y)` | обучение по эпохам с ранней остановкой; возвращает число ошибок на каждой эпохе |
| `score(X, y)` | доля правильных ответов |

Перцептрон сходится только на линейно разделимых данных: на AND он
обучается до точности 1.0, на XOR не сходится. Оба случая проверяются тестами.

## Структура

```
perceptron_lab/
    __init__.py      # экспорт Perceptron
    model.py         # класс Perceptron
tests/
    test_model.py    # тесты pytest
pyproject.toml       # зависимости и настройки ruff, mypy, pytest
uv.lock              # точные версии зависимостей
```

## Установка

Нужен [uv](https://docs.astral.sh/uv/). Окружение и зависимости ставятся одной командой:

```bash
uv sync
```

## Пример использования

```python
import numpy as np
from perceptron_lab import Perceptron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([0, 0, 0, 1], dtype=np.int64)  # логическое AND

model = Perceptron(n_features=2, learning_rate=1.0)
print(model.train(X, y))   # [2, 3, 3, 2, 1, 0]
print(model.score(X, y))   # 1.0
```

## Тесты и проверки качества

```bash
uv run pytest -v          # тесты
uv run ruff check         # линтер
uv run ruff format --check  # форматирование
uv run mypy               # проверка типов (strict)
```
