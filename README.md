# MLOps 4 Tracking

Проект обучает две модели на датасете Digits и логирует результаты в ClearML:

- `model1.py` - логистическая регрессия;
- `model2.py` - решающее дерево.

## Что логируется

Для обеих моделей:

- параметры запуска: модель, `random_state`, размер тестовой выборки и гиперпараметры;
- `accuracy`;
- `f1_score`;
- `auc_roc`;
- `confusion_matrix`.

Дополнительно:

- для логистической регрессии сохраняются коэффициенты регрессии и intercept как артефакты;
- для решающего дерева логируются фактическая глубина дерева и количество листьев.

## Запуск

```bash
pip install -r requirements.txt
clearml-init
python model1.py
python model2.py
```

После запуска откройте эксперименты в ClearML WebApp и сформируйте публичные ссылки через `Share -> Create link`.
Эта опция доступна в ClearML Hosted Service: в таблице задач или на странице эксперимента нажмите `Share`, затем `Create link`.
Проект в ClearML называется `MLOps 4 Tracking Digits`.

## ClearML experiments

- Logistic Regression: добавьте сюда Share-ссылку после запуска
- Decision Tree: добавьте сюда Share-ссылку после запуска
