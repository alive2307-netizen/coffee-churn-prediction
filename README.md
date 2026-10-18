Прогнозирование оттока клиентов сервиса доставки кофе

Учебный проект курса Data Science (Яндекс Практикум).
Цель — построить модель бинарной классификации, которая прогнозирует вероятность оттока клиента сервиса доставки кофе Happy Beans Coffee в следующем месяце.

Содержимое репозитория

- coffee_churn_prediction.ipynb — основной ноутбук проекта:
  - постановка задачи машинного обучения;
  - EDA (описание данных, пропуски, выбросы, корреляции, мультиколлинеарность);
  - предобработка (пайплайны SimpleImputer + OneHotEncoder + RobustScaler);
  - базовая модель (DummyClassifier) и LogisticRegression;
  - генерация новых признаков и повторная кросс-валидация;
  - подбор гиперпараметров (GridSearchCV);
  - финальная модель, оценка на тесте, матрица ошибок;
  - сохранение артефактов (joblib).
- requirements.txt — зависимости проекта (создаётся в первой ячейке ноутбука).
- model_metadata.json — метаданные финальной модели (создаётся в ноутбуке).
- .gitignore — исключает .pkl-артефакты и временные файлы.

Что сделано

1. EDA: изучены 10 450 клиентов, 27 признаков; выявлен дисбаланс классов (~6% оттока); обработаны пропуски (медиана/мода), отрицательные значения, выбросы.
2. Предобработка: пайплайн ColumnTransformer (категориальные — OneHotEncoder(drop='first'), числовые — SimpleImputer(median) + RobustScaler). Без утечек: всё обучается на train.
3. Модели: DummyClassifier, LogisticRegression. Базовая — ROC-AUC 0.8469.
4. Новые признаки: rating_change, opens_per_order, promo_gap_ratio, spent_sqrt, crashes_squared, promo_engagement, spent_value_ratio. ROC-AUC вырос до 0.8520.
5. Подбор гиперпараметров: GridSearchCV (20 комбинаций). Лучшая конфигурация — C=0.01, penalty='l1', solver='liblinear'. ROC-AUC (CV) = 0.9218.
6. Финальная модель (тест):
   - ROC-AUC = 0.9357
   - Precision = 0.3662
   - Recall = 0.8254
   - F1 = 0.5073
   - Accuracy = 0.90
   - Матрица ошибок: TN=1784, FP=180, FN=22, TP=104.

Как воспроизвести

1. Установите зависимости:
   pip install -r requirements.txt
2. Откройте ноутбук и выполните Kernel → Restart & Run All.
3. В конце ноутбука создаются артефакты:
   - churn_prediction_model.pkl — итоговая модель;
   - preprocessor.pkl — пайплайн предобработки;
   - model_metadata.json — метаданные модели.

Инструменты

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (LogisticRegression, DummyClassifier, Pipeline, ColumnTransformer, GridSearchCV, RobustScaler, OneHotEncoder, SimpleImputer)
- joblib

Примечание

Данные для проекта предоставлены платформой Яндекс Практикум (coffee_churn_dataset.csv).
.pkl-артефакты не коммитятся в репозиторий (см. .gitignore) — их можно воспроизвести, запустив ноутбук.
