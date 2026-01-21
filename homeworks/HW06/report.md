# HW06 – Report

> Файл: `homeworks/HW06/report.md`  
> Важно: не меняйте названия разделов (заголовков). Заполняйте текстом и/или вставляйте результаты.

## 1. Dataset

- Использован `S06-hw-dataset-04.csv`.
- Размер: 25000 строк, 62 столбца (включая `id` и `target`).
- Целевая переменная `target`: класс 0 — 23770 (95.08%), класс 1 — 1230 (4.92%).
- Признаки: 60 числовых `f01`–`f60`, `id` исключён.

## 2. Protocol

- Разделение train/test = 80/20, `random_state=42`, `stratify=y`.
- Подбор параметров через `GridSearchCV` только на train, 5 фолдов, оптимизация по `roc_auc`.
- Метрики: accuracy, F1 (binary), ROC-AUC; при дисбалансе ориентир на F1 и ROC-AUC.

## 3. Models

Базовые модели: DummyClassifier (`most_frequent`), LogisticRegression в Pipeline(StandardScaler).
Модели недели 6: DecisionTreeClassifier (подбор `max_depth`, `min_samples_leaf`, `ccp_alpha`), RandomForestClassifier (подбор `max_depth`, `max_features`, `min_samples_leaf`, `n_estimators=300`), GradientBoostingClassifier (подбор `n_estimators`, `learning_rate`, `max_depth`).

## 4. Results

- DummyClassifier — acc 0.9508, F1 0.0000, ROC-AUC 0.5000.
- LogisticRegression — acc 0.9632, F1 0.4286, ROC-AUC 0.8340.
- DecisionTree — acc 0.9664, F1 0.5670, ROC-AUC 0.8187.
- RandomForest — acc 0.9692, F1 0.5471, ROC-AUC 0.8944.
- GradientBoosting — acc 0.9752, F1 0.6804, ROC-AUC 0.8894.
- Победитель по CV ROC-AUC — GradientBoostingClassifier (CV 0.8876; test: acc 0.9752, F1 0.6804, ROC-AUC 0.8894).

## 5. Analysis

- Устойчивость: отдельно не проверялась (можно сделать 5 прогонов с разными `random_state`).
- Ошибки: confusion matrix для лучшей модели — TN=4744, FP=10, FN=114, TP=132; при дисбалансе важнее контролировать FN.
- Интерпретация: permutation importance (топ-15): f54, f25, f58, f13, f41, f11, f53, f33, f47, f38, f15, f43, f04, f52, f07.

## 6. Conclusion

Дерево решений требует контроля сложности, иначе происходит переобучение. Ансамбли дают более высокое качество на дисбалансных данных.
При дисбалансе ориентир — F1 и ROC-AUC, accuracy вторична.
