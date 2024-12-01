| # | Model | Parameters | Order | GridSearchScore | KaggleScore |
| - | ----- | ---------- | ----- | --------------- | ----------- |
| 1 | lr | {'fit_intercept': True} | None | -208.72944626947987 | 165.64413 |
| 1 | lr | {'fit_intercept': False} | 1 | -147.06278955513227 | 153.27953 | 
| 1 | lr | {'fit_intercept': False} | 2 | -147.06278955513227 | 153.22466 |
| 1 | knn | {'n_neighbors': 5, 'weights': 'distance'} | None | -165.6807516768186 | 138.12014 | 
| 1 | knn | {'n_neighbors': 5, 'weights': 'uniform'} | 1 | -259.29032833394825 | 176.59160 | 
| 1 | knn | {'n_neighbors': 5, 'weights': 'uniform'} | 2 | -259.29032833394825 | 176.59160 | 
| 1 | dt | {'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 4} | None | -165.6807516768186 | 133.32248 |
| 1 | dt | {'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 4} | 1 | -147.06278955513227 | 127.13037 |
| 1 | dt | {'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 4} | 2 | -147.06278955513227 | 127.13037 |
| 1 | rf | {'max_depth': 5, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 50} | None | -162.91009966752557 | 132.50509 |
| 1 | rf | {'max_depth': 5, 'min_samples_leaf': 3, 'min_samples_split': 6, 'n_estimators': 50} | 1 | -142.67609673610457 | 123.62669 |
| 1 | rf | {'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 6, 'n_estimators': 100} | 2 | -142.93404456876868 | 124.40087 |
| 1 | xgb | {'colsample_bytree': 0.7, 'learning_rate': 0.3, 'max_depth': 5, 'n_estimators': 100, 'subsample': 1} | None | -131.72206119894074 | 108.51858 |
| 1 | xgb | {'colsample_bytree': 1, 'learning_rate': 0.3, 'max_depth': 5, 'n_estimators': 100, 'subsample': 1} | 1 | -76.83551232499057 | 67.78968 |
| 1 | xgb | {'colsample_bytree': 1, 'learning_rate': 0.3, 'max_depth': 5, 'n_estimators': 100, 'subsample': 1} | 2 | -76.83551232499057 | 67.78968 |
| 2 | lr | {'copy_X': True, 'fit_intercept': False} | 1 | -192.8574671259826 | 153.27953 | 
| 2 | knn | {'n_neighbors': 5, 'weights': 'distance'} | None | -157.82608561604465 | 138.12014 |
| 2 | dt | {'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2} | 1 | -147.06278955513227 | 127.13073 |
| 2 | rf | {'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 6, 'n_estimators': 25} | 1 | -142.74319719931265 | 124.30066 |
| 2 | xgb | {'colsample_bytree': 1, 'learning_rate': 0.3, 'max_depth': 5, 'n_estimators': 100, 'subsample': 1} | 1 | -76.83551232499057 | 67.78968 |