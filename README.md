# # hidden-planets
Machine Learning dealing with data from the Kepler space telescope



TEST/DATA| RFT | LR | DL | SVC |
---------|-----|----|----|-----|
Test accuracy| 0.900| 0.884| 0.879| 0.879
Final # of Features Trained | 27 | 28 | 40 | 32 |
Training Data Score | 1.0 | 0.852 | N/A | 0.847 |
Testing Data Score | 0.893 | 0.856 | N/A | 0.837 |
Best Score | 0.892 | 0.872 | N/A | 0.882 |




# My Random Forest Tree model has the highest percentage of test accuracy at 0.900
Test accuracy: 0.894
The Random Forest Tree model 

RandomForestClassifier(n_estimators=500)

Training Data Score: 1.0
Testing Data Score: 0.8930205949656751
 rf.feature_importances_
 
 (6991, 27)
 
Training Data Score: 1.0
Testing Data Score: 0.8958810068649885


Training Data Score: 1.0
Testing Data Score: 0.8953089244851259

{'max_depth': 10, 'max_features': 'auto', 'min_samples_split': 2, 'n_estimators': 50}
0.8931941725671122

{'max_depth': 10, 'max_features': 'auto', 'min_samples_split': 2, 'n_estimators': 50}
0.8926221992591999


# My Logistic Regression model has a test accuracy percentage of 0.884

(6991, 28)
Training Data Score: 0.8521838641998856
Testing Data Score: 0.8569794050343249


print(grid.best_score_): 0.8729733515743818
print('Test accuracy: %.3f' % RFC.score(X_test_scaler, encoded_y_test)): Test accuracy: 0.884


# My Deep Learning model has a test accuracy percentage of 0.879

y_train_categorical.shape (5243, 3)

model = Sequential()
model.add(Dense(units=6, activation='relu', input_dim=40))
model.add(Dense(units=6, activation='relu'))
model.add(Dense(units=3, activation='softmax'))

model_loss, model_accuracy = model.evaluate(
    X_test_scaler, y_test_categorical, verbose=2)
print(
    f"Normal Neural Network - Loss: {model_loss}, Accuracy: {model_accuracy}")

1748/1748 - 1s - loss: 0.2965 - acc: 0.8793
Normal Neural Network - Loss: 0.29646258284625526, Accuracy: 0.8792906403541565



# My Support Vector Classifier model has a test accuracy percentage of 0.879

Training Data Score: 0.8439824527942018
Testing Data Score: 0.8415331807780321


(6991, 32) (6991,)

Training Data Score: 0.8474156017547205
Testing Data Score: 0.8375286041189931

print(grid.best_params_):  {'C': 50, 'gamma': 0.0001}
print(grid.best_score_):    0.8821285630080264

RFC = grid.best_estimator_
print('Test accuracy: %.3f' % RFC.score(X_test_scaler, y_test)):  Test accuracy: 0.879

from sklearn.feature_selection import RFE
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 3)
fit = rfe.fit(X, y)


(6991, 34) (6991,)

Training Data Score: 0.7901964524127408
Testing Data Score: 0.7814645308924485

* Create a README that reports a comparison of each model's performance as well as a summary about your findings and 
any assumptions you can make based on your model (is your model good enough to predict new exoplanets? 
Why or why not? What would make your model be better at predicting new exoplanets?).

Reporting
✓ README compares each of
the models’ performances and
predictions
✓ README summarizes the
findings and makes assumptions
based on the data and their
models.
✓ README discusses the
predictions of the possible
exoplanets with their models.
