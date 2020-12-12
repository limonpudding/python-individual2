import pandas as pd
import numpy as np
from sklearn.linear_model._stochastic_gradient import BaseSGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.datasets import make_classification
from joblib import dump


def model_GBC(train_x, train_y, test_x, test_y, n_est=100):
    model = GradientBoostingClassifier(n_estimators=n_est, learning_rate=.05)
    model.fit(train_x, train_y)
    sc = model.score(test_x, test_y)
    prediction = model.predict(test_x)
    mae = mean_absolute_error(test_y, prediction)
    return (sc, mae, prediction, model)


def model_RandomForest(train_x, train_y, test_x, test_y, n_est=100):
    model = RandomForestClassifier(n_estimators=n_est, max_depth=2, random_state=0)
    model.fit(train_x, train_y)
    sc = model.score(test_x, test_y)
    prediction = model.predict(test_x)
    mae = mean_absolute_error(test_y, prediction)
    return (sc, mae, prediction, model)


def model_CVS(train_x, train_y, test_x, test_y, n_est=100):
    model = svm.SVC()
    model.fit(train_x, train_y)
    sc = model.score(test_x, test_y)
    prediction = model.predict(test_x)
    mae = mean_absolute_error(test_y, prediction)
    return (sc, mae, prediction, model)


def model_SGD(train_x, train_y, test_x, test_y, loss, penalty, max_iter):
    model = SGDClassifier(loss=loss, penalty=penalty, max_iter=max_iter)
    model.fit(train_x, train_y)
    sc = model.score(test_x, test_y)
    prediction = model.predict(test_x)
    mae = mean_absolute_error(test_y, prediction)
    return (sc, mae, prediction, model)


def model_PassiveAggressive(train_x, train_y, test_x, test_y, n_est=100):
    model = PassiveAggressiveClassifier()
    model.fit(train_x, train_y)
    sc = model.score(test_x, test_y)
    prediction = model.predict(test_x)
    mae = mean_absolute_error(test_y, prediction)
    return (sc, mae, prediction, model)


def make_prediction(model, x, ids):
    return (model.predict(x))




def start_GBC():
    ## MODELING GBC
    print("\n\nGBC")
    max_acc = 0
    for i in range(300, 460, 20):
        (acc, mae, prediction, model) = model_GBC(train_x, train_y, test_x, test_y, i)
        if acc > max_acc:
            gbs_model = model
            max_acc = acc
        print("GBC Est =%d\t Acc = %f\t MAE = %f" % (i, acc, mae))

    print("GBC selected model with NEst = %d" % (gbs_model.n_estimators))

    bad_cols = [col for col in predictors.columns if np.var(predictors[col]) < .2]
    print(bad_cols)

    new_train = train_x.drop(bad_cols, axis=1)
    new_valid = test_x.drop(bad_cols, axis=1)
    # new_test = test_predictors.drop(bad_cols, axis=1)

    print("\n\nGBC after removing")
    max_acc = 0
    for i in range(100, 500, 20):
        (acc, mae, prediction, model) = model_GBC(new_train, train_y, new_valid, test_y, i)
        if acc > max_acc:
            gbs_model = model
            max_acc = acc
        print("GBC Est =%d\t Acc = %f\t MAE = %f" % (i, acc, mae))

    print("GBC selected model with NEst = %d" % (gbs_model.n_estimators))


def start_RandomForest():
    ## MODELING RandomForest
    print("\n\nRandomForest")
    max_acc = 0
    est = 0
    for i in range(100, 460, 20):
        (acc, mae, prediction, model) = model_RandomForest(train_x, train_y, test_x, test_y, i)
        if acc > max_acc:
            gbs_model = model
            max_acc = acc
            est = i
        print("RandomForest Est =%d\t Acc = %f\t MAE = %f" % (est, acc, mae))

    print("RandomForest selected model with NEst = %d" % (gbs_model.n_estimators))

    # bad_cols = [col for col in predictors.columns if np.var(predictors[col]) < .2]
    # print(bad_cols)
    #
    # new_train = train_x.drop(bad_cols, axis=1)
    # new_valid = test_x.drop(bad_cols, axis=1)
    # # new_test = test_predictors.drop(bad_cols, axis=1)
    #
    # print("\n\nRandomForest after removing")
    # (acc, mae, prediction, model) = model_RandomForest(new_train, train_y, new_valid, test_y, est)
    # if acc > max_acc:
    #     gbs_model = model
    #     max_acc = acc
    # print("RandomForest Est =%d\t Acc = %f\t MAE = %f" % (est, acc, mae))
    #
    # print("RandomForest selected model with NEst = %d" % (gbs_model.n_estimators))
    return gbs_model


def start_CVS():
    ## MODELING CVS
    print("\n\nCVS")
    max_acc = 0
    for i in range(300, 460, 20):
        (acc, mae, prediction, model) = model_CVS(train_x, train_y, test_x, test_y, i)
        if acc > max_acc:
            gbs_model = model
            max_acc = acc
        print("CVS Est =%d\t Acc = %f\t MAE = %f" % (i, acc, mae))
    return gbs_model


def start_SGD():
    ## MODELING SGD
    print("\n\nSGD")
    max_acc = 0
    params = []
    for loss in ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber',
                 'epsilon_insensitive', 'squared_epsilon_insensitive']:
        for penalty in ['l1', 'l2', 'elasticnet']:
            for max_iter in range(1, 100):
                (acc, mae, prediction, model) = model_SGD(train_x, train_y, test_x, test_y, loss, penalty, max_iter)
                if acc > max_acc:
                    gbs_model = model
                    max_acc = acc
                    params = loss, penalty, max_iter
    (acc, mae, prediction, model) = model_SGD(train_x, train_y, test_x, test_y, params[0], params[1], params[2])
    print(f"SGD \t Loss = {params[0]}\t Penalty = {params[1]}\t Max_iter = {params[2]}\t Acc = {acc}\t MAE = {mae}")
    return model


def start_PassiveAggressive():
    ## MODELING PassiveAggressive
    print("\n\nPassiveAggressive")
    max_acc = 0
    for i in range(100, 500, 30):
        (acc, mae, prediction, model) = model_PassiveAggressive(train_x, train_y, test_x, test_y, i)
        if acc > max_acc:
            gbs_model = model
            max_acc = acc
        print("PassiveAggressive Est =%d\t Acc = %f\t MAE = %f" % (i, acc, mae))

    return gbs_model



np.random.seed(0)

training = pd.read_csv("data.csv")

y = training.Target
predictors = training.drop(['ID', 'Target'], axis=1)
print(predictors.columns)


train_x, test_x, train_y, test_y = train_test_split(predictors, y, random_state=0)

# start_GBC()
model = start_RandomForest()
# model = start_CVS()
# model = start_SGD()
# model = start_PassiveAggressive()

dump(model, 'models/RandomForest.joblib')
# dump(model, 'models/CVS.joblib')
# dump(model, 'models/SGD.joblib')
# dump(model, 'models/PassiveAggressive.joblib')