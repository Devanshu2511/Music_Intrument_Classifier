import numpy as np
import preprocessing
import joblib
from sklearn.linear_model import LogisticRegression
import scratch_pca


def unpack_data(data):
    #unpacking data
    src_names = np.array(list(map(lambda n: n[0], data)))
    features = np.array(list(map(lambda n: n[1], data)))
    labels = np.array(list(map(lambda n: n[2], data)))

    return src_names, features, labels


def train_and_test(data, method, cv_fold=10):
    # extracting features
    print("started......")
    fold_unit = len(data) // cv_fold
    np.random.shuffle(data)
    accu_rates = []
    models = []
    for fold in range(cv_fold):
        print('start fold:', fold)
        #separating the train data and test data
        train_data = data[:fold_unit*fold] + data[fold_unit*(fold+1):]
        test_data = data[fold_unit*fold:fold_unit*(fold+1)]
        model = train(train_data, method)
        print ('training done. start testing...')
        #getting the statistics for after test data
        accu_rate = test(model, test_data, method)
        accu_rates.append(accu_rate)
        models.append(model)
    print(accu_rates)
    print('average: ', np.average(accu_rates))
    #cache the best model
    save_model(model)

    return model, np.average(accu_rates)


def train(data, method):
    src_names, features, labels = unpack_data(data)
    print('train feature vector dim:', features.shape)
    # initialize models
    params = method[0]
    #pca = decomposition.PCA(n_components=params['pca_n'])
    #pca.fit_transform(features)
    features,components = scratch_pca.scratch_fit_transform(features, n_components=params['pca_n'])
    #generating the logistic regression model and fitting it in model
    classifier = LogisticRegression(random_state=0,max_iter=2000)
    classifier.fit(features, labels)

    return components, classifier


def predict(model, features, method):
    components,classifier = model
    params = method[0]
    if 'pca' in method:
        #features = pca.transform(features)
        features = scratch_pca.scratch_transform(features,components)
        ypred = classifier.predict(features)
        return ypred
    print('error,no method specified')

    return []


def test(model, data, method):
    src_names, features, labels = unpack_data(data)
    predicted = predict(model, features, method)

    # get stats for accuracy
    test_size = src_names.shape[0]
    accuracy = (predicted == labels)
    accu_rate = np.sum(accuracy) / float(test_size)
    print (np.sum(accuracy), 'correct out of', test_size)
    print ('accuracy rate: ', accu_rate)

    # write out all the wrongly-classified samples
    wrongs = np.array([src_names, labels, predicted])
    wrongs = np.transpose(wrongs)[np.invert(accuracy)]
    with open('last_wrong.txt', 'w') as log:
        for w in wrongs:
            log.write('{} truly {} classified {}\n'.format(w[0], w[1], w[2]))

    return accu_rate


def save_model(model):           #saving the model
    pca = model
    joblib.dump(pca, 'test_pca.model')
    return


def load_model(params):         #loading the model
    pca = joblib.load('test_pca.model')
    return pca


def main():
    data = preprocessing.feature_extract()
    print('processed data.')
    model_params = {
        'pca_n': 10,
    }
    train_and_test(data, [model_params, 'pca'])
    #model = load_model(model_params)
    #test(model, data, [model_params,'pca'])


if __name__ == '__main__':
    main()
