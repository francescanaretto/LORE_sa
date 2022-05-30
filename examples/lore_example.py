import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from lore_sa.util import record2str, neuclidean
from lore_sa.datamanager import prepare_adult_dataset, prepare_dataset

# Load a dataset
def main():
    df, class_name = prepare_adult_dataset('../datasets/adult.csv')
    df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map = prepare_dataset(
        df, class_name, encdec='onehot')

    print(df.shape)
    print(df.head(1))

    # Prepare data for learning method

    test_size = 0.30
    random_state = 0

    X_train, X_test, Y_train, Y_test = train_test_split(df[feature_names].values, df[class_name].values,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=df[class_name].values)

    _, K, _, _ = train_test_split(rdf[real_feature_names].values, rdf[class_name].values,
                                  test_size=test_size,
                                  random_state=random_state,
                                  stratify=df[class_name].values)


    # Train a random forest model

    bb = RandomForestClassifier(n_estimators=20, random_state=random_state)
    # bb = MLPClassifier(random_state=random_state)
    bb.fit(X_train, Y_train)

    # encapulating predition methods in functions
    def bb_predict(X):
        return bb.predict(X)

    def bb_predict_proba(X):
        return bb.predict_proba(X)


    # example of predictions
    Y_pred = bb_predict(X_test)

    print('Accuracy %.3f' % accuracy_score(Y_test, Y_pred))
    print('F1-measure %.3f' % f1_score(Y_test, Y_pred))


    i2e = 3
    x = X_test[i2e]

    print('x = %s' % record2str(x, feature_names, numeric_columns))
    print('')

    bb_outcome = bb_predict(x.reshape(1, -1))[0]
    bb_outcome_str = class_values[bb_outcome]

    print('bb(x) = { %s }' % bb_outcome_str)
    print('')


    from lore_sa.lorem import LOREM

    lore_explainer = LOREM(X_test, bb_predict, bb_predict_proba, feature_names, class_name, class_values, numeric_columns, features_map,
                           neigh_type='geneticp', categorical_use_prob=True, continuous_fun_estimation=False,
                           size=1000, ocr=0.1, random_state=random_state, ngen=10, bb_predict_proba=bb_predict_proba,
                           verbose=True, encdec=None, K_transformed=X_test)

    exp = lore_explainer.explain_instance_stable(x, samples=1000, use_weights=True, metric=neuclidean)

    print(exp)

if __name__ == '__main__':
    main()