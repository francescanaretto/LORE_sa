import pytest
from lore_sa.datamanager import prepare_adult_dataset, prepare_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score


def test_can_load_a_dataframe():
    df, class_name = prepare_adult_dataset('./datasets/adult.csv')
    df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map = prepare_dataset(
        df, class_name, encdec='onehot')

    assert df.shape[0] > 1, "The dataframe should contain at least 2 rows"
    assert len(feature_names) > 1, "The number of transformed features should be greater than 2"
    assert len(real_feature_names) > 1, "The number of real features should be greater than 2"
    assert len(feature_names) >= len(real_feature_names), "The number of transformed features can not be smaller " \
                                                          "than the original ones"
    assert real_feature_names == ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'workclass',
                                  'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                                  'native-country'], \
        "The features do not match with the expected list"
    assert class_values == ['<=50K', '>50K'], "The expected class values do not match"
    assert numeric_columns == ['age', 'capital-gain', 'capital-loss', 'hours-per-week'], "The list of numeric columns does not match"
    # assert features_map == {}, "The map of features to one-hot-encoding does not match"
#     it is difficult to check the whole dictionary of features_map: checking only a few selection of values

def test_learn_a_model():
    df, class_name = prepare_adult_dataset('./datasets/adult.csv')
    df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map = prepare_dataset(
        df, class_name, encdec='onehot')

    # Prepare data for learning method

    test_size = 0.30
    random_state = 0

    X_train, X_test, Y_train, Y_test = train_test_split(df[feature_names].values, df[class_name].values,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=df[class_name].values)

    assert X_train.shape[1] == 103, "There should be 103 attributes in the training set"
    assert X_test.shape[1] == 103, "There should be 103 attributes in the test set"

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
    assert Y_pred[0] == 0, "The expected class for record 0 should be 0"
    assert Y_pred[10] == 1, "The expected class for record 10 should be 1"
