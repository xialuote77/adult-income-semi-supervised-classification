"""Mixed Naive Bayes with validation-selected confidence pseudo-labelling.

Portfolio revision of an individual coursework experiment. Run from repository root:
python src/adult_income.py --data-dir /path/to/Assignment1_data --output results
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

CONTINUOUS = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
CATEGORICAL = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
FEATURES = CONTINUOUS + CATEGORICAL
TARGET = 'income'
THRESHOLDS = [0, 1, 2, 3, 5, 10, 12, 15, 20]


def read_frame(path, labelled=True):
    frame = pd.read_csv(path, na_values=['?', ' ?'])
    frame.columns = frame.columns.str.strip()
    for col in frame.select_dtypes(include=['object', 'str']).columns:
        frame[col] = frame[col].str.strip()
    required = FEATURES + ([TARGET] if labelled else [])
    if not set(required).issubset(frame.columns):
        raise ValueError('Input is missing required feature/label columns.')
    if labelled:
        frame[TARGET] = frame[TARGET].str.rstrip('.')
        if not frame[TARGET].isin(['<=50K', '>50K']).all():
            raise ValueError('Invalid or missing labels.')
    return frame


class MixedNB:
    """Train-only encoding/imputation and one shared class prior."""
    def fit(self, frame, y):
        self.medians_ = frame[CONTINUOUS].median()
        if self.medians_.isna().any():
            raise ValueError('A continuous training feature has no observed values.')
        self.encoder_ = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        cats = frame[CATEGORICAL].fillna('__missing__')
        self.encoder_.fit(cats)
        self.n_categories_ = [len(c) + 1 for c in self.encoder_.categories_]
        self.gnb_ = GaussianNB().fit(self.continuous(frame), y)
        self.cnb_ = CategoricalNB(alpha=1.0, min_categories=self.n_categories_).fit(self.categorical(frame), y)
        self.classes_ = self.gnb_.classes_
        if not np.array_equal(self.classes_, self.cnb_.classes_):
            raise ValueError('Component class ordering differs.')
        self.fit_rows_ = len(frame)
        return self

    def continuous(self, frame):
        return frame[CONTINUOUS].fillna(self.medians_).to_numpy(dtype=float)

    def categorical(self, frame):
        values = self.encoder_.transform(frame[CATEGORICAL].fillna('__missing__')).astype(int)
        for j, categories in enumerate(self.encoder_.categories_):
            values[values[:, j] == -1, j] = len(categories)
        return values

    def scores(self, frame):
        # Each component already includes the prior. Remove one copy.
        return (self.gnb_.predict_log_proba(self.continuous(frame))
                + self.cnb_.predict_log_proba(self.categorical(frame))
                - np.log(self.gnb_.class_prior_))

    def predict(self, frame):
        return self.classes_[self.scores(frame).argmax(axis=1)]

    def pseudo_labels(self, frame):
        scores = self.scores(frame)
        positive = list(self.classes_).index('>50K')
        negative = list(self.classes_).index('<=50K')
        return self.classes_[scores.argmax(axis=1)], np.abs(scores[:, positive] - scores[:, negative])


def retrain(base, labelled, unlabelled, threshold):
    labels, confidence = base.pseudo_labels(unlabelled)
    selected = confidence >= threshold
    if not selected.any():
        return base, 0
    combined = pd.concat([labelled[FEATURES], unlabelled.loc[selected, FEATURES]], ignore_index=True)
    y = np.concatenate([labelled[TARGET].to_numpy(), labels[selected]])
    return MixedNB().fit(combined, y), int(selected.sum())


def select_threshold(train_main, validation, unlabelled):
    base = MixedNB().fit(train_main, train_main[TARGET])
    rows = []
    for threshold in THRESHOLDS:
        model, selected = retrain(base, train_main, unlabelled, threshold)
        predicted = model.predict(validation)
        rows.append({'threshold': threshold, 'pseudo_rows': selected,
                     'validation_f1_positive': f1_score(validation[TARGET], predicted, pos_label='>50K')})
    scores = pd.DataFrame(rows)
    # Stable tie rule: first threshold in the fixed ascending grid.
    best = int(scores.loc[scores.validation_f1_positive.idxmax(), 'threshold'])
    return best, scores, base


def metrics(name, truth, predicted):
    return {'model': name, 'accuracy': accuracy_score(truth, predicted),
            'precision_positive': precision_score(truth, predicted, pos_label='>50K', zero_division=0),
            'recall_positive': recall_score(truth, predicted, pos_label='>50K'),
            'f1_positive': f1_score(truth, predicted, pos_label='>50K'),
            'macro_f1': f1_score(truth, predicted, average='macro'), 'test_rows': len(truth)}


def check_model(model, frame):
    # Check the combination independently using learned densities and one prior.
    sample = frame.iloc[:8].copy()
    sample.loc[sample.index[0], CATEGORICAL[0]] = '__unseen_category__'
    continuous = model.continuous(sample)
    encoded = model.categorical(sample)
    manual = np.log(model.gnb_.class_prior_)[None, :].copy()
    manual = np.repeat(manual, len(sample), axis=0)
    for i in range(len(model.classes_)):
        manual[:, i] += (-.5 * (np.log(2*np.pi*model.gnb_.var_[i])
                         + (continuous-model.gnb_.theta_[i])**2/model.gnb_.var_[i])).sum(axis=1)
        for j in range(len(CATEGORICAL)):
            manual[:, i] += model.cnb_.feature_log_prob_[j][i, encoded[:, j]]
    scores = model.scores(sample)
    assert np.allclose(scores[:, 1]-scores[:, 0], manual[:, 1]-manual[:, 0])
    assert np.isfinite(scores).all()
    before = model.medians_.copy()
    sample.loc[sample.index[0], CONTINUOUS] = np.nan
    assert np.allclose(model.continuous(sample)[0], before.to_numpy())
    assert model.medians_.equals(before)


def run(data_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    paths = {k: data_dir/v for k, v in {'train': 'adult_supervised_train.csv',
             'unlabelled': 'adult_unlabelled.csv', 'test': 'adult_test.csv'}.items()}
    train_raw = read_frame(paths['train'])
    unlabelled_raw = read_frame(paths['unlabelled'], labelled=False)
    # Keep the original complete-case training policy. Test rows are all retained.
    train = train_raw.dropna(subset=FEATURES).copy()
    unlabelled = unlabelled_raw.dropna(subset=FEATURES).copy()
    main, validation = train_test_split(train, test_size=.2, random_state=42, stratify=train[TARGET])
    assert set(main.index).isdisjoint(validation.index)
    threshold, search, submodel = select_threshold(main, validation, unlabelled)
    assert submodel.fit_rows_ == len(main)
    assert np.allclose(submodel.medians_, main[CONTINUOUS].median())
    search.to_csv(output/'threshold-search.csv', index=False)
    supervised = MixedNB().fit(train, train[TARGET])
    semi, selected = retrain(supervised, train, unlabelled, threshold)
    assert semi.fit_rows_ == len(train)+selected
    # Labels in the optional unlabelled-with-label file are never loaded.
    # Read and evaluate the supplied test only after fixing the threshold and models.
    test = read_frame(paths['test'])
    check_model(supervised, test)
    results = []
    for name, model in [('Supervised', supervised), ('Pseudo-labelled', semi)]:
        predicted = model.predict(test)
        results.append(metrics(name, test[TARGET], predicted))
        cm = confusion_matrix(test[TARGET], predicted, labels=['<=50K', '>50K'])
        pd.DataFrame(cm, index=['actual_<=50K', 'actual_>50K'], columns=['predicted_<=50K', 'predicted_>50K']).to_csv(output/(name.lower()+'-confusion.csv'))
    result = pd.DataFrame(results)
    result.to_csv(output/'test-metrics.csv', index=False)
    pseudo_y, confidence = supervised.pseudo_labels(unlabelled)
    pseudo_counts = pd.Series(pseudo_y[confidence>=threshold]).value_counts().rename_axis('pseudo_label').reset_index(name='rows')
    pseudo_counts.to_csv(output/'pseudo-label-counts.csv', index=False)
    record = {'seed': 42, 'validation_fraction': .2, 'threshold_grid': THRESHOLDS,
              'selected_threshold': threshold, 'selected_pseudo_rows': selected,
              'train_raw_rows': len(train_raw), 'train_complete_rows': len(train),
              'unlabelled_raw_rows': len(unlabelled_raw), 'unlabelled_complete_rows': len(unlabelled),
              'train_main_rows': len(main), 'validation_rows': len(validation), 'test_rows': len(test),
              'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
              'scikit_learn': sklearn.__version__,
              'sha256': {k: hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
              'checks': {'split_row_indices_disjoint': True, 'encoding_and_medians_fit_on_training_partition': True,
                         'single_prior_matches_density_formula': True, 'unseen_categories_finite': True,
                         'missing_continuous_uses_training_median': True, 'combined_training_count': True}}
    (output/'run-record.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf8')
    return result, search, record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    result, search, record = run(args.data_dir, args.output)
    print(result.round(4).to_string(index=False))
    print('Selected threshold:', record['selected_threshold'], '| pseudo-label rows:', record['selected_pseudo_rows'])
