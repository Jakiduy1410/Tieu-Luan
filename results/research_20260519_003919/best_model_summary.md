# Best Tuned Model

- Preprocessing: basic
- Model: logistic_regression
- CV F1 Macro: 0.7798 +/- 0.0144
- Test F1 Macro: 0.7906
- Test F1 Spam: 0.6872
- Best Params: `{"classifier__C": 10.0, "classifier__class_weight": null, "classifier__solver": "liblinear", "tfidf__max_features": 10000, "tfidf__min_df": 1, "tfidf__ngram_range": [1, 2], "tfidf__sublinear_tf": true}`

The independent test set is used only after hyperparameter selection.
