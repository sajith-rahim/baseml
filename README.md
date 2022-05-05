## BaseML

Implementation of Base ML Algorithms :card_index_dividers:

### Structure
```
{
 baseml:
|   .gitignore
|   main.py
|   README.md
|   requirements.txt
|
+---docs
|   \---reference
|           BayesBook.pdf
|           simplified_smo.pdf
|
+---examples
|       bayesian_regression.py
|       guassianNB.py
|       knn_classifier.py
|       knn_regressor.py
|       linear_regression.py
|       logistic_regression.py
|       pca.py
|       polynomial_regression.py
|       svm.py
|
+---ml
|   |   base.py
|   |   __init__.py
|   |
|   +---dimensionality
|   |       pca.py
|   |       __init__.py
|   |
|   +---regularizers
|   |       ElasticNetL1L2.py
|   |       L1.py
|   |       L2.py
|   |       NoRegularization.py
|   |       readme.md
|   |       __init__.py
|   |
|   +---supervised
|   |   |   __init__.py
|   |   |
|   |   +---fischers_lda
|   |   |       lda.py
|   |   |       __init__.py
|   |   |
|   |   +---knn
|   |   |       knn.py
|   |   |       __init__.py
|   |   |
|   |   +---naive_bayes
|   |   |       gaussian_naive_bayes.py
|   |   |       naive_bayes.py
|   |   |       __init__.py
|   |   |
|   |   +---regression
|   |   |   |   regression.py
|   |   |   |   __init__.py
|   |   |   |
|   |   |   +---bayesian
|   |   |   |       bayesian.py
|   |   |   |       bayesian_regression_update.md
|   |   |   |       __init__.py
|   |   |   |
|   |   |   +---linear
|   |   |   |       linear.py
|   |   |   |       README.md
|   |   |   |       __init__.py
|   |   |   |
|   |   |   +---logistic
|   |   |   |       logistic.py
|   |   |   |       newtons_method.md
|   |   |   |       __init__.py
|   |   |   |
|   |   |   \---polynomial
|   |   |           polynomial.py
|   |   |           __init__.py
|   |   |
|   |   \---svm
|   |       |   SimplifiedSMO.md
|   |       |   support_vector_machine.py
|   |       |   __init__.py
|   |       |
|   |       \---kernals
|   |               linear.py
|   |               polynomial.py
|   |               rbf.py
|   |               __init__.py
|   |
|   \---utils
|           data_utils.py
|           distances.py
|           distributions.py
|           viz.py
|           __init__.py
|
\---tests
        __init__.py
}
```

