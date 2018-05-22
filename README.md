# scikit-visualizations
Scikit-learn-compatible visualizations

## Status
[![Build Status](https://travis-ci.com/daviddiazvico/scikit-visualizations.svg?branch=master)](https://travis-ci.com/daviddiazvico/scikit-visualizations)
[![Maintainability](https://api.codeclimate.com/v1/badges/0d44dcdbb204b8f4fc37/maintainability)](https://codeclimate.com/github/daviddiazvico/scikit-visualizations/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/0d44dcdbb204b8f4fc37/test_coverage)](https://codeclimate.com/github/daviddiazvico/scikit-visualizations/test_coverage)

## Installation
Available in [PyPI](https://pypi.python.org/pypi?:action=display&name=scikit-visualizations)
```
pip install scikit-visualizations
```

## Documentation
Autogenerated and hosted in [GitHub Pages](https://daviddiazvico.github.io/scikit-visualizations/)

## Distribution
Run the following command from the project home to create the distribution
```
python setup.py sdist bdist_wheel
```
and upload the package to [testPyPI](https://testpypi.python.org/)
```
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```
or [PyPI](https://pypi.python.org/)
```
twine upload dist/*
```

## Citation
If you find scikit-visualizations useful, please cite it in your publications. You can use this [BibTeX](http://www.bibtex.org/) entry:
```
@misc{scikit-visualizations,
      title={scikit-visualizations},
      author={Diaz-Vico, David},
      year={2017},
      publisher={GitHub},
      howpublished={\url{https://github.com/daviddiazvico/scikit-visualizations}}}
```