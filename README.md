# Classification of textual descriptions in cultural heritage records

Binary classification task to automatically identify high-quality and low-quality descriptions
in cultural heritage records.  

## Resources 

* [FastText](https://fasttext.cc/)
* [ScikitLearn](https://scikit-learn.org/stable/index.html)
* [Wikipedia Embeddings](https://fasttext.cc/docs/en/pretrained-vectors.html)
* [Annotated dataset](https://figshare.com/articles/dataset/Annotated_dataset_to_assess_the_accuracy_of_the_textual_description_of_cultural_heritage_records/13359104)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)
## Dependencies 

* Python3
* Pandas

## Pipeline

* [converter.py](https://github.com/matteoLorenzini/description_quality/blob/master/converter.py):creates vector data in .tsv format

* FastText folder:
	* [prepare4fastTextClassifier.py](https://github.com/matteoLorenzini/description_quality/blob/master/FastText/prepare4fastTextClassifier.py): 
		* converts the input .csv file in FT format (i.e __label__Good).
		* splits the input dataset in folds, by default 10. (--fold option)
	* [fasttextClassifier.py](https://github.com/matteoLorenzini/description_quality/blob/master/FastText/fasttextClassifier.py)
		* classifies the descriptions (*.csv.fbclass file) and returns a file (.eval.gz) with the classification report.
	* [evaluate.py](https://github.com/matteoLorenzini/description_quality/blob/master/FastText/evaluate.py)
		* evaluates the classification task results from the *.eval.gz file
* LibSVM folder:
	* [K-fold.py](https://github.com/matteoLorenzini/description_quality/blob/master/LibSVM/K-fold.py)
		* runs the K-foldvalidation on the .tsv created by the [converter.py](https://github.com/matteoLorenzini/description_quality/blob/master/converter.py)
* learning_curve folder:
	* [estimator.py](https://github.com/matteoLorenzini/description_quality/blob/master/learning_curve/estimator.py)
		* computes the F1 score by using different splits 
	* [learning_curve.py](https://github.com/matteoLorenzini/description_quality/blob/master/learning_curve/learning_curve.py)
		* plot the learning curve

[![License: CC BY-SA 4.0](https://licensebuttons.net/l/by-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-sa/4.0/) [![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)