# Cat-Dog-Frog Classifier
This is a project to classify some 32*32 images with three categories: dogs, cats and frogs. They can be found in the 'data.zip' archive.

## Installing

First run a virtual environment. You can usually do this using:

```
$ virtualenv venv
```


For image visualizations through matplotlib to work, on **OSX** you would have to:
```
$ python -m venv venv
```

Then you would have to install the requirements (after activating the virtual environment through `source venv/bin/activate`:

```
$ pip install -r requirements.txt
```

Extract the dataset:
```
$ unzip data.zip
```
This will create a folder called `dataset` with the relevan files.

## Running
First extract the data. It would be easiest to put the contents in a folder called `datasets` in the same folder as the code.  
The code works both with and without a gpu. The main file is `model.py` and in fact it is self sufficient.
The code used to be distributed in the three files `model.py`, `dataset.py` and `data.py`, but the assignment requirements
stated that I had to have everything in one file, so I basically copied the other two files into `model.py`. The contents of the 
two other files are put in the two classes called `dataset` and `data` in `model.py`. Those files are kept since they have useful main functions.    
To run, do `python model.py -h` first
to see the options you would have available. You can control:
- the directory of the dataset (defaults to './datasets')
- the number of epochs to train
- the transformers to use
- to load model checkpoint from a saved file
- the name of the file to save the best model in
- to validate the data (disables training)
- to test the data and overwrite the labels in `testlabel.pickle` in the dataset folder (you will create this folder by extracting the zip archive)
- to merge the validation data for training (instead of just validating on them)

Then you can train with something like:
```
python model.py -e 20 -d './custom_dataset_folder' -m 'my_model.model' 
```
Validate with something like:
```
python model.py -v -d './custom_dataset_folder' -l 'my_model.model'
```

You can generate the testlabels by adding the `-t` option to the previous command or replacing `-v` if you don't want to recheck
the validation results (to see if you have the right model and etc.)
