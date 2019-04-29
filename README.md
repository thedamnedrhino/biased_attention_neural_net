# Cat-Dog-Frog Classifier
This is a project to classify 32x32 images from the CIFAR-10 dataset with three categories: dogs, cats and frogs. They can be found in the 'data.zip' archive.

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

## Comparing and Testing Your Outputs
The `-m` option supplied to the script also determines the name of the **accuracy output file generated by the script**. 
It is the best accuracy obtained from the validation set throughout all the training epochs.
The accuracy file name is what you supply to `-m` suffixed with `.accuracy`.
If you get a bunch of accuracy files from different models/runs, you can compare them using `outputparse.py`. The usage is simple:
```
python outputparse.py folder1 [folder2 [...]] 
```
This will gather all the `*.accuracy` files inside the given folders, (try to) parse them, sort them, and
print them in a nice table. It will also tell you if it is not able to parse any `*.accuracy` files that it found.
*Note that it is not recursive and won't go into subdirectories*.

## Experimental Outputs
Thousands of networks possible, through using different network architectures and hyperparameters were tested and
their accuracy outputs can be found in the archive `accuracy_outputs.zip`. The corresponding model files are stored on "the"
server and available upon request. 
You can view the results, best of which is a **81.8** at the moment, by extracting that archive into a folder called `accuracy_outputs`. Running `python results.py` at this point _should_ give you the top-10 performing networks (with the best one being in the last row).
The outside interface of the `results.py` script are expressed in freestanding functions:
- `run()`: displays the top networks found in `accuracy_outputs`
- `cos_reg()`: displays a plot (with matplotlib) for the `lcos` regularization penalty that was created for the project
- `top_5_acc()`: displays a plot for the accuracy of the top-5 networks and their baselines (manually entered data)

