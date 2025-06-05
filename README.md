# TVAE-RNA
Installation
Please first install the required packages using the specified versions listed in requirements.yml on your server.

Testing
We provide testing code in the test folder. You can use your own data to test our model; the input format should be consistent with the training code.

The test pipeline will output the following:

F1 score

Accuracy

Recall

Violin plots

A .pkl file where the keys are RNA sequences and the values are their predicted pairing matrices.

If you prefer not to generate violin plots, we also provide code for generating .ct files. You can use this to batch convert predictions into .ct format for your own test data.

Training
The training code is located in the train folder. Example training data is provided in the data folder, and pre-trained model weights are included in the models folder.

You can:

Directly run training with the provided data, or

Use your own data by following the same data format.

All necessary data processing code is implemented in shuju.py. Please make sure that both shuju.py and the fm folder are in the same directory, as the RNA-FM feature files are large and need to be downloaded separately.

If you prefer not to use RNA-FM embeddings, we also provide an alternative version of the model that uses one-hot encoding as input.

To start training, simply run the following command in the current directory:

python train.py

GHA Algorithm
The executable and source code for the GHA algorithm are provided in the GHA folder. You can either use the precompiled binary or compile it yourself using the following steps:

1. Compile the source code:
g++ -std=c++11 -O2 -o hungarian_rna hungarian_rna.cpp
2. Check the compilation result:
ls -la hungarian_rna
3. Make it executable:
chmod +x hungarian_rna
4. Create test input files:
echo -e "0.1 0.2 0.8 0.1\n0.2 0.1 0.2 0.7\n0.8 0.2 0.1 0.2\n0.1 0.7 0.2 0.1" > test_fg.mat
echo -e "0.5 0.5 0.5 0.5\n0.5 0.5 0.5 0.5\n0.5 0.5 0.5 0.5\n0.5 0.5 0.5 0.5" > test_bg.mat
5. Run the test:
./hungarian_rna test_fg.mat test_bg.mat
