### Preprocessing
Before training the model, we must preprocess the dataset that was produced by pythia. This script will load the root file into python as awkward arrays, apply some basic cuts using a boolean mask, pad tracks to a common length, and export a pickled serialized object with the torch Tensors needed for training. Run the following commmand:

```
./run_preprocessing.sh {Efrac|Mfrac} ../pythia/output/<input>.root data/<output>.pkl plots/<Dir Name>
```
Please provide the four arguments: (1) label type (2) path to input file (3) path to output file (4) path to plotting file notice there is NO front slash after the \<Dir Name>!
### Attention Model
To train the model on the preprocessed data, please run the following command:

```
./train_model.sh {Num_Epochs} data/<input>.pkl results/<out model>.torch plots/<Dir Name>
```
Please provide the four arguments: (1) Num Epochs (int) (2) path to input file (3) path to output file (4) path to plotting file notice there is NO front slash after the \<Dir Name>!
