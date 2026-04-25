# CDMLC
Code for the paper: [Few-shot Learning Based on Multi-level Contrast for Cross-domain Hyperspectral Image Classification]

## Usage
Take CDMLC method on the Chikusei (Source Data) and Indian Pines (Target Data) as an example: 

1.Running the script `Chikusei_imdb_128.py` to generate preprocessed source domain data, where `patch_length = 4` is used for 9*9 patch size.

2.Running `python train_CDMLC.py --config config/Indian_pines.py`
