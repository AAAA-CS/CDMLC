Code for the paper: [Few-shot Learning Based on Multi-level Contrast for Cross-domain Hyperspectral Image Classification]

## Requirements

CUDA Version: 10.2

torch: 1.6.0

Python: 3.6.5

## Usage
Take CDMLC method on the Chikusei (Source Data) and HoustonU 2013 (Target Data) as an example: 

1.Running the script `Chikusei_imdb_128.py` to generate preprocessed source domain data, where `patch_length = 4` is used for 9*9 patch size.

2.Running `python train_CDMLC.py --config config/houston.py`

3.`config/salinas.py` and `config/Indian_pines.py` are used for Salinas data and Indian Pines data.
 * `test_lsample_num_per_class` denotes the number of labeled samples per class for the target data.
 * `tar_input_dim` denotes the number of bands for the target data.

## 使用方法
以Chikusei（源数据）和HoustonU 2013（目标数据）上的CDMLC方法为例：

1. 运行脚本`Chikusei_imdb_128.py`生成预处理的源域数据，其中`patch_length = 4`用于9*9的补丁大小。

2. 运行`python train_CDMLC.py --config config/paviaU.py`

3. 对于Salinas数据和Indian Pines数据，使用`config/salinas.py`和`config/Indian_pines.py`。
   * `test_lsample_num_per_class`表示目标数据每个类别的标记样本数量。
   * `tar_input_dim`表示目标数据的波段数。
