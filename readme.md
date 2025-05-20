### code structure

data: raw train data
dataset: data processing
experiments: including train code and test code
loss: loss code
model: 
    bidirectional: to enable model bidirectional attention
    model_main: 
    model_hcf: Hirarchical-Cross-Fusion code
test_configs: NAN
train_configs: NAN



### train

cd Hierarchical-Cross-Fusion
python experiments/train_decoder.py train_configs/decoder.json

### evaluation

cd Hierarchical-Cross-Fusion
python experiments/mteb_eval.py