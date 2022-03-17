# pytorch-translator

cli for training sequence to sequence models fro translation tasks

## How to use?

```
python train.py \
    --input-lang {spa} \
    --output-lang {kr,jap} \
    --epochs EPOCHS \
    --sentences-per-epoch SENTENCES_PER_EPOCH \
    --gpu \
    --samples SAMPLES \
    --split-size SPLIT_SIZE \
    --hidden-size HIDDEN_SIZE \
    --lr LR \
    --save-model
```

```
python test.py \
    --input-lang {spa} \
    --output-lang {kr,jap} \
    --hidden-size HIDDEN_SIZE \
    --gpu
```
