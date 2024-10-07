# Sign-Language-project

## Environment

- OS : Ubuntu 18.04.5(Docker) LTS or Colab
- Cuda : 10.0
- GPU : Tesla V100-32GB

## Data

- sample video downlaod - ```$ sh download_sh/sample_data_dowonload.sh```

<a href='https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-003'>DataSet Download</a>

<img src='https://github.com/winston1214/Sign-Language-project/blob/master/picture/sample_data.gif?raw=true' height='50%' width='50%'></img>

## Enviorment Setting
```
$ pip install -r requirements.txt
$ python -m pip install cython
$ sudo apt-get install libyaml-dev
```
- Setting(Alphapose)
```
$ git clone https://github.com/winston1214/Sign-Language-project.git && cd Sign-Language-project
$ python setup.py build develop
```

If you don't run in the COLAB environment or the **cuda version is 10.0**, refer to <a href='https://bigdata-analyst.tistory.com/328?category=908124'>this link</a>.

- Download pretrained File(**Please Download**)

If you run this command, you can download weight file at once. ```$ sh downlaod_sh/weight_download.sh```

## PreProcessing

**1. Split frame**
```
$ python frame_split.py # You have to add the main code.
```
**2. Extract KeyPoint(Alphapose)**
```
python scripts/demo_inference.py --cfg configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml --checkpoint pretrained_models/halpe136_fast_res50_256x192.pt --indir ${img_folder_path} --outdir ${save_dir_path} --form boaz --vis_fast --sp
```

If you use multi-gpu, you don't have to **sp** option

## Extract KeyPoint
<img src='https://github.com/winston1214/Sign-Language-project/blob/master/picture/alphapose.gif?raw=true'></img>

## Train

```
$ python train.py --X_path ${X_train.pickle path} --save_path ${model save directory} \
--pt_name ${save pt model name} --model ${LSTM or GRU} --batch ${BATCH SIZE}

## Example

$ python train.py --X_path /sign_data/ --save_path pt_file/ \
--pt_name model1.pt --model GRU --batch 128 --epochs 100 --dropout 0.5
```
- X_train.pickle : For convenience, we stored and used the values extracted from the keypoint in **pickle file format**.
  - (shape : [video_len, max_frame_len, keypoint_len] # [7129, 376, 246] )

## Inference

```
$ python inference.py --video ${VIDEO_NAME} --outdir ${SAVE_PATH} --pt ${WEIGHT_PATH} --model ${MODEL NAME}
```

You can simply enjoy demo video at the COLAB [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winston1214/Sign-Language-project/blob/master/Inference.ipynb)

## Result

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Hyperparameter</th>
            <th>Metrics</th>
            <th>Final Model</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4><b>GRU-Attention</b></td>
            <td rowspan=2>Adam<br>CrossEntropy</td>
            <td>BLEU</td>
            <td>93.4</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>93.5</td>
        </tr>
        <tr>
            <td rowspan=2><b>AdamW<br>Scheduler</b></td>
            <td><b>BLEU</b></td>
            <td><b>95.1</b></td>
        </tr>
        <tr>
            <td><b>Accuracy</b></td>
            <td><b>95.0</b></td>
        </tr>
        <tr>
            <td rowspan=4>LSTM</td>
            <td rowspan=2>Adam<br>CrossEntropy</td>
            <td>BLEU</td>
            <td>49.6</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>50.0</td>
        </tr>
        <tr>
            <td rowspan=2>AdamW<br>Scheduler</td>
            <td>BLEU</td>
            <td><b>51.5</b></td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td><b>51.5</b></td>
        </tr>
    </tbody>
</table>

We selected a method that applied the **(HAND+BODY Keypoint) + (All Frame Random Augmentation) + (Frame Noramlization)** technique as the final model.

More experimental results are shown <a href='https://github.com/winston1214/Sign-Language-project/blob/master/docs/RESULT.md'>here</a>.

## Demo Video
<a href='https://www.youtube.com/watch?v=4E18JKXhl8w'>youtube link</a>

https://user-images.githubusercontent.com/47775179/150514151-e50bb76c-7556-42dd-bad3-24ffd2047066.mp4




