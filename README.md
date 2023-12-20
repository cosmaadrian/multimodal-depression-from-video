# Reading Between the Frames: Multi-Modal Depression Detection in Videos from Non-Verbal Cues

## Downloading datasets

- For D-Vlog, albeit the features extracted by the authors are publicly available [here](https://sites.google.com/view/jeewoo-yoon/dataset), original vlog videos are available upon request.

- For DAIC-WOZ and E-DAIC, the features are only available upon request [here](https://dcapswoz.ict.usc.edu/).

## Extracting non-verbal modalities from D-Vlog

### D-Vlog

- To extract the audio embeddings:

```
conda create -y -n pase+ python=3.7
conda activate pase+
bash ./scripts/conda_envs/prepare_pase+_env.sh
bash ./scripts/features
scripts/feature_extraction/extract-dvlog-pase+-feats.sh
conda deactivate pase+
```

- To extract face, body, and hand landmarks:

```
conda create -y -n landmarks python=3.8
conda activate landmarks
bash scripts/conda_envs/prepare_landmarks_env.sh
scripts/feature_extraction/extract-dvlog-landmarks.sh
conda deactivate landmarks
```

- To extract face EmoNet embeddings:

```
conda create -y -n emonet python=3.8
conda activate emonet
bash ./scripts/conda_envs/prepare_emonet_env.sh
bash ./scripts/feature_extraction/extract-dvlog-emonet-feats.sh
conda deactivate emonet
```

- To extract gaze tracking:

```
conda create -y -n mpiigaze python=3.8
conda activate mpiigaze
bash ./scripts/conda_envs/prepare_mpiigaze_env.sh
bash ./scripts/feature_extraction/extract-dvlog-gaze-feats.sh
conda deactivate mpiigaze
```

- To extract blinking features:

```
conda create -y -n instblink python=3.7
conda activate instblink
bash ./scripts/conda_envs/prepare_instblink_env.sh
bash ./scripts/feature_extraction/extract-dvlog-blinking-feats.sh
conda deactivate instblink
```

### DAIC-WOZ

- To pre-process the DAIC-WOZ features:

```
conda activate landmarks
bash ./scripts/feature_extraction/extract-daicwoz-features.sh
conda deactivate
```

### E-DAIC
- To pre-process the DAIC-WOZ features:

```
conda activate landmarks
bash ./scripts/feature_extraction/extract-edaic-features.sh
conda deactivate
```

## Implementation Detail

Once all the data has been pre-processed, you should indicate the absule path to the directory where it is stored
in the 'configs/env_config.yaml' file for each one of the corresponding datasets.

In addition, you can continue working in the 'landmarks' environment, since it has everything we 
need for training and evaluating our model:

```
conda activate landmarks
```

## Training and Evaluation
To train and evaluate the models and the results reported in the paper, you can run the following commands:

```
cd experiments/
bash run-exps.sh
```

## License

This work is protected by CC BY-NC-ND 4.0 License (Non-Commercial & No Derivatives).
