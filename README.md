# Sign-Language-Recognition
This repo contains the code for sign-language-recognition as part of our final year project.

## Dataset Link For INCLUDE 50:[Sign Language Dataset](https://zenodo.org/records/4010759)

## Video [Explanation Video](https://drive.google.com/file/d/1QfFWgh3hXmhawjZ0twCvGleM8yk79JOW/view?usp=sharing)

# Dependencies

Install the dependencies through the following command

```bash
>> pip install -r requirements.txt
```



## Steps
- Download the INCLUDE dataset
- Run `generate_keypoints.py` to save keypoints from Mediapipe Hands and Blazepose for train, validation and test videos. 
```bash
>> python generate_keypoints.py --include_dir <path to downloaded dataset> --save_dir <path to save dir> --dataset <include/include50>
```
- Run `runner.py` to train a machine learning model on the dataset
```bash
>> python runner.py --dataset <include/include50> --use_augs --model transformer --data_dir <location to saved keypoints>
```
- Use the `--use_pretrained` flag to either perform only inference using pretrained model or resume training with the pretrained model. 
```bash
>> python runner.py --dataset <include/include50> --use_augs --model transformer --data_dir <location to saved keypoints> --use_pretrained <evaluate/resume_training>
```
- To get predictions for videos from a pretrained model, run the following command.
```bash
>> python evaluate.py --data_dir <dir with videos>
```
