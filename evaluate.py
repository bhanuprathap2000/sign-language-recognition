import os
import glob
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils import data
from generate_keypoints import process_video
from models import Transformer,LSTM
from configs import TransformerConfig,LstmConfig
from utils import load_json, load_label_map
import shutil
import os, sys
import stat
parser = argparse.ArgumentParser(description="Evaluate function")
parser.add_argument("--data_dir", required=True, help="data directory")
args = parser.parse_args()



class KeypointsDataset(data.Dataset):
    def __init__(
        self,
        keypoints_dir,
        max_frame_len=200,
        frame_length=1080,
        frame_width=1920,
    ):
        self.files = sorted(glob.glob(os.path.join(keypoints_dir, "*.json")))
        self.max_frame_len = max_frame_len
        self.frame_length = frame_length
        self.frame_width = frame_width

    def interpolate(self, arr):

        arr_x = arr[:, :, 0]
        arr_x = pd.DataFrame(arr_x)
        arr_x = arr_x.interpolate(method="linear", limit_direction="both").to_numpy()

        arr_y = arr[:, :, 1]
        arr_y = pd.DataFrame(arr_y)
        arr_y = arr_y.interpolate(method="linear", limit_direction="both").to_numpy()

        if np.count_nonzero(~np.isnan(arr_x)) == 0:
            arr_x = np.zeros(arr_x.shape)
        if np.count_nonzero(~np.isnan(arr_y)) == 0:
            arr_y = np.zeros(arr_y.shape)

        arr_x = arr_x * self.frame_width
        arr_y = arr_y * self.frame_length

        return np.stack([arr_x, arr_y], axis=-1)

    def combine_xy(self, x, y):
        x, y = np.array(x), np.array(y)
        _, length = x.shape
        x = x.reshape((-1, length, 1))
        y = y.reshape((-1, length, 1))
        return np.concatenate((x, y), -1).astype(np.float32)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        row = pd.read_json(file_path, typ="series")
        label = row.label
        label = "".join([i for i in label if i.isalpha()]).lower()

        pose = self.combine_xy(row.pose_x, row.pose_y)
        h1 = self.combine_xy(row.hand1_x, row.hand1_y)
        h2 = self.combine_xy(row.hand2_x, row.hand2_y)

        pose = self.interpolate(pose)
        h1 = self.interpolate(h1)
        h2 = self.interpolate(h2)

        df = pd.DataFrame.from_dict(
            {
                "uid": row.uid,
                "pose": pose.tolist(),
                "hand1": h1.tolist(),
                "hand2": h2.tolist(),
                "label": label,
            }
        )


 

        pose = (
            np.array(list(map(np.array, df.pose.values)))
            .astype(np.float32)
        )
        #
        pose_shape=pose.shape
        old_total=pose_shape[0]*pose_shape[1]*pose_shape[2]
        total_frame=pose_shape[0]
        # print("total frame:",total_frame)
        temp_total_shape=total_frame
        # print("temp total shape:",temp_total_shape)

        while (temp_total_shape*pose_shape[1]*pose_shape[2])%50!=0:
            temp_total_shape+=1
        pose=np.concatenate([pose,np.zeros((temp_total_shape-total_frame,33,2))])
        
        
        # print("shape of pose",pose.shape)

        #
        pose=pose.reshape(-1,50)
        new_total=pose.shape[0]*pose.shape[1]
        new_pose=pose.shape[0]
        # print("new shape",new_pose)
        # print("total frame",total_frame)

        n_rows=1
        # print('new_total',new_total)
        # print('old_total',old_total)
        # print('diff',new_total-old_total)
        while n_rows*50<(new_total-old_total):
            n_rows+=1
            new_pose-=1
        
        # print("new pose",new_pose)
        # print("n_rows",n_rows)
        pose=pose[:-n_rows,:]
        # print("last pose shape",pose.shape)
        # sys.exit(1)
        
        
        h1 = (
            np.array(list(map(np.array, df.hand1.values)))
            .reshape(-1,42)
            .astype(np.float32)
        )
        h1=np.concatenate([h1,np.zeros((pose.shape[0]-total_frame,42))])
        h2 = (
            np.array(list(map(np.array, df.hand2.values)))
            .reshape(-1, 42)
            .astype(np.float32)
        )
        h2=np.concatenate([h2,np.zeros((pose.shape[0]-total_frame,42))])
        # print(pose.shape)
        # print(h1.shape,h2.shape)
        # sys.exit(1)
        final_data = np.concatenate((pose, h1, h2), -1)
        final_data = np.pad(
            final_data,
            ((0, self.max_frame_len - final_data.shape[0]), (0, 0)),
            "constant",
        )
        return {
            "uid": row.uid,
            "data": torch.FloatTensor(final_data),
        }

    def __len__(self):
        return len(self.files)

# def del_evenReadonly(action, name, exc):
#     os.chmod(name, stat.S_IWRITE)
#     os.remove(name)

@torch.no_grad()
def inference(dataloader, model, device, label_map):
    model.eval()
    predictions = []

    for batch in tqdm(dataloader, desc="Eval"):
        input_data = batch["data"].to(device)
        output = model(input_data).detach().cpu()
        output = torch.argmax(torch.softmax(output, dim=-1), dim=-1).numpy()
        predictions.append({"uid": batch["uid"][0], "predicted_label": label_map[output[0]]})

    return predictions


video_paths = glob.glob(os.path.join(args.data_dir, "*"))
print(video_paths)

save_dir = "keypoints_dir"
if os.path.isdir(save_dir):
    shutil.rmtree(save_dir)
os.mkdir(save_dir)


for path in tqdm(video_paths, desc="Processing Videos"):
    process_video(path, save_dir)

label_map = load_label_map("include50")

dataset = KeypointsDataset(
    keypoints_dir=save_dir,
    max_frame_len=169,
)

dataloader = data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)
label_map = dict(zip(label_map.values(), label_map.keys()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = LstmConfig()
model = LSTM(config=config, n_classes=50)
model = model.to(device)

pretrained_model_name = "include50_no_cnn_lstm.pth"
pretrained_model_links = load_json("pretrained_links.json")
if not os.path.isfile(pretrained_model_name):
    link = pretrained_model_links[pretrained_model_name]
    torch.hub.download_url_to_file(link, pretrained_model_name, progress=True)

ckpt = torch.load(pretrained_model_name,map_location=torch.device('cpu'))
model.load_state_dict(ckpt["model"])
print("### Model loaded ###")

preds = inference(dataloader, model, device, label_map)
print(json.dumps(preds, indent=2))
