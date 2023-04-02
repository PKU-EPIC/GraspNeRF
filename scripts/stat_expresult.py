from pathlib import Path
import sys
import pandas as pd
import os
import numpy as np

argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--"
log_root_dir = str(argv[0])
expname = str(argv[1])

class Data(object):
    """Object for loading and analyzing experimental data."""

    def __init__(self, logdir):
        self.logdir = logdir
        self.rounds = pd.read_csv(logdir / "rounds.csv")
        self.grasps = pd.read_csv(logdir / "grasps.csv")

    def num_rounds(self):
        return len(self.rounds.index)

    def num_grasps(self):
        return len(self.grasps.index)

    def success_rate(self):
        return self.grasps["label"].mean() * 100

    def percent_cleared(self):
        df = (
            self.grasps[["round_id", "label"]]
            .groupby("round_id")
            .sum()
            .rename(columns={"label": "cleared_count"})
            .merge(self.rounds, on="round_id")
        )
        return df["cleared_count"].sum() / df["object_count"].sum() * 100

    def avg_planning_time(self):
        return self.grasps["planning_time"].mean()

    def read_grasp(self, i):
        scene_id, grasp, label = io.read_grasp(self.grasps, i)
        score = self.grasps.loc[i, "score"]
        scene_data = np.load(self.logdir / "scenes" / (scene_id + ".npz"))

        return scene_data["points"], grasp, score, label

##############################
# Combine all trials
##############################
root_path = os.path.join(log_root_dir, "exp_results", expname)

round_dir_list = sorted(os.listdir(root_path))

if not os.path.exists(root_path + "_combine"):
    os.makedirs(root_path + "_combine")

df = pd.DataFrame()
for i in range(len(round_dir_list)):
    df_round = pd.read_csv(os.path.join(root_path, round_dir_list[i], "grasps.csv"))
    df_round["round_id"] = i
    df = pd.concat([df, df_round])
df = df.reset_index(drop=True)
df.to_csv(os.path.join(root_path + "_combine", "grasps.csv"), index=False)


df = pd.DataFrame()
for i in range(len(round_dir_list)):
    df_round = pd.read_csv(os.path.join(root_path, round_dir_list[i], "rounds.csv"))
    df_round["round_id"] = i
    df = pd.concat([df, df_round])
df = df.reset_index(drop=True)
df.to_csv(os.path.join(root_path + "_combine", "rounds.csv"), index=False)

##############################
# Print Stat
##############################
logdir = Path(os.path.join(log_root_dir, "exp_results", expname+"_combine"))
data = Data(logdir)

# First, we compute the following metrics for the experiment:
# * **Success rate**: the ratio of successful grasp executions,
# * **Percent cleared**: the percentage of objects removed during each round,
try:
    print("Path:              ",str(logdir))
    print("Num grasps:        ", data.num_grasps())
    print("Success rate:      ", data.success_rate())
    print("Percent cleared:   ", data.percent_cleared())
except:
    print("[W] Incomplete results, exit")
    exit()
##############################
# Calc first-time grasping SR
##############################

sum_label = 0
firstgrasp_fail_expidx_list = []
for i in range(len(round_dir_list)):
    #print(i)
    df_round = pd.read_csv(os.path.join(root_path, round_dir_list[i], "grasps.csv"))
    df = df_round.iloc[0:1,:]   
    
    label = df[["label"]].to_numpy(np.float32)
    if label.shape[0] == 0:
        firstgrasp_fail_expidx_list.append(i)
        continue
    sum_label += label[0,0]
    if label[0,0]==0:
        firstgrasp_fail_expidx_list.append(i)

print("First grasp success rate: ", sum_label / len(round_dir_list))
print("First grasp fail:", len(firstgrasp_fail_expidx_list),"/",len(round_dir_list), ", exp id: ", firstgrasp_fail_expidx_list)
