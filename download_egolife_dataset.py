############
#added by Kousik
# for downloading EgoLife dataset from HuggingFace Hub
############

from huggingface_hub import snapshot_download
import os
os.environ["HF_HUB_READ_TIMEOUT"] = "60"
os.environ["HF_HUB_TIMEOUT"] = "300"  # total timeout




# local_path = snapshot_download(
#     repo_id="lmms-lab/EgoLife",
#     repo_type="dataset",
#     local_dir="data/EgoLife",
#     allow_patterns=["A2_ALICE/**"],
#     max_workers=4,
#     token=True,
# )
# Download the EgoLife dataset snapshot
local_path = snapshot_download(
    repo_id="lmms-lab/EgoIT-99K", #"lmms-lab/EgoLife",
    repo_type="dataset",
    local_dir="EgoIT_99K/EgoLife_audio", #"d/EgoLife",          # <-- where to store it
    local_dir_use_symlinks=False,      # use hard copies instead of symlinks
    resume_download=True,
    max_workers=2,                     # fewer threads reduces timeout risk
    token=True ,                        # needed if it's gated/private
    allow_patterns=[
        "EgoLife_audio/A2_ALICE/**",  # <-- only download Alice's audio
        
    ]
)

print(f"Downloaded to: {local_path}")

#for egolife

# #for castel 
# from huggingface_hub import snapshot_download
# import os

# os.environ["HF_HUB_READ_TIMEOUT"] = "60"
# os.environ["HF_HUB_TIMEOUT"] = "300"

# local_path = snapshot_download(
#     repo_id="CASTLE-Dataset/CASTLE2024",   # replace with actual HF repo id
#     repo_type="dataset",
#     local_dir="CASTLE2024",
#     local_dir_use_symlinks=False,
#     resume_download=True,
#     max_workers=2,
#     token=True,  # set to True if authentication is needed
#     # allow_patterns=[
#     #     "main/day1/Allie/video/09.mp4",
#     # ]
#     allow_patterns = [
#     "main/day1/Allie/video/1?.mp4",  # 10–19
#     "main/day1/Allie/video/20.mp4",  # 20
#     ]
# )

#print(f"Downloaded to: {local_path}")