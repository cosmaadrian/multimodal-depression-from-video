import os
from tqdm import tqdm

if __name__ == "__main__":
   root_dir = "./backup/"
   dest_dir = "./data/"
   tar_files = sorted( os.listdir(root_dir) )

   for tar_file in tqdm(tar_files):
       tar_path = os.path.join(root_dir, tar_file)
       os.system(f"tar -xf {tar_path} -C {dest_dir}")
