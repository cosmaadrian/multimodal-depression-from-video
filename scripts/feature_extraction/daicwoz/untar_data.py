import os
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root-dir", type=str, default="./data/DAIC-WOZ/backup/")
    parser.add_argument("--dest-dir", type=str, default="./data/DAIC-WOZ/data/")
    args = parser.parse_args()

   tar_files = sorted( os.listdir(args.root_dir) )

   for tar_file in tqdm(tar_files):
       tar_path = os.path.join(args.root_dir, tar_file)
       os.system(f"tar -xf {tar_path} -C {args.dest_dir}")
