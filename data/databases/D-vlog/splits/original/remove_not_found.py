import sys
import pandas as pd

if __name__ == "__main__":
    source_df_path = sys.argv[1]
    dest_df_path = sys.argv[2]

    source = pd.read_csv(source_df_path)
    not_found = pd.read_csv("../not_found.csv")["video_id"].tolist()
    print(source, len(source))
    for sampleID in not_found:
        source = source.drop(source[source["video_id"] == sampleID].index)
    print(source, len(source))
    source.to_csv(dest_df_path)
