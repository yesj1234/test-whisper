import os 
import librosa 

def list_of_strings(x):
    return x.split(",")

def main(args, category, content_idxs):
    duration = 0
    with open(f"{category}_predictions.txt", encoding="utf-8", mode="r") as f:
        lines = f.readlines()
        for line in lines: 
            wavname, _, _, _ = line.split(" :: ")
            duration_in_seconds = librosa.get_duration(path=wavname)
            duration += float(duration_in_seconds)
    in_hour = round((duration / 3600 ), 2)  
    with open(f"{args.lang}_{category}_{in_hour}시간.txt", mode="w", encoding="utf-8") as g:
        for idx in content_idxs:
            g.write(f"{idx}\n")
            
if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=list_of_strings, help="list of files to get running time")
    parser.add_argument("--lang", help="일본어, 중국어, 한국어, 영어")
    args = parser.parse_args()
    
    for file in args.files: 
        category = file.split("/")[-1].split(".")[0].split("_")[0]
        with open(file, mode="r", encoding="utf-8") as t:
            lines = t.readlines()
            lines = list(map(lambda x: x.split(":")[0], lines))[-4:]
        main(args, category=category, content_idxs=lines)