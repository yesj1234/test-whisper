import os 

def list_of_strings(x):
    return x.split(",")

def main(lines, name):
    #/home/ubuntu/3.보완조치완료/2.Validation/1.원천데이터/2.일본어/일상,소통/9622/9622_52898_472.00_475.00.wav
    contents_scores = {}
    for line in lines: 
        wav, pred, ref, score = line.split(" :: ")
        score = float(score[:-2])
        cur_idx = wav.split("/")[8]
        if not cur_idx in contents_scores.keys():
            contents_scores[cur_idx] = [score] 
        else:
            contents_scores[cur_idx].append(score)
    contents_scores = {k: (float(sum(v)) / int(len(v))) for k, v in contents_scores.items()}
    contents_scores = {k:v for k, v in sorted(contents_scores.items(), key=lambda x: x[1])}
    with open(name, mode="w", encoding="utf-8") as t:
        for key, value in contents_scores.items():
            t.write(f"{key}: {value}\n")
    
if __name__ == "__main__": 
    import argparse 
    parser =  argparse.ArgumentParser()
    parser.add_argument('--files', type=list_of_strings, help="files to classify")
    
    args = parser.parse_args()
    for file in args.files:
        name = file.split("/")[-1]
        name = name.split("_")[0]
        with open(file, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            main(lines = lines, name = f"{name}_idx_scores.txt")
