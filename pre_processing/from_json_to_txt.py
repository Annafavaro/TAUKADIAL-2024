
fold = '/export/b01/afavaro/IS_2024/TAUKADIAL-24/train_transcriptions_and_diarization/'
out = '/export/b01/afavaro/IS_2024/TAUKADIAL-24/train_transcriptions_diarization/'

import json
import os

all_jsons = [os.path.join(fold, elem) for elem in os.listdir(fold) if elem.endswith(".json")]

for json_file in all_jsons:
    f = open(json_file)
    path_base = os.path.basename(json_file.split(".json")[0])
    print(path_base)
    path_out = os.path.join(out, path_base + '.txt')
    data = json.load(f)
    sents = []
    for i in range(len(data['segments'])):
        sent = data['segments'][i]['text']#.strip()
        sents += [sent]
        with open(path_out,'w') as output:
            for line in sents:
                output.write(line)