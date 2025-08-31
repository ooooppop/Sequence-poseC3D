import os.path as osp
import os
import pickle
result = []
path = "D:/WuShuang/mmaction2-main/tools/data/skeleton/data/my_video/pkl4/"
for d in os.listdir(path):
    if d.endswith('.pkl'):
        with open(osp.join(path, d), 'rb') as f:
            content = pickle.load(f)
        result.append(content)
with open('train.pkl', 'wb') as out:
    pickle.dump(result, out, protocol=pickle.HIGHEST_PROTOCOL)