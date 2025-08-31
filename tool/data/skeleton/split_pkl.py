import pickle

with open('train.pkl', 'rb') as f:
    annotations = pickle.loads(f.read())
# 按照自己的方式切割训练集和验证集
split = {"train": [], "val": []}
for i in range(794):
    if i%4 != 0:
        split["train"].append(annotations[i]['frame_dir'])
    else:
        split["val"].append(annotations[i]['frame_dir'])

with open('mnist_train.pkl', 'wb') as f:
        pickle.dump(dict(split=split, annotations=annotations), f)
print(len(split["train"])+len(split["val"]), len(annotations))
