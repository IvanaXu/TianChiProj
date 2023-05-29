import json

dev_file = './ccl-cfn/cfn-dev.json'
predictions_file = './ccl-cfn/result/task2_dev.json'

with open(dev_file, 'r') as f:
    dev_data = json.load(f)

with open(predictions_file, 'r') as f:
    predictions_data = json.load(f)

# assert len(dev_data) == len(predictions_data)

j = 0
total_TP = 0
total_FP = 0
total_FN = 0
preds = {}
labels = {}
for i, span in enumerate(predictions_data):
    preds.setdefault(span[0], set())
    preds[span[0]].add((span[1], span[2]))
for i, data in enumerate(dev_data):
    labels.setdefault(data['task_id'], set())
    for span in data['cfn_spans']:
        labels[data['task_id']].add((span[0], span[1]))

for taskid in labels:
    TP = 0
    FP = 0
    FN = 0
    if taskid not in preds:
        FN += len(labels[taskid])
        total_FN += FN
        continue
    
    for pred in preds[taskid]:
        if pred in labels[taskid]:
            TP += 1
        else:
            FP += 1
    for label in labels[taskid]:
        if label not in preds[taskid]:
            FN += 1
    total_TP += TP
    total_FP += FP
    total_FN += FN
    


print(total_TP, total_FP, total_FN)
precision = total_TP / (total_TP + total_FP)
recall = total_TP / (total_TP + total_FN)
F1 = 2 * precision * recall / (precision + recall)
print(f'precision: {precision}, recall: {recall}, F1: {F1}')