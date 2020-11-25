import csv
from tqdm import tqdm
import json
import csv
from sklearn.metrics import cohen_kappa_score
SAVE_PATH = './sentiment_results.txt'
SAMPLE_PATH = './samples.txt'
PATH = r'/home/yzh2749/BioNLP/final/repo/sentiment/brest_cancer_by_treatments.csv'


def load_csv(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for i, line in enumerate(reader):
            if i == 0:
                continue
            data.append(line)
    return data

from transformers import pipeline
classifier = pipeline('sentiment-analysis')
data = load_csv(PATH)


# ## sample data
# import random
# csv_writer = csv.writer(open(SAMPLE_PATH,'w'))
# csv_writer.writerows([[i, x,''] for i, x in random.sample(list(enumerate(data)), 100)])

## run the pipline
with open(SAVE_PATH, 'w') as file:
    for i, text in enumerate(tqdm(data)):
        res = classifier(text[1])
        file.write(json.dumps({'id':text[0],
                               'text':text[1],
                               'treatments':text[2],
                               'sentiment_label':res[0]['label'],
                               'sentiment_score':str(res[0]['score'])}) + '\n')
        # print(i,'/', len(data))
        # print(text)
        # print(res)
        # print('\n\n')

# computer IAA
SAVE_PATH = "./sentiment_results.jsonl"
pred_labels = {}
text = {}
scores = {}
with open(SAVE_PATH, 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        data = json.loads(line)
        pred_labels[i] = data['sentiment_label']
        text[i] = data['text']
        scores[i] = data['sentiment_score']

alpha = 0.9
labeled_samples = {}
with open("samples.csv", 'r', encoding='utf-8') as file:
    for line in file:
        line = line.split(',')
        try:
            labeled_samples[int(line[0])] = int(line[-1])
        except:
            continue
        id = int(line[0])
        pred_label = 1 if pred_labels[id] == "POSITIVE" else 0
        if pred_label == 0 and float(scores[id]) < alpha:
            pred_label = 1

        if pred_label != int(line[-1]):
            print(line, text[int(line[0])], scores[int(line[0])])


truth = []
pred = []
for k in labeled_samples.keys():
    truth.append(labeled_samples[k])
    pred.append(1 if pred_labels[k]=='POSITIVE' or float(scores[k]) < alpha else 0)
from sklearn.metrics import accuracy_score, f1_score, classification_report

print(classification_report(truth, pred))
print(cohen_kappa_score(truth,pred))