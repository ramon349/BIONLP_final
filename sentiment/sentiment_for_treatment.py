import json
from collections import defaultdict
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

from LDA import LDA

sentiment_data_path = "./sentiment_results.txt"
#  format
#  'id':str,
#  'text':str,
#  'treatments':str,
#  'sentiment_label':['NEGATIVE','POSITIVE'],
#  'sentiment_score': float


def load_sentiment_results():
    data = []
    with open(sentiment_data_path,  'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def split_data_by_treatment(data):
    treatment_to_samples = defaultdict(list)
    for sample in data:
        treatment_to_samples[sample['treatments']].append(sample)
    return treatment_to_samples


if __name__ == '__main__':
    sentiment_data = load_sentiment_results()
    treat_to_sample = split_data_by_treatment(sentiment_data)

    treatment_sorted_keys = sorted(treat_to_sample.keys(), key=lambda x: -len(treat_to_sample[x]))
    print('Data Distribution:')
    print([(x, len(treat_to_sample[x])) for x in treatment_sorted_keys])
    print('total number of treatment:', len(treatment_sorted_keys))
    for treatment in treatment_sorted_keys[:2]:
        print('Analysing treatment:', treatment)
        print('total number of samples:', len(treat_to_sample[treatment]))
        all_samples = treat_to_sample[treatment]
        # LDA analysis
        text_collection = [x['text'] for x in all_samples]
        LDA_result = LDA(text_collection)
        print('Topic Analysis:\n', LDA_result)

        # Label analysis
        label_collection = [1 if x['sentiment_label'] == 'POSITIVE' else 0 for x in all_samples]
        print('sentiment bias:')
        print('positive:', sum(label_collection))
        print('negative:', len(label_collection) - sum(label_collection))
        print('positive rate:', sum(label_collection)/len(label_collection))

        # word cloud analysis
        for number_of_topics, topics in LDA_result:
            topic = '0.027*"breast" + 0.024*"cancer" + 0.016*"chemo" + 0.010*"treatment" + 0.010*"year" + 0.009*"get" + 0.009*"feel" + 0.009*"surgeri" + 0.008*"mom" + 0.008*"worri" + 0.007*"radiotherapi" + 0.006*"know" + 0.006*"good" + 0.006*"mani" + 0.006*"got" + 0.005*"like" + 0.005*"hope" + 0.005*"menopaus" + 0.004*"remov" + 0.004*"professor"'
            wordcloud = WordCloud(background_color='white').generate(topic)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.show()


        print('\n\n')