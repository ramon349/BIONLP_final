import pandas as pd 
from vectorizers import myVectorizer, text2pred
sentiments = ['noReport','selfReport']
sentiments = ['noReport','selfReport']

def main_three(db,reportType,out_file):
    import services.mongo_setup as mongo_setup 
    import services.data_service as data_service
    classi = text2pred()
    mongo_setup.global_init(db) 
    sample_tweet= data_service.find_tweet_by_mood(report=reportType).all()
    rep_list = list()
    print('Start appending')
    for e in sample_tweet: 
        text=e.text
        report = sentiments[classi(text)]
        id = e.tweetID
        if text.find("@nadidoesart")>=1: 
            #print('SKIP')
            continue 
        d = pd.DataFrame.from_dict({'id':id,'text':text,'report':report},orient='index').T
        rep_list.append(d)
    print('Done appending')
    data = pd.concat(rep_list)
    print(data.shape)
    data.to_csv(out_file)
    
if __name__ == "__main__":  
    name = ['ramen']
    category = ['selfReport','noReport']
    for n in name: 
        for cat in category: 
            out_file = f'{n}_{cat}_data.csv'
            print('working on ')
            main_three(n,cat,out_file)