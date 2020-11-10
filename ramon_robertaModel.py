
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split
from gary_utils import * 

def main(): 
    f_path = 'Breast Cancer(Raw_data_2_Classes).csv'
    data = loadDataAsDataFrame(f_path)
    X=data
    y=data['Class'].tolist()
    training_set_size = int(0.8*len(X))
    training_rows, test_rows, training_classes, test_classes = train_test_split(
    X, y, train_size=training_set_size, random_state=42069)
    training_rows, test_rows, training_classes, test_classes = train_test_split(X, y, train_size=training_set_size, random_state=42069)
    model_args={'overwrite_output_dir':True}
    # Create a TransformerModel
    model = ClassificationModel('roberta', 'roberta-base', use_cuda=False, args=model_args)
    #model = ClassificationModel('roberta', 'roberta-base', use_cuda=True, args=model_args)

    #change our data into a format that simpletransformers can process
    training_rows['text']=training_rows['Text']
    training_rows['labels']=training_rows['Class']
    test_rows['text']=test_rows['Text']
    test_rows['labels']=test_rows['Class']

    # Train the model
    model.train_model(training_rows)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(test_rows)

    print("f1 score")
    precision=result['tp'] / (result['tp'] + result['fp'])
    recall=result['tp'] / (result['tp'] + result['fn'])
    f1score= 2 * precision * recall / (precision + recall)
    print(f1score)

if __name__=="__main__":
    main() 