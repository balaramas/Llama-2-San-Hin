import pandas as pd
from datasets import Dataset


data = pd.read_csv('./csv_files/train_sahi3.csv', header=None)


formatted_data = []

for index, row in data.iterrows():
    
    sanskrit_text = row[0].replace("\xa0","")
    hindi_text = row[1].replace("\xa0","")
    

    formatted_text = f"<s><INST> translate this text {sanskrit_text} from Sanskrit to Hindi </INST> {hindi_text} </s>"
    

    formatted_data.append({'text': formatted_text})

# Create a Dataset object
transformed_dataset = Dataset.from_dict({"text": [example['text'] for example in formatted_data]})


transformed_dataset.push_to_hub("sahikpsir_train") # name to push as