import pandas as pd
import json
import random
import argparse

def convert_to_formatted_list(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    dialogues = []
    for item in data:
        system_prefix = item["system_prefix"]
        text = ""
        if system_prefix:
            text += f"<s>System: {system_prefix}\n</s>"

        history_dialogue = item["input_message"]
        for dialog in history_dialogue:
            role = dialog["role"]
            content = dialog["content"]
            text += f"<s>{role}: {content}\n</s>"

        for item_human in item["human_label_message"]:
            human_label_message = item_human["content"]
            if len(human_label_message)>0:
                all_text =text+ f"<s>Assistant: {human_label_message}\n</s>"
                dialog_dict = {
                    "text": all_text
                }
                dialogues.append(dialog_dict)
        
        if len(item["human_label_message"])==0:
            for item_good in item["good_label_message"]:
                good_label_message = item_good["content"]
                if len(good_label_message)>0:
                    all_text =text+ f"<s>Assistant: {good_label_message}\n</s>"
                    dialog_dict = {
                        "text": all_text
                    }
                    dialogues.append(dialog_dict)

        
        
    return dialogues


def convert_list_to_csv(dialogues, csv_file_path):
    df = pd.DataFrame(dialogues)
    df.to_csv(csv_file_path, index=False,quoting=1)

def split_data(num_dev, dialogues):
    random.shuffle(dialogues)  # 随机打乱列表顺序
    dialogues_dev = dialogues[:num_dev]  # 前num_dev个作为dev集
    dialogues_train = dialogues[num_dev:]  # 剩下的作为train集
    return dialogues_train, dialogues_dev

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert JSON to CSV")
    parser.add_argument("--file_path", type=str, default="./financial_xiaozhu/xiaozhu.json",
                        help="Path to the input JSON file")
    parser.add_argument("--training_data_path", type=str, 
                        help="Path to save the training data")
    parser.add_argument("--dev_data_path", type=str,
                        help="Path to save the dev data")
    parser.add_argument("--num_dev", type=int, default=30,
                        help="The number of dev set")
    args = parser.parse_args()
    file_path = args.file_path
    training_data_path = args.training_data_path
    dev_data_path = args.dev_data_path
    num_dev = args.num_dev
    dialogues = convert_to_formatted_list(file_path)
    dialogues_train, dialogues_dev = split_data(num_dev, dialogues)
    convert_list_to_csv(dialogues_train, training_data_path)
    convert_list_to_csv(dialogues_dev, dev_data_path)


# python3 ./train/data_prepro/convert_for_atomhub.py --file_path ./financial_xiaozhu/xiaozhu.json --training_data_path ./financial_xiaozhu/training.csv --dev_data_path ./financial_xiaozhu/test.csv --num_dev 100
