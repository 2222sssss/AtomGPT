import pandas as pd
import json

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

        human_label_message = item["human_label_message"][0]["content"]
        text += f"<s>Assistant: {human_label_message}\n</s>"

        dialog_dict = {
            "text": text
        }
        dialogues.append(dialog_dict)

    return dialogues


def convert_list_to_csv(dialogues, csv_file_path):
    df = pd.DataFrame(dialogues)
    df.to_csv(csv_file_path, index=False)

if __name__ == "__main__":
    file_path = "./financial_xiaozhu/xiaozhu.json"
    csv_saved_path = "./financial_xiaozhu/xiaozhu_saved.csv"
    dialogues = convert_to_formatted_list(file_path)
    convert_list_to_csv(dialogues, csv_saved_path)

