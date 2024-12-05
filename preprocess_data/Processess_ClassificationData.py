# ensures folders are ready for training yolov8 classification model
import os
import shutil
import pandas as pd

def reformat_folder(folder_name):
    folder_path = os.path.join(os.getcwd(), folder_name)
    classes_file = os.path.join(folder_path, '_classes.csv')

    df = pd.read_csv(classes_file)
    class_names = df.columns[1:]

    # make class folders
    for class_name in class_names:
        class_folder = os.path.join(folder_path, class_name)
        os.makedirs(class_folder, exist_ok=True)

    # move images to their class folders
    for index, row in df.iterrows():
        filename = row['filename']
        image_path = os.path.join(folder_path, filename)
        class_label = class_names[row[1:].tolist().index(1)]

        rename = f"{index}_{class_label}.jpg"
        new_path = os.path.join(folder_path, class_label, rename)

        if os.path.exists(image_path):
            shutil.move(image_path, new_path)

if __name__ == '__main__':
    reformat_folder('valid')
    reformat_folder('test')
    reformat_folder('train')