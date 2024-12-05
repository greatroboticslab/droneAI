# roboflow has a max image count for free accounts so ensure label folders don't go over
import os
import random
import shutil

def balance_dataset(parent_folder, target_total=9500):
    labels = ['Crash', 'Flight', 'Landing', 'No signal', 'Started']
    target_per_folder = target_total // len(labels)
    total_images_selected = 0
    
    # go through each folder
    for label in labels:
        label_folder = os.path.join(parent_folder, label)
        if not os.path.exists(label_folder):
            print(f"Folder '{label}' not found in '{parent_folder}'. Skipping.")
            continue

        # get all the images
        image_files = [f for f in os.listdir(label_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(image_files)} images in '{label_folder}'.")

        # grab target amount of random images
        random.shuffle(image_files)
        selected_images = image_files[:target_per_folder]
        total_images_selected += len(selected_images)

        # delete other images
        for extra_image in image_files[target_per_folder:]:
            os.remove(os.path.join(label_folder, extra_image))
            print(f"Removed {extra_image} from {label_folder}.")

        print(f"Kept {len(selected_images)} images in '{label_folder}'.")

    print(f"Total images after balancing: {total_images_selected} (Target: {target_total}).")
    print("Dataset balanced successfully.")

def verify_image_counts(parent_folder):
    labels = ['Crash', 'Flight', 'Landing', 'No signal', 'Started']
    total_images = 0
    
    # count how many images there are in total before uploading to roboflow
    for label in labels:
        label_folder = os.path.join(parent_folder, label)
        
        if not os.path.exists(label_folder):
            print(f"Folder '{label}' not found in '{parent_folder}'. Skipping.")
            continue

        image_files = [f for f in os.listdir(label_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_count = len(image_files)
        total_images += image_count

        print(f"Folder '{label}': {image_count} images.")
    
    print(f"Total images across all folders: {total_images}")

if __name__ == '__main__':
    parent_folder = './labeled_frames'
    balance_dataset(parent_folder, target_total=9500) # roboflow has max 10k so to be safe do 9.5k
    verify_image_counts(parent_folder) # double check we didnt go over target_total
