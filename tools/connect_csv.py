import pandas as pd
from sklearn.model_selection import train_test_split


csv_files = [
    '../data/Semeval2016/train/train_2015_10_22.utf-8/STSint.input.headlines.csv',
    '../data/Semeval2016/train/train_2015_10_22.utf-8/STSint.input.images.csv',
    '../data/Semeval2016/train/train_students_answers_2015_10_27.utf-8/STSint.input.answers-students.csv'
    ]

combined_data = pd.concat([pd.read_csv(file) for file in csv_files])

# combined_data.to_csv("../data/Semeval2016/train/train_healines_images_students.csv", index=False)
# print("All files have been combined and saved as csv")



validation_data, test_data = train_test_split(combined_data, test_size=0.3, random_state=42)

validation_data.to_csv("../data/Semeval2016/train/validation_healines_images_students.csv", index=False)
test_data.to_csv("../data/Semeval2016/test/test_healines_images_students.csv", index=False)

print("Validation and test sets have been created and saved.")
