import csv


file_path = r'F:\学校\课程文件\dl lab\IDRID_dataset\labels\train.csv'

filtered_image_names_0 = []
filtered_image_names_1 = []
filtered_image_names_2 = []
filtered_image_names_3 = []
filtered_image_names_4 = []

#open csv file
with open(file_path,  encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file) #csv_reader is an iterator
    header = next(csv_reader)  # read the header
    #print("Header:", header)

    # find the column index of "Image name" and "Retinopathy grade"
    image_name_index = header.index("Image name")
    grade_index = header.index("Retinopathy grade")


    for row in csv_reader: # read each row of it
        if row[grade_index] == '0':
            filtered_image_names_0.append(row[image_name_index])
        elif row[grade_index] == '1':
            filtered_image_names_1.append(row[image_name_index])
        elif row[grade_index] == '2':
            filtered_image_names_2.append(row[image_name_index])
        elif row[grade_index] == '3':
            filtered_image_names_3.append(row[image_name_index])
        elif row[grade_index] == '4':
            filtered_image_names_4.append(row[image_name_index])

for i in range(5):
    name = f"filtered_image_names_{i}"

    print(f"{name}: {globals()[name]}")


