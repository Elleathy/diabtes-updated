from Utiles import *
Data_Path = os.path.dirname("./Data/")
MOdel_Path = os.path.dirname("./Model/")
file_path =  os.path.join(Data_Path ,"diabetes.csv")
print("Loading data from:", file_path)

df = get_data(file_path)

data_info(df)
df = preproccessing_data(df)
df = removing_outliers(df)
df.to_csv(f"{Data_Path}diabetes_cleaned.csv", index=False)
print("Data preprocessed and saved to diabetes_cleaned.csv")

visualize_data(df)
print("Data visualization completed")

x_train, x_test, y_train, y_test  = split_data(df,'Outcome')

y_train , y_pred , y_train_pred = modeling(x_train, x_test, y_train)
print("Model trained and predictions made")

evaluate_model(y_test, y_pred, y_train, y_train_pred)
print("Model evaluation completed")

