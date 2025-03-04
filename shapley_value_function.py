def shapley_value(data_frame, input_model, iterations):
    feature_shapley_value_matrix = {}
    list_of_shapleys = []

    # Iterate over each column in the data frame (feature columns)
    for which_column in range(data_frame.shape[1]):
        feature_shapley_value_matrix[which_column] = []

        # Perform iterations to compute Shapley values
        for i in range(iterations):
            # Shuffle the rows of the data frame
            row_shuffled_dataframe = data_frame.sample(frac=1).reset_index(drop=True)

            # Select a random row from the shuffled data
            random_row = row_shuffled_dataframe.sample(n=1).reset_index(drop=True)

            # Make a copy of the row to alter
            random_row_altered = random_row.copy()

            # Alter the selected feature (column)
            random_row_altered.iloc[0, which_column] = data_frame.iloc[:, which_column].sample(n=1).values[0]

            # Calculate the difference in predictions
            original_prediction = input_model.predict(random_row)
            altered_prediction = input_model.predict(random_row_altered)
            difference = abs(original_prediction - altered_prediction)

            # Append the difference to the list for this feature
            feature_shapley_value_matrix[which_column].append(difference[0])

        # Compute the mean Shapley value for the feature
        mean_shapley_value = np.nanmean(feature_shapley_value_matrix[which_column])
        list_of_shapleys.append(round(mean_shapley_value, 3))

    return list_of_shapleys
shaps = shapley_value(X, model, 15)


values = shaps  # Heights of the bars
labels = X.columns.tolist()  # Labels for the x-axis


sorted_values, sorted_labels = zip(*sorted(zip(values, labels)))

# Create the bar plot
plt.bar(sorted_labels, sorted_values, color='green')
#plt.figure(figsize=(10, 6))
plt.grid(axis='y', linestyle='--', color='gray', alpha=0.3)

# Add labels and title (optional
#plt.bar(labels, shaps, width=.4)

# Add labels and title (optional)
plt.xlabel('features')
plt.ylabel('impacts')
plt.title('shapley_values')
plt.xticks(fontsize = 8,rotation=90)
# Show the plot
plt.show()
