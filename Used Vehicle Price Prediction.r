# TABLE OF CONTENTS

# 1. LOAD PACKAGES, DATASET AND IMPUTE MISSING VALUES
  # 1.1. Load necessary packages
  # 1.2. Load data set

# 2. DATA PRE-PROCESSING
  # 2.1. Remove outliers
  # 2.2. Create train and test data sets and out-of-time data set

# 3. DATA EXPLORATION
  # 3.1. Distribution by Make
  # 3.2. Disdtribution by Model
  # 3.3. Distribution by City and Average Price by City
  # 3.4. Distribution of cars by year
  # 3.5. Distribution of cars by Mileage
  # 3.6. Distribution of cars by Engine Displacement vs Year
  # 3.7. Distribution of Price
  # 3.8. Average Price per Year

# 4. MODEL DEVELOPMENT AND PERFORMANCE ANALYSIS
  # 4.1. Benchmark Model
  # 4.2. Linear Regression Model
  # 4.3. Linear Regression Model - Corolla Car Model
  # 4.4. Machine Learning model: kNN
  # 4.5. Machine Learning model: Random Forest
  # 4.6. Machine Learning model: LightGBM
  # 4.7. Machine Learning model: Xgboost

# 5. MODEL VALIDATION


###############################################################################
# 1. LOAD PACKAGES, DATASET AND IMPUTE MISSING VALUES                         #
###############################################################################
  # 1.1. Load necessary packages
  # 1.2. Load data set

###############################################################################
# 1.1. Load necessary packages
###############################################################################

if (!require(rstudioapi)) install.packages("rstudioapi", repos = "http://cran.us.r-project.org")
if (!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if (!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if (!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if (!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if (!require(cluster)) install.packages("cluster", repos = "http://cran.us.r-project.org")
if (!require(class)) install.packages("class", repos = "http://cran.us.r-project.org")
if (!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if (!require(lightgbm)) install.packages("lightgbm", repos = "http://cran.us.r-project.org")
if (!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if (!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if (!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")

###############################################################################
# 1.2. Load data set 
# ***NTR: Save the file "used_cars_Pakistan.csv" in the working directory***
###############################################################################

# Get the current script's directory
script_directory <- dirname(rstudioapi::getActiveDocumentContext()$path)

# Set the script directory as the working directory
setwd(script_directory)

# Construct the file path to the CSV file
csv_file_path <- file.path(script_directory, "used_cars_Pakistan.csv")

# Read the CSV file into a data frame
data <- read.csv(csv_file_path)

# Check the structure of the data frame to ensure it loaded correctly
str(data)
# 'data.frame':	86120 obs. of  9 variables:

# Convert to US dollars using exchange rate: 0.0034
data$Price_Us <- data$Price_Rs * 0.0034
str(data)
# 'data.frame':	86120 obs. of  10 variables:

# Subset the data to include relevant features and target variable (Price_Us)
data_ml <- data %>% select(Price_Us, make, year, mileage, Engine_displacement, model, city)
str(data_ml)
# 'data.frame':	86120 obs. of  7 variables:

# Check for missing values in each column and count them
missing_values <- sum(is.na(data_ml))
missing_values <- colSums(is.na(data_ml))

# Find the columns with missing values
columns_with_missing <- names(missing_values[missing_values > 0])

# Print the names of columns with missing values and their missing counts
for (col in columns_with_missing) {
  cat("Column:", col, "- Missing Values:", missing_values[col], "\n")
}
# Column: Engine_displacement - Missing Values: 189 

# Impute missing values in Engine_displacement

# Calculate the mean of the non-missing values in the "Engine_displacement" column
mean_engine_displacement <- mean(data_ml$Engine_displacement, na.rm = TRUE)

# Impute missing values with the mean
data_ml$Engine_displacement[is.na(data_ml$Engine_displacement)] <- mean_engine_displacement

# Check for missing values
missing_values <- sum(is.na(data_ml))
missing_values
# [1] 0

###############################################################################
# 2. DATA PRE-PROCESSING                                                      #
###############################################################################
  # 2.1. Remove outliers
  # 2.2. Create train and test data sets and out-of-time data set

###############################################################################
# 2.1. Remove outliers
###############################################################################

# 2.1.1. Remove makes with less than 100 cars to remove small sample size

# Calculate the frequency of each make
make_frequency <- table(data_ml$make)

# Convert the frequency table to a data frame
make_freq_df <- data.frame(make = names(make_frequency), Freq = as.numeric(make_frequency))

# Filter the data to keep only makes with at least 100 cars
data_ml <- data_ml[data_ml$make %in% make_freq_df$make[make_freq_df$Freq >= 100], ]

str(data_ml)
# 'data.frame':	85144 obs. of  7 variables:

# 2.1.2. Remove Price outliers using z_scores

mean(data_ml$Price_Us)
#[1] 13951.91

# Calculate the Z-scores for the 'Price_Us' variable
z_scores <- scale(data_ml$Price_Us)

# Define a threshold for Z-scores
threshold <- 3

# Remove rows with Z-scores above the threshold
data_ml <- data_ml[abs(z_scores) <= threshold, ]
mean(data_ml$Price_Us)
#[1] 12790.49

# 2.1.3. Remove data with negative price

# Identify the rows with negative price values
negative_price_years <- data_ml[data_ml$Price_Us < 0, "year"]
#[1] 1987

# Remove records from the year 1987
data_ml <- data_ml[data_ml$year != 1987, ]

str(data_ml)
# 'data.frame':	84324 obs. of  7 variables:

mean(data_ml$Price_Us)
#[1] 12819.99

# Save data_ml as a CSV file in the current working directory
write.csv(data_ml, file = "data_ml.csv")

###############################################################################
# 2.2. Create train and test data sets and out-of-time data set
###############################################################################

# Split data into out-of-time hold_out (2021 - 2023), train and test data sets (<= 2020)
hold_out_data <- data_ml[data_ml$year > 2020, ]
non_hold_out_data <- data_ml[data_ml$year <= 2020, ]

set.seed(1974)
splitIndex <- createDataPartition(non_hold_out_data$Price_Us, p = 0.8, list = FALSE)
train_data <- non_hold_out_data[splitIndex, ]
test_data <- non_hold_out_data[-splitIndex, ]

str(hold_out_data)
# 'data.frame':	16668 obs. of  7 variables:
str(train_data)
# 'data.frame':	54127 obs. of  7 variables:
str(test_data)
# 'data.frame':	13529 obs. of  7 variables:

# Save hold_out_data as a CSV file
write.csv(hold_out_data, file = "hold_out_data.csv")

# Save train_data as a CSV file
write.csv(train_data, file = "train_data.csv")

# Save test_data as a CSV file
write.csv(test_data, file = "test_data.csv")

###############################################################################
# 3. DATA EXPLORATION                                                         #
###############################################################################
  # 3.1. Distribution by Make
  # 3.2. Disdtribution by Model
  # 3.3. Distribution by City and Average Price by City
  # 3.4. Distribution of cars by year
  # 3.5. Distribution of cars by Mileage
  # 3.6. Distribution of cars by Engine Displacement vs Year
  # 3.7. Distribution of Price
  # 3.8. Average Price per Year

###############################################################################
# 3.1. Distribution by Make
###############################################################################

# Calculate the frequency of each make
make_frequency <- table(data_ml$make)

# Convert the frequency table to a data frame and order in descending order
top_makes <- as.data.frame(make_frequency)
top_makes <- top_makes[order(-top_makes$Freq), ]

# Select the top 20 makes
top_makes <- head(top_makes, 20)

# Create a histogram of the top 20 makes
plot_3_1 <- ggplot(top_makes, aes(x = reorder(Var1, -Freq), y = Freq)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(x = "Make", y = "Frequency", title = "Top 20 Makes by Frequency") +
  theme_economist_white() +
  theme(axis.text.x = element_text(angle = 90, hjust = 0))

# Save the plot in the working directory
ggsave("plot_3_1.png", plot = plot_3_1, width = 8, height = 6, units = "in")

###############################################################################
# 3.2. Disdtribution by Model
###############################################################################

# Calculate the frequency of each model
model_frequency <- table(data_ml$model)

# Convert the frequency table to a data frame
model_freq_df <- data.frame(model = names(model_frequency), Freq = as.numeric(model_frequency))

# Order the data frame by frequency in descending order
model_freq_df <- model_freq_df[order(-model_freq_df$Freq), ]

# Select the top 20 models
top_models <- head(model_freq_df, 20)

# Convert 'model' to a factor with levels ordered by frequency
data_ml$model <- factor(data_ml$model, levels = top_models$model)

# Create a histogram of the top 20 models
plot_3_2 <- ggplot(data_ml, aes(x = model, fill = model)) +
  geom_bar() +
  labs(x = "Model", y = "Frequency", title = "Top 20 Models by Frequency") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Save the plot in the working directory
ggsave("plot_3_2.png", plot = plot_3_2, width = 8, height = 6, units = "in")

###############################################################################
# 3.3. Distribution by City and Average Price by City
###############################################################################

# Calculate the frequency of each city
city_frequency <- table(data_ml$city)

# Convert the frequency table to a data frame
city_freq_df <- data.frame(city = names(city_frequency), Freq = as.numeric(city_frequency))

# Order the data frame by frequency in descending order
city_freq_df <- city_freq_df[order(-city_freq_df$Freq), ]

# Select the top 20 cities
top_cities <- head(city_freq_df, 20)

# Calculate the average price for each of the top 20 cities
average_prices <- sapply(top_cities$city, function(city_name) {
  mean(data_ml[data_ml$city == city_name, "Price_Us"], na.rm = TRUE)
})

# Add the average prices to the top_cities data frame
top_cities$AveragePrice <- average_prices

# Create a histogram of the top 20 cities by number of cars
plot_3_3 <- ggplot(top_cities, aes(x = reorder(city, -Freq), y = Freq)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(x = "City", y = "Number of Cars", title = "Top 20 Cities by # of Cars and Avg Price") +
  theme_economist_white() +
  theme(axis.text.x = element_text(angle = 90, hjust = 0)) +
  # Add a second y-axis for average price
  geom_line(aes(x = reorder(city, -Freq), y = AveragePrice * max(top_cities$Freq) / max(top_cities$AveragePrice), group = 1), color = "red") +
  geom_point(aes(x = reorder(city, -Freq), y = AveragePrice * max(top_cities$Freq) / max(top_cities$AveragePrice), group = 1), color = "red") +
  scale_y_continuous(sec.axis = sec_axis(~./max(top_cities$Freq)*max(top_cities$AveragePrice), name = "Average Price"))

# Save the plot in the working directory
ggsave("plot_3_3.png", plot = plot_3_3, width = 8, height = 6, units = "in")

###############################################################################
# 3.4. Distribution of cars by year
###############################################################################

# Create a histogram of the number of cars by year
plot_3_4 <- ggplot(data_ml, aes(x = year)) +
  geom_histogram(binwidth = 1, fill = "skyblue") +
  labs(x = "Year", y = "Number of Cars", title = "Number of Cars by Year") +
  theme_economist_white()

# Save the plot in the working directory
ggsave("plot_3_4.png", plot = plot_3_4, width = 8, height = 6, units = "in")

###############################################################################
# 3.5. Distribution of cars by Mileage
###############################################################################

plot_3_5 <- ggplot(data_ml, aes(x = year, y = mileage)) +
  geom_point(size = 2, color = "skyblue") +
  labs(x = "Year", y = "Mileage", title = "Scatter Plot of Mileage vs. Year") +
  theme_economist_white()

# Save the plot in the working directory
ggsave("plot_3_5.png", plot = plot_3_5, width = 8, height = 6, units = "in")

# 3.6. Distribution of cars by Engine Displacement vs Year

plot_3_6 <- ggplot(data_ml, aes(x = year, y = Engine_displacement)) +
  geom_point(size = 2, color = "skyblue") +
  labs(x = "Year", y = "Engine Displacement", 
       title = "Scatter Plot of Engine Displacement vs. Year") +
  theme_economist_white()

# Save the plot in the working directory
ggsave("plot_3_6.png", plot = plot_3_6, width = 8, height = 6, units = "in")

###############################################################################
# 3.7. Distribution of Price
###############################################################################

ggplot(data_ml, aes(x = Price_Us)) +
  geom_histogram(binwidth = 1000, fill = "skyblue", color = "black") +
  labs(x = "Price (Us)", y = "Frequency", title = "Price Distribution") +
  theme_economist_white()

plot_3_7 <- ggplot(data_ml, aes(x = year, y = Price_Us)) +
  geom_point(size = 2, color = "skyblue") +
  labs(x = "Year", y = "Price (Us)", title = "Scatter Plot of Price vs. Year") +
  theme_economist_white()

# Save the plot in the working directory
ggsave("plot_3_7.png", plot = plot_3_7, width = 8, height = 6, units = "in")

###############################################################################
# 3.8. Average Price per Year
###############################################################################

# Calculate the average Price_Us by year
average_price_by_year <- aggregate(Price_Us ~ year, data_ml, mean)

# Create a line plot of the average Price_Us over the years
plot_3_8 <- ggplot(average_price_by_year, aes(x = year, y = Price_Us)) +
  geom_line(color = "blue") +
  geom_vline(xintercept = 2020, linetype = "dotted", color = "red") +  # Add the vertical line
  labs(x = "Year", y = "Average Price (Us)", title = "Average Price vs. Year")

# Save the plot in the working directory
ggsave("plot_3_8.png", plot = plot_3_8, width = 8, height = 6, units = "in")

###############################################################################
# 4. MODEL DEVELOPMENT                                                        #
###############################################################################
  # 4.1. Benchmark Model
  # 4.2. Linear Regression Model
  # 4.3. Linear Regression model using Machine Learning clusters
  # 4.4. Machine Learning model: kNN
  # 4.5. Machine Learning model: Random Forest (Adding categorical variables)
  # 4.6. LightGBM model
  # 4.7. Machine Learning model: Xgboost

###############################################################################
# 4.1. Benchmark Model                                                        #
###############################################################################

# Calculate the mean for car prices
mean_price <- mean(train_data$Price_Us)
mean_price
#[1] 10673.1

# Create vectors of mean predictions for the test data
mean_predictions <- rep(mean_price, nrow(test_data))

# Calculate RMSE for mean predictions
rmse_mean <- sqrt(mean((mean_predictions - test_data$Price_Us)^2))
rmse_mean
#[1] 11304.45

# Calculate residuals for mean predictions
mean_residuals <- test_data$Price_Us - mean_predictions

# Create a histogram of residuals for mean predictions
hist(mean_residuals, main="Distribution of Residuals for Benchmark Model",
     xlab="Residuals", ylab="Frequency", col="#028090") +
  theme_economist_white()

# Create a density plot
plot_4_1 <- ggplot(data.frame(Residuals = mean_residuals), aes(x = Residuals)) +
  geom_density(fill = "lightblue", alpha = 0.5) +
  labs(title = "Density of Residuals for Benchmark Model", x = "Residuals") +
  theme_economist_white()

# Save the plot in the working directory
ggsave("plot_4_1.png", plot = plot_4_1, width = 8, height = 6, units = "in")

# Create a data frame with actual and predicted values
scatter_data_4_1 <- data.frame(
  Actual_Price_Us = test_data$Price_Us,
  Predicted_Price_Us = mean_predictions
)

# Create the scatter plot with equal scales
scatter_4_1 <- ggplot(scatter_data_4_1, aes(x = Actual_Price_Us, y = Predicted_Price_Us)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(
    title = "Actual vs. Predicted Price_Us Scatter Plot Benchmark Model",
    x = "Actual Price_Us",
    y = "Predicted Price_Us"
  )

# Save the plot in the working directory
ggsave("scatter_4_1.png", plot = scatter_4_1, width = 8, height = 6, units = "in")

# Calculate the KS statistic
ks_statistic <- ks.test(mean_predictions, test_data$Price_Us)$statistic
print(ks_statistic)
#0.5481461

# Create data frames for predicted and actual values
cdf_data <- data.frame(
  Predicted = mean_predictions,
  Actual = test_data$Price_Us
)

# Create the CDF plot
cdf_plot_4_1 <- ggplot(cdf_data, aes(x = Predicted)) +
  stat_ecdf(aes(color = "Predicted"), geom = "step") +
  geom_step(data = cdf_data, aes(x = Actual, color = "Actual"), stat = "ecdf") +
  labs(title = "CDF of Predicted vs Actual Prices Benchmark Model",
       x = "Price",
       y = "Cumulative Probability") +
  theme_economist_white() +
  scale_color_manual(values = c("Predicted" = "#456990", "Actual" = "#F45B69")) +
  annotate("text", x = max(cdf_data$Predicted), y = 0.2, label = paste("KS Statistic =", round(ks_statistic, 4)), color = "black")

# Save the plot in the working directory
ggsave("cdf_plot_4_1.png", plot = cdf_plot_4_1, width = 8, height = 6, units = "in")

###############################################################################
# 4.2. Linear Regression Model                                                #
###############################################################################

# 4.2.1. Analyze correlation in numerical variables

# Select the numerical variables
data_ml_num_vars <- train_data[, c("Price_Us", "year", "mileage", "Engine_displacement")]
str(train_data)
str(data_ml_num_vars)

# Calculate the correlation matrix
correlation_matrix <- cor(data_ml_num_vars, use = "complete.obs")

# Print the correlation matrix
print(correlation_matrix)

# Save the correlation matrix as a CSV file
write.csv(correlation_matrix, file = "correlation_matrix.csv")

# Print the table using kable
knitr::kable(correlation_matrix, format = "markdown")

#|                    |   Price_Us|       year|    mileage| Engine_displacement|
#  |:-------------------|----------:|----------:|----------:|-------------------:|
#  |Price_Us            |  1.0000000|  0.3444260| -0.1282792|           0.4932434|
#  |year                |  0.3444260|  1.0000000| -0.2683262|          -0.1381026|
#  |mileage             | -0.1282792| -0.2683262|  1.0000000|           0.0773300|
#  |Engine_displacement |  0.4932434| -0.1381026|  0.0773300|           1.0000000|

# Fit a linear regression model for Price_Us
lm_model <- lm(Price_Us ~ ., data = data_ml_num_vars)
print(lm_model)
#Coefficients:
#   (Intercept)                 year              mileage  Engine_displacement  
#     -1.078e+06            5.362e+02           -7.755e-03            8.351e+00

# Create a scatter plot to visualize Price vs. Year correlation
plot_4_2_1 <- plot(data_ml$year, data_ml$Price_Us, 
     xlab = "Year", ylab = "Price (Us)", 
     main = "Linear Regression: Price vs. Year") +
  abline(lm_model, col = "red")

# Predict prices using the linear regression model
predicted_prices <- predict(lm_model, newdata = test_data)

# Calculate the residuals between predicted and actual prices
residuals <- test_data$Price_Us - predicted_prices

# Calculate the RMSE
rmse_lm <- sqrt(mean(residuals^2))
print(rmse_lm)
#[1] 8492.5

# Create a histogram of residuals
hist(residuals, main = "Distribution of Residuals", 
     xlab = "Residuals", col = "lightblue", border = "black")

# Create a density plot
plot_4_2 <- ggplot(data.frame(Residuals = residuals), aes(x = Residuals)) +
  geom_density(fill = "lightblue", alpha = 0.5) +
  labs(title = "Density of Residuals for Linear Regression Model", x = "Residuals") +
  theme_economist_white() +
  theme(plot.title = element_text(size = 12))  

# Save the plot in the working directory
ggsave("plot_4_2.png", plot = plot_4_2, width = 8, height = 6, units = "in")
  
# Create a data frame with actual and predicted values
scatter_data_4_2 <- data.frame(
  Actual_Price_Us = test_data$Price_Us,
  Predicted_Price_Us = predicted_prices
)

# Create the scatter plot with equal scales
scatter_4_2 <- ggplot(scatter_data_4_2, aes(x = Actual_Price_Us, y = Predicted_Price_Us)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(
    title = "Actual vs. Predicted Price_Us Scatter Plot Linear Regression",
    x = "Actual Price_Us",
    y = "Predicted Price_Us"
  )

# Save the plot in the working directory
ggsave("scatter_4_2.png", plot = scatter_4_2, width = 8, height = 6, units = "in")

# Calculate the KS statistic
ks_statistic_lm <- ks.test(predicted_prices, test_data$Price_Us)$statistic
print(ks_statistic_lm)
#0.177101 

# Create data frames for predicted and actual values
cdf_data_4_2<- data.frame(
  Predicted = predicted_prices,
  Actual = test_data$Price_Us
)

# Create the CDF plot
cdf_plot_4_2 <- ggplot(cdf_data_4_2, aes(x = Predicted)) +
  stat_ecdf(aes(color = "Predicted"), geom = "step") +
  geom_step(data = cdf_data, aes(x = Actual, color = "Actual"), stat = "ecdf") +
  labs(title = "CDF of Predicted vs Actual Prices Linear Model",
       x = "Price",
       y = "Cumulative Probability") +
  theme_economist_white() +
  scale_color_manual(values = c("Predicted" = "#456990", "Actual" = "#F45B69")) +
  annotate("text", x = max(cdf_data$Predicted), y = 0.2, label = paste("KS Statistic =", round(ks_statistic_lm, 4)), color = "black")

# Save the plot in the working directory
ggsave("cdf_plot_4_2.png", plot = cdf_plot_4_2, width = 8, height = 6, units = "in")

# NOTE: Option to evaluate segmentation by car model

###############################################################################
# 4.3. Linear Regression Model - Corolla Car Model                            #
###############################################################################

# Calculate correlation for variables only for the largest car model (Corolla)

# Filter the data for the "Corolla" model
train_data_corolla  <- subset(test_data, model == "Corolla")
test_data_corolla <- subset(test_data, model == "Corolla")
corolla_data_ml <- subset(data_ml, model == "Corolla")

# Calculate model benchmark for Corolla - simple average

# Calculate the mean for car prices
mean_price <- mean(train_data_corolla$Price_Us)
mean_price
#[1] 10811.93

# Create vectors of mean predictions for the test data
mean_predictions_corolla <- rep(mean_price, nrow(test_data_corolla))

# Calculate RMSE for mean predictions
rmse_mean_corolla <- sqrt(mean((mean_predictions_corolla - test_data_corolla$Price_Us)^2))
rmse_mean_corolla
#[1] 4852.834
# NOTE: Linear Model for Corolla only must surpass this RMSE

# Select the numerical variables
corolla_data_ml_num_vars <- train_data_corolla[, 
                                            c("Price_Us", "year", "mileage", "Engine_displacement")]

# Calculate the correlation matrix
correlation_matrix <- cor(corolla_data_ml_num_vars, use = "complete.obs")
correlation_matrix

# Save the correlation matrix as a CSV file
write.csv(correlation_matrix, file = "correlation_matrix_corolla.csv")

#                     Price_Us        year    mileage Engine_displacement
#Price_Us             1.0000000  0.79742212 -0.4928801          0.21359110
#year                 0.7974221  1.00000000 -0.3940537          0.07205596
#mileage             -0.4928801 -0.39405368  1.0000000         -0.09737100
#Engine_displacement  0.2135911  0.07205596 -0.0973710          1.00000000
# NOTE: Correlations are stronger for year & mileage in a unique car model,
# and weaker for Engine_displacement (as expected, i.e., car models tend to have similar)

# Fit a linear regression model for Price_Us
lm_model_corolla <- lm(Price_Us ~ ., data = corolla_data_ml_num_vars)
print(lm_model_corolla)
#Call:
#lm(formula = Price_Us ~ ., data = corolla_data_ml_num_vars)
#
#Coefficients:
#  (Intercept)                 year              mileage  Engine_displacement  
#-1.019e+06            5.118e+02           -1.387e-02            2.184e+00  

# Predict prices using the linear regression model
predicted_prices_corolla <- predict(lm_model_corolla, newdata = test_data_corolla)

# Calculate the residuals between predicted and actual prices
residuals_corolla <- test_data_corolla$Price_Us - predicted_prices_corolla

# Calculate the RMSE for lm_model_corolla
rmse_corolla <- sqrt(mean(residuals_corolla^2))
print(rmse_corolla)
#[1] 2748.727
# NOTE: RMSE is lower than benchmark of 4852.834

# Create a histogram of residuals
hist(residuals_corolla, main = "Distribution of Residuals - Corolla", 
     xlab = "Residuals", col = "lightblue", border = "black")

# Create a density plot
plot_4_3 <- ggplot(data.frame(Residuals = residuals_corolla), aes(x = Residuals)) +
  geom_density(fill = "lightblue", alpha = 0.5) +
  labs(title = "Density of Residuals for LM Model - Corolla", x = "Residuals") +
  theme_economist_white() +
  theme(plot.title = element_text(size = 12))  

# Save the plot in the working directory
ggsave("plot_4_3.png", plot = plot_4_3, width = 8, height = 6, units = "in")

# Create a ggplot scatter plot (correlation)
correlation_4_3 <- ggplot(data = corolla_data_ml, aes(x = year, y = Price_Us)) +
  geom_point() +  # Scatter plot
  geom_smooth(method = "lm", color = "red", se = FALSE) +  # Add linear regression line
  labs(x = "Year", y = "Price (Us)", title = "Linear Regression: Price vs. Year (Corolla)")

# Save the plot in the working directory
ggsave("correlation_4_3.png", plot = correlation_4_3, width = 8, height = 6, units = "in")

# Create a data frame with actual and predicted values
scatter_data <- data.frame(
  Actual_Price_Us = test_data_corolla$Price_Us,
  Predicted_Price_Us = predicted_prices_corolla
)

# Create the scatter plot with equal scales
scatter_4_3 <- ggplot(scatter_data, aes(x = Actual_Price_Us, y = Predicted_Price_Us)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(
    title = "Actual vs. Predicted Price_Us Scatter Plot (Corolla)",
    x = "Actual Price_Us",
    y = "Predicted Price_Us"
  )

# Save the plot in the working directory
ggsave("scatter_4_3.png", plot = scatter_4_3, width = 8, height = 6, units = "in")

# Calculate the KS statistic
ks_statistic_corolla <- ks.test(predicted_prices_corolla, test_data_corolla$Price_Us)$statistic
print(ks_statistic_corolla)
#0.1635581 

# Create data frames for predicted and actual values
cdf_data_4_3 <- data.frame(
  Predicted = predicted_prices_corolla,
  Actual = test_data_corolla$Price_Us
)

# Create the CDF plot
cdf_plot_4_3 <- ggplot(cdf_data_4_3, aes(x = Predicted)) +
  stat_ecdf(aes(color = "Predicted"), geom = "step") +
  geom_step(data = cdf_data, aes(x = Actual, color = "Actual"), stat = "ecdf") +
  labs(title = "CDF of Predicted vs Actual Prices LM Corolla",
       x = "Price",
       y = "Cumulative Probability") +
  theme_economist_white() +
  scale_color_manual(values = c("Predicted" = "#456990", "Actual" = "#F45B69")) +
  annotate("text", x = max(cdf_data$Predicted), y = 0.2, label = paste("KS Statistic =", round(ks_statistic_corolla, 4)), color = "black")

# Save the plot in the working directory
ggsave("cdf_plot_4_3.png", plot = cdf_plot_4_3, width = 8, height = 6, units = "in")

###############################################################################
# 4.4. Machine Learning model: kNN                                            #
###############################################################################

# Read the "train_data.csv" file from the current working directory
train_data <- read.csv("train_data.csv")

# Read the "test_data.csv" file from the current working directory
test_data <- read.csv("test_data.csv")

# Data pre processing - addressing missing values - replace with mean value

# Check for missing values in the training data
missing_values_train <- sum(is.na(train_data))
missing_values_test <- sum(is.na(test_data))

# Impute missing values with mean for numeric columns
train_data <- train_data %>%
  mutate_if(is.numeric, ~ifelse(is.na(.), mean(., na.rm = TRUE), .))
test_data <- test_data %>%
  mutate_if(is.numeric, ~ifelse(is.na(.), mean(., na.rm = TRUE), .))

missing_values_train <- sum(is.na(train_data$Price_Us))
missing_values_test <- sum(is.na(test_data))

class(train_data$Price_Us)
str(train_data)

# Label encode character variables
train_data$model <- as.integer(factor(train_data$model))
train_data$make <- as.integer(factor(train_data$make))
train_data$city <- as.integer(factor(train_data$city))
str(train_data)
test_data$model <- as.integer(factor(test_data$model))
test_data$make <- as.integer(factor(test_data$make))
test_data$city <- as.integer(factor(test_data$city))
str(test_data)

# Define the number of neighbors (K)
k <- 5

# Train the KNN regression model
knn_model <- knn(
  train = train_data[, -1], test = test_data[, -1], cl = train_data$Price_Us, k = k)
class(knn_model)

# Save the kNN model to a file
saveRDS(knn_model, file = "knn_model.rds")

# Load the saved kNN model
knn_model <- readRDS("knn_model.rds")

# Convert knn_model to numeric
knn_model <- as.numeric(knn_model)

# Convert test_data$Price_Us to numeric
test_data$Price_Us <- as.numeric(test_data$Price_Us)

# Calculate the Mean Absolute Error (MAE) to evaluate the model's performance
mae <- mean(abs(knn_model - test_data$Price_Us))
mae
#[1] 10068.13

# Calculate RMSE for all data points
rmse_all <- sqrt(mean((knn_model - test_data$Price_Us)^2))
rmse_all
#[1] 14892.31

# Calculate residuals
residuals <- knn_model - test_data$Price_Us

# Create a histogram to visualize the distribution of residuals
hist(residuals, main="Distribution of Residuals", xlab="Residuals")

# Create a density plot
plot_4_4<- ggplot(data.frame(Residuals = residuals), aes(x = Residuals)) +
  geom_density(fill = "darkred", alpha = 0.5) +
  labs(title = "Density of Residuals for kNN Model", x = "Residuals") +
  theme_economist_white() +
  theme(plot.title = element_text(size = 12))  

# Save the plot in the working directory
ggsave("plot_4_4.png", plot = plot_4_4, width = 8, height = 6, units = "in")

# Create a data frame with actual and predicted values
combined_data <- data.frame(
  Actual_Price_Us = test_data$Price_Us,
  Predicted_Price_Us = knn_model
)

# Create a scatter plot comparing the actual and predicted values
scatter_4_4 <- ggplot(combined_data, aes(x = Actual_Price_Us, y = Predicted_Price_Us)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Actual vs. Predicted Price_Us kNN Model",
       x = "Actual Price_Us",
       y = "Predicted Price_Us")

# Save the plot in the working directory
ggsave("scatter_4_4.png", plot = scatter_4_4, width = 8, height = 6, units = "in")

# Calculate the KS statistic
ks_statistic_knn <- ks.test(knn_model, test_data$Price_Us)$statistic
print(ks_statistic_knn)
#0.9628945 

# Create data frames for predicted and actual values
cdf_data_4_4 <- data.frame(
  Predicted = knn_model,
  Actual = test_data$Price_Us
)

# Create the CDF plot
cdf_plot_4_4 <- ggplot(cdf_data_4_4, aes(x = Predicted)) +
  stat_ecdf(aes(color = "Predicted"), geom = "step") +
  geom_step(data = cdf_data, aes(x = Actual, color = "Actual"), stat = "ecdf") +
  labs(title = "CDF of Predicted vs Actual Prices kNN Model",
       x = "Price",
       y = "Cumulative Probability") +
  theme_economist_white() +
  scale_color_manual(values = c("Predicted" = "#456990", "Actual" = "#F45B69")) +
  annotate("text", x = max(cdf_data$Predicted), y = 0.2, label = paste("KS Statistic =", round(ks_statistic_knn, 4)), color = "black")

# Save the plot in the working directory
ggsave("cdf_plot_4_4.png", plot = cdf_plot_4_4, width = 8, height = 6, units = "in")

###############################################################################
# 4.5. Machine Learning model: Random Forest (Adding categorical variables)   #
###############################################################################

# Read the "data_ml.csv" file from the current working directory
data_ml <- read.csv("data_ml.csv")

# Subset the data to include relevant features and target variable (Price_Us)
subset_data <- data_ml %>% select(Price_Us, year, mileage, Engine_displacement, model)

# Check structure
str(subset_data)

# Save subset_data as a CSV file in the current working directory
write.csv(subset_data, file = "subset_data.csv")

# Read the "subset_data.csv" file from the current working directory
subset_data <- read.csv("subset_data.csv")
subset_data <- subset_data %>% select(Price_Us, year, mileage, Engine_displacement, model)
str(subset_data)

# Impute missing values in the "model" feature with the most frequent category (model)
most_frequent_model <- as.character(levels(subset_data$model)[which.max(table(subset_data$model))])
subset_data$model[is.na(subset_data$model)] <- most_frequent_model

missing_values <- sum(is.na(subset_data))
missing_values

# Check for missing values in each column and count them
missing_values <- colSums(is.na(subset_data))

# Find the columns with missing values
columns_with_missing <- names(missing_values[missing_values > 0])

# Print the names of columns with missing values and their corresponding missing counts
for (col in columns_with_missing) {
  cat("Column:", col, "- Missing Values:", missing_values[col], "\n")
}

# Calculate the mean of the non-missing values in the "Engine_displacement" column
mean_engine_displacement <- mean(subset_data$Engine_displacement, na.rm = TRUE)

# Impute missing values with the mean
subset_data$Engine_displacement[is.na(subset_data$Engine_displacement)] <- mean_engine_displacement

missing_values <- sum(is.na(subset_data))
missing_values
# [1] 0

# Save subset_data as a CSV file in the current working directory
write.csv(subset_data, file = "subset_data_clusters.csv")

# Group categories with low frequencies into an "Other" category
subset_data$model <- as.character(subset_data$model)
top_categories <- names(sort(table(subset_data$model), decreasing = TRUE))[1:20]
subset_data$model[!(subset_data$model %in% top_categories)] <- "Other"

# Convert the "model" feature back to a factor
subset_data$model <- as.factor(subset_data$model)

# Split the data into training and testing sets
set.seed(1974)
sample_indices <- sample(1:nrow(subset_data), 0.7 * nrow(subset_data))
train_data <- subset_data[sample_indices, ]
test_data <- subset_data[-sample_indices, ]
str(train_data)
str(test_data)

# Train the Random Forest regression model ### TAKES 5 MIN TO RUN ###
rf_model <- randomForest(Price_Us ~ ., data = train_data)

# Save the rf_model object to the current working directory
saveRDS(rf_model, file = "rf_model.rds")

# Read the "rf_model.rds" file from the current working directory
rf_model <- readRDS("rf_model.rds")

# Make predictions on the test data
rf_predictions <- predict(rf_model, newdata = test_data)

# Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
mae_rf <- mean(abs(rf_predictions - test_data$Price_Us))
rmse_rf <- sqrt(mean((rf_predictions - test_data$Price_Us)^2))
mae_rf
#[1] 2078.378
rmse_rf
#[1] 5206.832

# Calculate the residuals
residuals <- test_data$Price_Us - rf_predictions

# Plot a histogram of the residuals
hist(residuals, main = "Distribution of Residuals", xlab = "Residuals", col = "skyblue")

# Create a density plot
plot_4_5 <- ggplot(data = data.frame(residuals = residuals), aes(x = residuals)) +
  geom_density(fill = "skyblue") +
  labs(title = "Distribution of Residuals Random Forest Model", x = "Residuals") +
  theme(plot.title = element_text(size = 12))  

# Save the plot in the working directory
ggsave("plot_4_5.png", plot = plot_4_5, width = 8, height = 6, units = "in")

# Create a data frame with actual and predicted Price_Us values
scatter_data <- data.frame(Actual = test_data$Price_Us, Predicted = rf_predictions)

# Create a scatter plot
scatter_4_5 <- ggplot(data = scatter_data, aes(x = Actual, y = Predicted)) +
  geom_point(color = "black", alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Scatter Plot of Actual vs. Predicted Price_Us RF Model",
       x = "Actual Price_Us",
       y = "Predicted Price_Us") +
  theme_economist_white() +
  theme(plot.title = element_text(size = 12))  

# Save the plot in the working directory
ggsave("scatter_4_5.png", plot = scatter_4_5, width = 8, height = 6, units = "in")

# Calculate the KS statistic
ks_statistic_rf <- ks.test(rf_predictions, test_data$Price_Us)$statistic
print(ks_statistic_rf)
#0.04656495

# Create data frames for predicted and actual values
cdf_data_4_5 <- data.frame(
  Predicted = rf_predictions,
  Actual = test_data$Price_Us
)

# Create the CDF plot
cdf_plot_4_5 <- ggplot(cdf_data_4_5, aes(x = Predicted)) +
  stat_ecdf(aes(color = "Predicted"), geom = "step") +
  geom_step(data = cdf_data, aes(x = Actual, color = "Actual"), stat = "ecdf") +
  labs(title = "CDF of Predicted vs Actual Prices RF Model",
       x = "Price",
       y = "Cumulative Probability") +
  theme_economist_white() +
  scale_color_manual(values = c("Predicted" = "#456990", "Actual" = "#F45B69")) +
  annotate("text", x = max(cdf_data$Predicted), y = 0.2, label = paste("KS Statistic =", round(ks_statistic_rf, 4)), color = "black")

# Save the plot in the working directory
ggsave("cdf_plot_4_5.png", plot = cdf_plot_4_5, width = 8, height = 6, units = "in")

# Code tested as of Oct 7, 2023 12:07 am

###############################################################################
# 4.6. Machine Learning model: LightGBM                                       #
###############################################################################

# Read the "subset_data.csv" file from the current working directory
subset_data <- read.csv("subset_data.csv")
subset_data <- subset_data %>% select(Price_Us, year, mileage, Engine_displacement, model)

# Check the structure of the loaded data frame
str(subset_data)
colSums(is.na(subset_data))

# Split the data into training (70%) and testing (30%) sets
set.seed(1974)
split_index <- sample(1:nrow(subset_data), 0.7 * nrow(subset_data))
train_data <- subset_data[split_index, ]
test_data <- subset_data[-split_index, ]

# Convert data to matrices
train_matrix <- as.matrix(train_data[, -1])
test_matrix <- as.matrix(test_data[, -1])
str(train_matrix)
str(test_matrix)

# Check for missing values
colSums(is.na(train_matrix))
colSums(is.na(test_matrix))

# Create LightGBM datasets
train_data_lgbm <- lgb.Dataset(data = train_matrix, label = train_data$Price_Us)
test_data_lgbm <- lgb.Dataset(data = test_matrix, label = test_data$Price_Us, reference = train_data_lgbm)

# Define LightGBM parameters
params <- list(
  objective = "regression", # Use "regression" as this is a regression model
  metric = "rmse",          # Evaluation metric
  learning_rate = 0.1,      # Learning rate
  max_depth = 6,            # Maximum depth of trees
  n_estimators = 100        # Number of boosting rounds
)

# Train the LightGBM model
model_lgbm <- lgb.train(
  params = params,
  data = train_data_lgbm,
  nrounds = 100,  # Number of boosting rounds
  valids = list(validation = test_data_lgbm),
  early_stopping_rounds = 10
)

# Save the LightGBM model to a file
saveRDS.lgb.Booster(model_lgbm, file = "lgbm_model.rds")

# Load the saved LGBM model
model_lgbm <- readRDS.lgb.Booster("lgbm_model.rds")

# Make predictions on the test data
predictions_lgbm <- predict(model_lgbm, test_matrix)

# Evaluate the model
rmse_lgbm <- sqrt(mean((predictions_lgbm - test_data$Price_Us)^2))
rmse_lgbm
#[1] 6299.642

# Calculate residuals
residuals <- predictions_lgbm - test_data$Price_Us

# Create a histogram of residuals
hist(residuals, main = "Distribution of Residuals", 
     xlab = "Residuals", ylab = "Frequency", col = "lightblue")

# Create a density plot
plot_4_6 <- ggplot(data = data.frame(residuals = residuals), aes(x = residuals)) +
  geom_density(fill = "skyblue") +
  labs(title = "Distribution of Residuals LightGBM Model", x = "Residuals") +
  theme(plot.title = element_text(size = 12))  

# Save the plot in the working directory
ggsave("plot_4_6.png", plot = plot_4_6, width = 8, height = 6, units = "in")

# Create a data frame with actual and predicted Price_Us values
scatter_data_4_6 <- data.frame(Actual = test_data$Price_Us, 
                               Predicted = predictions_lgbm)

# Create a scatter plot
scatter_4_6 <- ggplot(data = scatter_data_4_6, aes(x = Actual, y = Predicted)) +
  geom_point(color = "black", alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Scatter Plot of Actual vs. Predicted Price_Us LGBM Model",
       x = "Actual Price_Us",
       y = "Predicted Price_Us") +
  theme_economist_white() +
  theme(plot.title = element_text(size = 12))  

# Save the plot in the working directory
ggsave("scatter_4_6.png", plot = scatter_4_6, width = 8, height = 6, units = "in")

# Calculate the KS statistic
ks_statistic_lgbm <- ks.test(predictions_lgbm, test_data$Price_Us)$statistic
print(ks_statistic_lgbm)
#0.04711835 

# Create data frames for predicted and actual values
cdf_data_4_6 <- data.frame(
  Predicted = predictions_lgbm,
  Actual = test_data$Price_Us
)

# Create the CDF plot
cdf_plot_4_6 <- ggplot(cdf_data_4_6, aes(x = Predicted)) +
  stat_ecdf(aes(color = "Predicted"), geom = "step") +
  geom_step(data = cdf_data, aes(x = Actual, color = "Actual"), stat = "ecdf") +
  labs(title = "CDF of Predicted vs Actual Prices LGBM Model",
       x = "Price",
       y = "Cumulative Probability") +
  theme_economist_white() +
  scale_color_manual(values = c("Predicted" = "#456990", "Actual" = "#F45B69")) +
  annotate("text", x = max(cdf_data$Predicted), y = 0.2, label = paste("KS Statistic =", round(ks_statistic_lgbm, 4)), color = "black")

# Save the plot in the working directory
ggsave("cdf_plot_4_6.png", plot = cdf_plot_4_6, width = 8, height = 6, units = "in")

# Code tested as of Oct 8, 2023 10:38 pm AM

###############################################################################
# 4.7. Machine Learning model: Xgboost                                        #
###############################################################################

# Read the "train_data.csv" file from the current working directory
train_data <- read.csv("train_data.csv")

# Read the "test_data.csv" file from the current working directory
test_data <- read.csv("test_data.csv")

# Reorder columns in train_data to put the target variable first
train_data <- train_data[, c("Price_Us", setdiff(names(train_data), "Price_Us"))]

# Reorder columns in test_data to put the target variable first
test_data <- test_data[, c("Price_Us", setdiff(names(test_data), "Price_Us"))]

# Remove categorical variables to numeric
train_data <- train_data[, !(names(train_data) %in% c("make", "city", "model"))]
test_data <- test_data[, !(names(test_data) %in% c("make", "city", "model"))]

train_matrix <- as.matrix(train_data[, -1])
test_matrix <- as.matrix(test_data[, -1])

str(train_matrix)
str(test_matrix)

train_data$Price_Us <- as.numeric(train_data$Price_Us)
test_data$Price_Us <- as.numeric(test_data$Price_Us)

# Define the XGBoost model

xgb_model <- xgboost(
  data = train_matrix,         # Training data in matrix
  label = train_data$Price_Us, # Target variable from data frame
  objective = "reg:squarederror", # Mnimizes squared error loss for regression tasks
  nrounds = 500,               # Boosting rounds
  max_depth = 6,              # Limit the depth of each decision tree to 6, prevent overfitting
  eta = 0.1                   # Learning rate, controlling the step size for parameter updates
)

#[500]	train-rmse:3959.115034

# Create a data frame with RMSE improvement along boosting round
rmse_data <- data.frame(
  Boosting_Round = 1:270,
  RMSE = c(14068.012235, 12928.732139, 11919.913284, 11028.011791, 10246.543368,
      9559.478079, 8957.851194, 8433.495375, 7984.067066, 7594.818552,
      7263.899040, 6977.278149, 6737.088623, 6530.303306, 6355.594346,
      6205.309951, 6081.288070, 5975.476792, 5880.920892, 5806.326962,
      5742.784861, 5685.149918, 5638.422894, 5598.862608, 5566.167430,
      5529.004697, 5503.337893, 5481.194603, 5454.649828, 5431.677558,
      5410.765952, 5394.978896, 5374.169575, 5357.656307, 5339.737762,
      5325.590766, 5302.286858, 5293.503102, 5280.787236, 5266.218721,
      5255.749379, 5242.431835, 5235.782992, 5226.200249, 5218.700511,
      5213.046785, 5207.051162, 5202.145687, 5195.391832, 5190.154217,
      5186.230841, 5180.824745, 5176.880865, 5172.212662, 5166.127822,
      5162.177182, 5154.454192, 5146.841923, 5141.440634, 5133.998522,
      5124.942009, 5123.236458, 5121.307683, 5117.184073, 5114.182983,
      5108.437775, 5105.272576, 5103.501625, 5102.578367, 5093.708942,
      5092.433889, 5091.338001, 5078.062981, 5076.742220, 5075.454737,
      5072.968401, 5072.086523, 5066.307033, 5064.436684, 5062.329992,
      5056.424209, 5055.547452, 5043.759051, 5042.040801, 5040.559656,
      5039.837742, 5037.523311, 5035.987242, 5035.352345, 5028.953598,
      5019.018833, 5017.875561, 5012.980541, 5011.964919, 5006.598108,
      5005.663818, 5000.244533, 4995.185096, 4988.770895, 4984.535943,
      4984.015691, 4983.510184, 4982.216678, 4981.780462, 4975.008359,
      4969.941231, 4967.797821, 4966.946516, 4963.708553, 4951.755703,
      4939.308690, 4937.846175, 4932.929502, 4925.466194, 4924.973617,
      4923.409004, 4922.975961, 4916.012619, 4911.029276, 4905.794866,
      4903.682457, 4903.052824, 4899.233265, 4896.015590, 4892.057894,
      4891.676993, 4887.152937, 4882.154604, 4870.199831, 4865.646250,
      4865.098982, 4862.405784, 4857.232231, 4852.850552, 4852.482043,
      4850.031738, 4849.681522, 4845.878803, 4843.083474, 4842.640987,
      4839.934707, 4839.492692, 4838.091082, 4837.835683, 4832.410354,
      4827.934224, 4827.611894, 4824.702911, 4820.616339, 4809.459503,
      4805.918349, 4805.720083, 4803.258693, 4795.758688, 4795.180162,
      4793.260321, 4788.535801, 4785.318601, 4777.524139, 4777.214228,
      4776.051597, 4773.672377, 4771.316824, 4770.373312, 4764.103015,
      4762.175408, 4760.930926, 4756.971556, 4756.183716, 4751.609924,
      4749.708618, 4749.104628, 4748.916358, 4744.208634, 4742.247499,
      4742.080089, 4736.931959, 4732.692826, 4725.992808, 4725.057131,
      4722.437038, 4716.464334, 4716.366709, 4708.907217, 4707.115979,
      4705.488136, 4705.000178, 4704.631751, 4702.587965, 4700.529046,
      4699.711914, 4699.548458, 4697.159558, 4696.263539, 4694.280616,
      4694.146320, 4693.871166, 4691.051581, 4690.135609, 4689.725105,
      4689.121307, 4686.154062, 4685.308344, 4684.730274, 4684.123935,
      4683.634303, 4683.010920, 4681.107126, 4680.900372, 4680.162330,
      4677.347917, 4676.507411, 4676.217360, 4674.392250, 4673.698936,
      4673.260573, 4671.329160, 4670.936366, 4670.225722, 4669.420919,
      4668.769283, 4668.586683, 4668.434774, 4668.153001, 4668.011172,
      4667.501388, 4666.427273, 4665.307445, 4665.121204, 4664.989812,
      4664.429743, 4663.313235, 4663.134826, 4662.737789, 4661.900688,
      4661.729032, 4660.989763, 4660.841110, 4660.706271, 4660.466031,
      4659.942366, 4659.792188, 4659.276675, 4659.118712, 4658.621504,
      4658.367781, 4657.948537, 4657.609419, 4657.450142, 4657.082416,
      4656.717612, 4656.565469, 4656.464280, 4656.347763, 4656.178780,
      4656.097262, 4656.065752, 4656.044771, 4655.974870, 4655.933390,
      4655.904769, 4655.886594, 4655.856489, 4655.839998, 4655.812230,
      4655.800406, 4655.782230, 4655.768252, 4655.754257, 4655.749406
    )
)

# Create the RMSE improvement plot
improvement_4_7 <- ggplot(rmse_data, aes(x = Boosting_Round, y = RMSE)) +
  geom_line() +
  labs(title = "RMSE Improvement Along XGB Boosting Rounds", x = "Boosting Round", y = "RMSE")

# Save the plot in the working directory
ggsave("improvement_4_7.png", plot = improvement_4_7, width = 8, height = 6, units = "in")

# Save the XGBoost model to a file
saveRDS(xgb_model, file = "xgboost_model.rds")

# Load the saved XGBoost model
xgb_model <- readRDS("xgboost_model.rds")

# Make predictions on the test data
xgb_predictions <- predict(xgb_model, test_matrix)

# Evaluate the XGBoost model
rmse_xgb <- sqrt(mean((xgb_predictions - test_data$Price_Us)^2))
rmse_xgb
#[1] 5462.64

# Calculate residuals
xgb_residuals <- xgb_predictions - test_data$Price_Us

# Create a histogram of residuals
hist(xgb_residuals, main = "Distribution of Residuals (XGBoost)",
     xlab = "Residuals", ylab = "Frequency", col = "lightblue")

# Create a density plot
plot_4_7 <- ggplot(data = data.frame(residuals = residuals), aes(x = residuals)) +
  geom_density(fill = "skyblue") +
  labs(title = "Distribution of Residuals XGBoost Model", x = "Residuals") +
  theme(plot.title = element_text(size = 12))  

# Save the plot in the working directory
ggsave("plot_4_7.png", plot = plot_4_7, width = 8, height = 6, units = "in")

# Divide the Price_Us values by 1000 to convert them to thousands of dollars
actual_price_in_thousands <- test_data$Price_Us / 1000
predicted_price_in_thousands <- xgb_predictions / 1000

# Create a vector to specify colors based on the comparison between actual and predicted prices
colors <- ifelse(actual_price_in_thousands < predicted_price_in_thousands, "skyblue", "lightgreen")

# Create a data frame with actual and predicted prices in thousands of dollars
scatter_data_4_7 <- data.frame(Actual = actual_price_in_thousands, 
                   Predicted = predicted_price_in_thousands, 
                   Color = colors)

# Create the scatter plot
scatter_4_7 <- ggplot(scatter_data_4_7, aes(x = Actual, y = Predicted, color = Color)) +
  geom_point(shape = 19) +
  labs(
    title = "Actual vs. Predicted Price_Us (XGBoost)",
    x = "Actual Price_Us (in thousands of dollars)",
    y = "Predicted Price_Us (in thousands of dollars)"
  ) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  scale_color_manual(values = c("skyblue" = "skyblue", "lightgreen" = "lightgreen")) +
  theme_economist_white()

# Save the plot in the working directory
ggsave("scatter_4_7.png", plot = scatter_4_7, width = 8, height = 6, units = "in")

# Calculate the KS statistic
ks_statistic_xgb <- ks.test(xgb_predictions, test_data$Price_Us)$statistic
print(ks_statistic_xgb)
#0.04959716

# Create data frames for predicted and actual values
cdf_data_4_7 <- data.frame(
  Predicted = xgb_predictions,
  Actual = test_data$Price_Us
)

# Create the CDF plot
cdf_plot_4_7 <- ggplot(cdf_data_4_7, aes(x = Predicted)) +
  stat_ecdf(aes(color = "Predicted"), geom = "step") +
  geom_step(data = cdf_data, aes(x = Actual, color = "Actual"), stat = "ecdf") +
  labs(title = "CDF of Predicted vs Actual Prices XGB Model",
       x = "Price",
       y = "Cumulative Probability") +
  theme_economist_white() +
  scale_color_manual(values = c("Predicted" = "#456990", "Actual" = "#F45B69")) +
  annotate("text", x = max(cdf_data$Predicted), y = 0.2, label = paste("KS Statistic =", round(ks_statistic_xgb, 4)), color = "black")

# Save the plot in the working directory
ggsave("cdf_plot_4_7.png", plot = cdf_plot_4_7, width = 8, height = 6, units = "in")

# Code tested as of Oct 9, 2023 10:09 am AM

###############################################################################
# 5. MODEL VALIDATION                                                         #
###############################################################################

# 5. 1. Validate the best performing model: Xgboost model

# Read 'hold_out_data'
hold_out_data <- read.csv("hold_out_data.csv")

# View the structure of the data frame
str(hold_out_data)

# Reorder columns in train_data to put the target variable first
hold_out_data <- hold_out_data[, c("Price_Us", setdiff(names(hold_out_data), "Price_Us"))]

# Remove or convert categorical variables to numeric
hold_out_data <- hold_out_data[, !(names(hold_out_data) %in% c("make", "city", "model"))]

hold_out_matrix <- as.matrix(hold_out_data[, -1])

str(train_matrix)
str(hold_out_matrix)

hold_out_data$Price_Us <- as.numeric(hold_out_data$Price_Us)

# Make predictions on the test data
xgb_predictions_mv <- predict(xgb_model, hold_out_matrix)

# Calculate RMSE
rmse_xgb_v <- sqrt(mean((xgb_predictions_mv - hold_out_data$Price_Us)^2))
rmse_xgb_v
#[1] 10403.11

# Calculate residuals
residuals <- hold_out_data$Price_Us - xgb_predictions_mv

# Categorize residuals as over-predictions (positive residuals) and under-predictions (negative residuals)
over_predictions <- residuals > 0
under_predictions <- residuals < 0

# Create a scatter plot with different colors for over and under predictions
scatter_5_1 <- ggplot(data = hold_out_data, aes(x = Price_Us, y = xgb_predictions_mv, color = factor(over_predictions))) +
  geom_point() +
  scale_color_manual(
    values = c("skyblue", "lightgreen"), labels = c("Under Predictions", "Over Predictions")) +
  geom_abline(
    intercept = 0, slope = 1, linetype = "dashed", color = "red") +  # Add the 45-degree reference line
  labs(title = "Actual vs. Predicted Price_Us Xgboost - Model Validation",
       x = "Actual Price_Us",
       y = "Predicted Price_Us") +
  theme_economist_white() +
  theme(plot.title = element_text(size = 12))  

# Save the plot in the working directory
ggsave("scatter_5_1.png", plot = scatter_5_1, width = 8, height = 6, units = "in")

# Calculate the KS statistic
ks_statistic_xgb_mv <- ks.test(xgb_predictions_mv, test_data$Price_Us)$statistic
print(ks_statistic_xgb_mv)
#0.5772509

# Create data frames for predicted and actual values
cdf_data_5_1 <- data.frame(
  Predicted = xgb_predictions_mv,
  Actual = hold_out_data$Price_Us
)

# Create the CDF plot
cdf_plot_5_1 <- ggplot(cdf_data_5_1, aes(x = Predicted)) +
  stat_ecdf(aes(color = "Predicted"), geom = "step") +
  geom_step(data = cdf_data, aes(x = Actual, color = "Actual"), stat = "ecdf") +
  labs(title = "CDF of Predicted vs Actual Prices XGB Model Validation (hold-out)",
       x = "Price",
       y = "Cumulative Probability") +
  theme_economist_white() +
  scale_color_manual(values = c("Predicted" = "#456990", "Actual" = "#F45B69")) +
  annotate("text", x = max(cdf_data$Predicted), y = 0.2, label = paste("KS Statistic =", round(ks_statistic_xgb_mv, 4)), color = "black")

# Save the plot in the working directory
ggsave("cdf_plot_5_1.png", plot = cdf_plot_5_1, width = 8, height = 6, units = "in")


#* Log of Actions
#* Code tested as of Oct 15, 2023 2:12 pm AM
#* Plots saved as of Oct 15, 2023 2:13 pm AM
#* Models saved as of Oct 15, 2023 2:27 pm AM
#* References completed RMD as of Oct 18, 2023 7:53 pm AM

#* Code upload to GitHub
#* Three files submit on EDX course