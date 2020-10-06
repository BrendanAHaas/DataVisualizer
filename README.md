# DataVisualizer
Read in a data file and output a PDF showing column correlations and behaviors.

This script is designed to read in a dataframe and automatically analyze the data and generate a PDF that summarizes the data present in the dataframe.  No input is required from the user other than the file location and name, though other parameters can be specified to improve results.

dataVisualization.py is the file to run after specifying the variables data_folder, data_filename, and data_extension in lines 28-30 of the file.  derivitizationUtils.py is a collection of utility functions necessary for the creation of the output PDF.


# PDF Output Summary:
Page 1-2:  Introduction Pages.  Summarizes the goals of the output PDF.

Page 3:  Columns Data Profile Summary.  Shows size of the input dataframe and number of analyzed columns.

Page 4:  Dataframe Columns Summary.  Shows basic information about all analyzed columns, such as the data type, the number of unique values, minimum/maximum values, and null values.

Page 5:  Columns with Null Values greater than 25%.  Highlights columns with large amounts of missing data.

Page 6:  Data Correlation Plot for Numeric Columns.  A data correlation plot for numeric columns indicating how well correlated the columns are (values range from 1 to -1 for correlation and anticorrelation, respectively).

Page 7:  Data Correlation Plot for Categorical Columns.  A data correlation plot for categorical columns indicating how well correlated the columns are (values range from 1 to 0 for highly correlated to no correlation, respectively).

Page 8:  Categorical Column and Measure Column Associations.  A data correlation plot for categorical columns versus numeric columns (values range from 1 to 0 for highly correlated to no correlation, respectively).

Page 9+:  Column Data Profile Details.  Each column is summarized in greater detail.  Each column has its data type, values, minima/maxima, nulls, min/mean/max values, and 25th/50th/75th percentile values listed.  Numeric columns have 6 summarized with 6 plots:  1.  Box plots for all values, 2.  Distribution plot, 3. Box plot without outliers, 4.  Violin plot, 5.  Quartile box plot, and 6.  Distribution plot (for values <95th percentile).  Datetime and Categorical columns are summarized with a single plot listing the most common values that appear within the column as well as the number of appearances for each of these values.

Page 10+:  Plots of columns with high correlations/anticorrelations as summarized in pages 6-8.

Page 11+:  Summary with list of highly correlated/anticorrelated columns.
