import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from statsmodels.graphics.mosaicplot import mosaic
from pylab import rcParams
import seaborn as sns
import datetime
import csv
import itertools
import math
import scipy.stats as ss
from collections import Counter


def load_dataframe(file_path, file_name, file_type, encode_type='latin1', separator = None): #default encoding is 'latin1'
    
    #read the file into a dataframe
    if separator == None:
        try: #first try pipe-separated variables
            df=pd.read_csv(file_path + file_name + file_type, sep="|", encoding=encode_type, low_memory=False)
            print('Loaded dataframe as pipe-separated variables.')
        except:
            try: #second try tab-separated variables
                df=pd.read_csv(file_path + file_name + file_type, sep="\t", encoding=encode_type, low_memory=False)
                print('Loaded dataframe as tab-separated variables.')
            except:
                try: #third try comma-separated variables
                    df=pd.read_csv(file_path + file_name + file_type, sep=",", encoding=encode_type, low_memory=False)
                    print('Loaded dataframe as comma-separated variables.')
                except:
                    print('Unable to load file as pipe, tab, or comma-separated.')
    
    else:
        try: #first try pipe-separated variables
            df=pd.read_csv(file_path + file_name + file_type, sep=separator, encoding=encode_type, error_bad_lines = False, low_memory=False)
            print('Loaded dataframe with specified separator.')
        except:
            print('Unable to load file with the specified separator.')
    return(df)


def save_file(df, file_path, file_name, save_dtypes):

    #Save Clean Dataset
    print("Saving Clean Data...")
    df.to_csv(file_path + file_name + "_CLEAN.csv", sep="|", index = False) #Use pipe seps
    print("Clean File Saved Successfully")
    
    #If you want to save the datatypes (so you can load the optimzed file later)
    if save_dtypes:
        
        #Gather a list of datetime columns
        date_field_list = []
        for column in df.columns:
            if df[column].dtype == 'datetime64[ns]':
                date_field_list.append(column)
        #save the date list
        df_date_fields = pd.DataFrame(date_field_list)
        df_date_fields.columns = ['Column_Name']
        df_date_fields.to_csv(file_path + file_name + "_CLEAN_DATES.csv", sep="|", index=False) #Use pipe seps
        print("Saved Date Fields")
            
        #Read column tipes into a dictionary
        dtypes = df.dtypes
        column_names = dtypes.index
        types = [i.name for i in dtypes.values]

        #Convert datetime columns to objects because pandas cannot import a datetime column
        types_converted = [x if x != 'datetime64[ns]' else 'object' for x in types]
        column_types_converted = dict(zip(column_names, types_converted))
        print("Save DTypes")

        #Write out the dicionary file
        pd.DataFrame.from_dict(data=column_types_converted, orient='index').to_csv(file_path + file_name + "_CLEAN_DICT.csv",  sep="|", index=False) #BH: Added sep                                                                                   sep="|", header=False) #BH:  Added sep
        print("Saved Clean Data File")
        

def load_dataframe_optimized(file_path, file_name):
    
    start_time = datetime.datetime.now()
    
    #Import the date fields list
    df_date_fields=pd.read_csv(file_path + file_name + "_CLEAN_DATES.csv")
    date_fields_list_import = list(df_date_fields['Column_Name'])
    
    #import DType dictionary
    with open(file_path + file_name + "_CLEAN_DICT.csv", mode='r') as infile:
        reader = csv.reader(infile)
        column_types_dict = {rows[0]:rows[1] for rows in reader}
    
    #Import the data file
    df_optimized=pd.read_csv(file_path + file_name + "_CLEAN.csv",dtype=column_types_dict, parse_dates=date_fields_list_import)
    
    end_time = datetime.datetime.now()
    print("Load Time: ", end_time - start_time)
    
    print(df_optimized.info(memory_usage='deep'))
    
    return(df_optimized)


def mem_usage(pandas_obj):
    
    if isinstance(pandas_obj,pd.DataFrame):
        
        usage_b = pandas_obj.memory_usage(deep=True).sum()
        
    else: # we assume if not a df it's a series
        
        usage_b = pandas_obj.memory_usage(deep=True)
        usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
        
    return "{:03.2f} MB".format(usage_mb)


def find_unique_combos(df, list_1, list_2, unique_cutoff):
    
    charts_skipped = 0
    unique_combos = []
    unique_combos_cutoff = []
    
    #create a dictionary of the unique value counts
    unique_counts = df.nunique()

    #List all combinations (there will be duplicates there)
    combos = list(itertools.product(list_1, list_2))

    for i in range(len(combos)):

        if (combos[i][1],combos[i][0]) not in unique_combos and combos[i][0] != combos[i][1]:
            
            unique_combos.append(combos[i])
            
            #Check to see if there are too many unique values
            if (unique_counts[combos[i][0]] + unique_counts[combos[i][1]]) <= unique_cutoff:

                unique_combos_cutoff.append(combos[i])
                
            else:
                
                charts_skipped = charts_skipped + 1

    #print("Unique Combos: ", len(unique_combos))
    #print("Charts to Create: ", len(unique_combos_cutoff))
    #print("Charts to Skip: ", charts_skipped)

    return(unique_combos_cutoff, charts_skipped)


def clean_date_data(df, fields):
    
    #Loop through the fields
    for field in fields:
        
        #Make sure the field is in the dataframe
        if field in df.columns:
        
            #If it is already a date, then skip it
            if field not in df.select_dtypes(include=['datetime','datetime64','datetime64[ns]','<M8[ns]']).columns:
                
                print(field)

                #Track memory usage #NOTE:  Simply tracking memory usage causes python to reduce memory usage
                MemUsage = df.memory_usage().sum()
                
                #Convert to string
                df[field] = df[field].astype(str)

                #Remove_Spaces
                df[field] = df[field].astype(str).str.strip()

                #Replace invalid entries with nulls
                df[field] = df[field].replace("N/A",pd.NaT)
                df[field] = df[field].replace("NaT",pd.NaT)
                df[field] = df[field].replace("nan",pd.NaT)
                df[field] = df[field].replace("",pd.NaT)
                
                #Replace Month Listings with numbers (3 letter month abbreviations sometimes appear in military data)
                try:
                    df[field] = df[field].astype(str).str.replace("Jan", "1", case=False)
                    df[field] = df[field].astype(str).str.replace("Feb", "2", case=False)
                    df[field] = df[field].astype(str).str.replace("Mar", "3", case=False)
                    df[field] = df[field].astype(str).str.replace("Apr", "4", case=False)
                    df[field] = df[field].astype(str).str.replace("May", "5", case=False)
                    df[field] = df[field].astype(str).str.replace("Jun", "6", case=False)
                    df[field] = df[field].astype(str).str.replace("Jul", "7", case=False)
                    df[field] = df[field].astype(str).str.replace("Aug", "8", case=False)
                    df[field] = df[field].astype(str).str.replace("Sep", "9", case=False)
                    df[field] = df[field].astype(str).str.replace("Oct", "10", case=False)
                    df[field] = df[field].astype(str).str.replace("Nov", "11", case=False)
                    df[field] = df[field].astype(str).str.replace("Dec", "12", case=False)
                except:
                    pass
                
                #Blank out fields that contain characters
                try:
                    regexp = "[a-zA-Z]+"
                    df[field] = np.where(df[field].astype(str).str.match(regexp),pd.NaT,df[field])
                except:
                    pass

                #Attempt conversions with different date formats; using the right format is much faster for to_datetime
                #Test first 1000 entries
                testdf = df[field].dropna().head(1000)
                dcthresh = 0.9 #Threshold for success in datetime conversion formattings
                #If first 1000 nonnull entries agree with one formatting above dcthresh, use it for the column
                if len(pd.to_datetime(testdf, format="%m/%d/%Y", errors='coerce').dropna())/len(testdf) >= dcthresh:
                    df[field] = pd.to_datetime(df[field], format="%m/%d/%Y", errors='coerce')
                elif len(pd.to_datetime(testdf, format="%m/%d/%Y %H:%M:%S.%f", errors='coerce').dropna())/len(testdf) >= dcthresh:
                    df[field] = pd.to_datetime(df[field], format="%m/%d/%Y %H:%M:%S.%f", errors='coerce')
                elif len(pd.to_datetime(testdf, format="%m-%d-%Y", errors='coerce').dropna())/len(testdf) >= dcthresh:
                    df[field] = pd.to_datetime(df[field], format="%m-%d-%Y", errors='coerce')
                elif len(pd.to_datetime(testdf, format="%m-%d-%Y %H:%M:%S.%f", errors='coerce').dropna())/len(testdf) >= dcthresh:
                    df[field] = pd.to_datetime(df[field], format="%m-%d-%Y %H:%M:%S.%f", errors='coerce')
                elif len(pd.to_datetime(testdf, format="%m %d, %Y", errors='coerce').dropna())/len(testdf) >= dcthresh:
                    df[field] = pd.to_datetime(df[field], format="%m-%d-%Y", errors='coerce')
                #Try inferring datetime format, if none of this works use the guaranteed but slow method 
                elif len(pd.to_datetime(testdf, infer_datetime_format = True, errors='coerce').dropna())/len(testdf) >= dcthresh:
                    df[field] = pd.to_datetime(df[field], infer_datetime_format = True, errors='coerce')
                else: #Slow but will almost always work
                    df[field] = pd.to_datetime(df[field], errors='coerce')
                df[field] = pd.to_datetime(df[field], errors='coerce') 
                
                #Remove dates that are in the future (these probably don't make sense)
                df[field] = df[field].apply(lambda x: x if x <= datetime.datetime.now() else pd.NaT)
                
                #Update memory usage
                MemUsage = df.memory_usage().sum()
        
    return(df)


def clean_categorical_data(df, fields):
    
    #Loop through the fields
    for field in fields:
        
        print(field)
        
        #Check to make sure the field is in the dataframe
        if field in df.columns:

            #Track memory usage  #NOTE:  Simply tracking memory usage causes python to reduce memory usage
            MemUsage = df.memory_usage().sum()
            
            #Remove_Spaces
            df[field] = df[field].astype(str).str.strip()

            #Set to null if it contains only a questionmark
            regexp = "^\\?$"
            df[field] = np.where(df[field].str.match(regexp),np.nan,df[field])

            regexp = "[0-9]{1,4}[/-][0-9]{1,2}[/-][0-9]{1,4}"
            df[field] = np.where(df[field].str.match(regexp),np.nan,df[field])

            #Replace invalid entries with nulls
            df[field] = df[field].replace("N/A",np.nan)
            df[field] = df[field].replace("nan",np.nan)
            df[field] = df[field].replace("NaN",np.nan)
            df[field] = df[field].replace("",np.nan)
            
            #Convert to category data types if the dtype is object and there are less 50% unique values
            if df[field].nunique() < (len(df)*0.5):

                #Convert the column to a category (this saves lots of memory)
                df[field] = df[field].astype('category')
                
            #Update memory usage
            MemUsage = df.memory_usage().sum() 

    return(df)


def clean_measure_data(df, fields):
    
    for field in fields:
        
        print(field)
        
        #Check to make sure the field is in the dataframe
        if field in df.columns:

            #Track memory usage  #NOTE:  Simply tracking memory usage causes python to reduce memory usage
            MemUsage = df.memory_usage().sum()
            
            #Remove_Spaces
            df[field] = df[field].astype(str).str.strip()
            
            #Remove $ signs and commas (common in dataframe columns)
            df[field] = df[field].str.replace(',','')
            df[field] = df[field].str.replace('$', '')

            regexp = "^\\?$"
            df[field] = np.where(df[field].str.match(regexp),np.nan,df[field])

            regexp = "[0-9]{1,4}[/-][0-9]{1,2}[/-][0-9]{1,4}"
            df[field] = np.where(df[field].str.match(regexp),np.nan,df[field])

            regexp = "^[ ]*[-]?[ ]*[0-9]*[.]?[0-9]+[ ]*$"
            df[field] = np.where(df[field].str.match(regexp),df[field], np.nan)

            #Replace invalid entries with nulls
            df[field] = df[field].replace("N/A",np.nan)
            df[field] = df[field].replace("nan",np.nan)
            df[field] = df[field].replace("NaN",np.nan)
            df[field] = df[field].replace("",np.nan)

            #Convert it back into a number
            df[field] = pd.to_numeric(df[field])
            
            #Update memory usage
            MemUsage = df.memory_usage().sum()
        
    return(df)


def check_directory(directory):
    
    #Check to see if directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
        
        
def basic_timeseries_analysis(df, filename, fields):
    
    start_time = datetime.datetime.now()
    print("Basic Timeseries Analysis")
    analysis_stats = []

    for field in fields:
        
        #print(field)

        #Check to make sure the field is in the dataframe
        if field in df.columns:
        
            analysis_stats.append({'Variable Name': field,
                                   'Number of Unique Values': df[field].nunique(),
                                   'Minimum Value': np.min(df[field]),
                                   'Maximum Value': np.max(df[field]),
                                   'Populated Fields': df[field].count(),
                                   'Missing Fields': df[field].isna().sum(),
                                   'Percentage Missing': round(df[field].isna().sum() / len(df[field]),2)})

    #Output the file
    pd.DataFrame(analysis_stats).to_csv(filename + ".csv", index=False)
    
    print("Run Time: ", datetime.datetime.now() - start_time)
    
    
def basic_category_analysis(df, filename, fields):
    
    start_time = datetime.datetime.now()
    print("Basic Category Analysis - " + filename)
    analysis_stats = []

    for field in fields:
        
        #Check to make sure the field is in the dataframe
        if field in df.columns:    
        
            analysis_stats.append({'Variable Name': field,
                                   'Number of Unique Values': df[field].nunique(),
                                   'Minimum Value Length': np.min(df[field].astype(str).map(len)),
                                   'Maximum Value Length': np.max(df[field].astype(str).map(len)),
                                   'Populated Fields': df[field].count(),
                                   'Missing Fields': df[field].isna().sum(),
                                   'Percentage Missing': round(df[field].isna().sum() / len(df[field]),2)})

    #Output the file
    pd.DataFrame(analysis_stats).to_csv(filename + ".csv", index=False, )
    
    print("Run Time: ", datetime.datetime.now() - start_time)
    
    
def basic_measure_analysis(df, filename, fields):
    
    start_time = datetime.datetime.now()
    print("Basic Measure Analysis - " + filename)
    analysis_stats = []

    for field in fields:
        
        #Check to make sure the field is in the dataframe
        if field in df.columns:
        
            analysis_stats.append({'Variable Name': field,
                                   'Number of Unique Values': df[field].nunique(),
                                   'Minimum Value': np.min(df[field]),
                                   'Maximum Value': np.max(df[field]),
                                   'Mean': np.mean(df[field]),
                                   'Median': np.nanmedian(df[field]),
                                   'Populated Fields': df[field].count(),
                                   'Missing Fields': df[field].isna().sum(),
                                   'Percentage Missing': round(df[field].isna().sum() / len(df[field]),2)})

    #Output the file
    pd.DataFrame(analysis_stats).to_csv(filename + ".csv", index=False, )
    
    print("Run Time: ", datetime.datetime.now() - start_time)
    
    
def histogram_analysis(df, directory, fields, isdate = False):
    
    start_time = datetime.datetime.now()
    if isdate == False:
        print("Histogram Analysis - Categorical - " + directory)
    if isdate == True:
        print("Histogram Analysis - Datetime - " + directory)
    
    #Check to see if directory exists
    check_directory(directory)
    
    for field in fields:
        
        #Check to make sure the field is in the dataframe
        if field in df.columns:
        
            if df[field].count() > 0:
                
                if isdate == True:  #Histogram creation for date fields
                    
                    #Make a plot of frequency of month+year combinations
                    df_datemonthyears = df[field].groupby([df[field].dt.year.rename('Year'), df[field].dt.month.rename('Month')]).count().reset_index(name='Counts')
                    df_datemonthyears['Year'] = df_datemonthyears['Year'].astype(int) #Convert to ints to remove .0 at end of values
                    df_datemonthyears['Month'] = df_datemonthyears['Month'].astype(int)
                    df_datemonthyears['Year'] = df_datemonthyears['Year'].astype(str) #Covert to strings to allow easy concatenation
                    df_datemonthyears['Month'] = df_datemonthyears['Month'].astype(str)
                    df_datemonthyears['YearMonth'] = df_datemonthyears[['Year','Month']].apply(lambda x: '-'.join(x), axis=1)
                    plot = df_datemonthyears.plot(kind='bar', x='YearMonth', y='Counts', legend=False)
                    plot.set_title("Histogram of " + str(field), {'fontsize':25, 'fontweight':1, 'verticalalignment':'baseline', 'horizontalalignment':'center'})
                    plot.set_xlabel('')
                    plot.tick_params(axis='y',labelsize=25)
                    if df_datemonthyears.shape[0] > 25: #If over 25 month/year combinations to plot
                        for label in plot.xaxis.get_ticklabels()[::2]: #Only show every other tick label (keeps plot clean)
                            label.set_visible(False)
                    fig = plot.get_figure()
                    fig.set_size_inches(16, 10)
                    fig.savefig(directory + "/" + field + ".png", bbox_inches = "tight")
                    plt.clf()
                    plt.close()
                    
                    #Code for plotting by year rather than month and year
                    #df_dateyears = df[field].groupby(df[field].dt.year).count().reset_index(name='counts')
                    #df_dateyears[field] = df_dateyears[field].astype(int) #Convert years from floats to ints
                    #plot = df_dateyears.plot(kind='bar',x=field,y='counts', legend=False)
                    #plot.set_title("Histogram of " + str(field), {'fontsize':25, 'fontweight':1, 'verticalalignment':'baseline', 'horizontalalignment':'center'})
                    #plot.set_xlabel('')
                    #plot.tick_params(axis='y',labelsize=25)
                    #plot.tick_params(axis='x',labelsize=25)
                    #fig = plot.get_figure()
                    #fig.set_size_inches(16, 10)
                    #fig.savefig(directory + "/" + field + ".png", bbox_inches = "tight")
                    #plt.clf()
                    #plt.close()
                    
                else:  #Histogram creation for categorical fields
                
                    #Build a dataframe containing the top 50 values
                    value_counts = df.groupby(field).size().reset_index(name='counts')
                    value_counts = value_counts.sort_values(by='counts', ascending=False)
                    df_top_50 = value_counts[:50]
                    
                    try:

                        plot = df_top_50.plot(kind='bar',x=field,y='counts', legend=False)
                        plot.set_title("Histogram of " + str(field), {'fontsize':25, 'fontweight':1, 'verticalalignment':'baseline', 'horizontalalignment':'center'})
                        plot.set_xlabel('')
                        plot.tick_params(axis='y',labelsize=20)
                        fig = plot.get_figure()
                        fig.set_size_inches(16, 10)
                        fig.savefig(directory + "/" + field + ".png", bbox_inches = "tight")
                        plt.clf()
                        plt.close()
                    
                    except:
                        print("Error on " + field)
                        pass
                
    print("Run Time: ", datetime.datetime.now() - start_time)
    
    
def weighted_histogram_analysis(df, directory, category_fields, measure_fields):
    
    start_time = datetime.datetime.now()
    print("Weighted Histogram Analysis - " + directory)
    c = 0
    charts_created = 0
    charts_errored = 0
    
    #Check to see if directory exists
    check_directory(directory)
    
    #Find all unique combinations of the fields
    unique_field_pairs, charts_skipped = find_unique_combos(df, category_fields, measure_fields, len(df))
    
    #Loop through the fields
    for category_field,measure_field in unique_field_pairs:
        
        #Increment the counter
        c = c + 1

        #Check to make sure the field is in the dataframe
        if category_field in df.columns and measure_field in df.columns:

            #Sum the measure field for the comparison field
            value_counts = df.groupby(category_field).sum()[measure_field].sort_values(ascending=False)

            if len(value_counts) > 0:

                try:

                    #Plot the top 50 values in the series
                    plot = value_counts[:50].plot(kind='bar', title=category_field + "_of_" + measure_field)
                    fig = plot.get_figure()
                    fig.set_size_inches(16, 10)
                    fig.savefig(directory + "/" + category_field + " of " + measure_field + ".png", bbox_inches = "tight")
                    plt.clf()
                    plt.close()
                    charts_created = charts_created + 1

                except:
                    #print("Error on " + category_field + " of " + measure_field)
                    charts_errored = charts_errored + 1
                    pass
            
    print("Run Time: " + str(datetime.datetime.now() - start_time) + "  Created: ", str(charts_created) +           "  Skipped: " + str(charts_skipped) + "  Errors: " + str(charts_errored))
    
    
def sample_data(df_plot, df_size, sample_size):
    
    #Take a random sample of the dataframe instead of plotting the whole thing
    #In test run:  Complete data set took 35.5 seconds to plot.  100,000 records took 3.9 seconds.
    #There are over 1800 date combinations to plot.  This results in a savings of roughly 16 hours of processing time
    
    if len(df_plot) > sample_size:
        
        df_sample = df_plot.sample(n=sample_size, random_state=0)
        
    else:
        
        df_sample = df_plot
    
    return(df_sample)


def internal_category_binary_analysis(df, directory, category_fields, unique_cutoff, sample_size):
    
    start_time = datetime.datetime.now()
    print("Internal category binary analysis - " + str(directory))
    c = 0
    charts_created = 0
    charts_errored = 0
    
    #Check to see if directory exists
    check_directory(directory)
    
    #Find all unique combinations of the fields
    unique_field_pairs, charts_skipped = find_unique_combos(df, category_fields, category_fields, unique_cutoff)
    
    #Loop through the fields
    for x_field,y_field in unique_field_pairs:
        
        #Increment the counter
        c = c + 1
        
        #Check to make sure the field is in the dataframe
        if x_field in df.columns and y_field in df.columns:

            #remove the blank and 0 values and convert numbers to strings (otherwise it will error)
            df_plot = df[[x_field,y_field]].copy().dropna()
            df_plot[x_field] = df_plot[x_field].astype(str)
            df_plot[y_field] = df_plot[y_field].astype(str)

            #Make sure there are records left after filtering out bad data
            if len(df_plot) > 0:

                #randomly sample the data (to improve run time)
                df_plot = sample_data(df_plot, len(df), sample_size)
                #print(str(c) + "/" + str(len(unique_field_pairs)) + ": " + x_field + " vs " + y_field + \
                #" - TR:" + str(len(df_plot)) + " UX:" + str(df[x_field].nunique()) + " UY:" + str(df[y_field].nunique()))

                try:
                    #Mosaic Plot
                    #rcParams['figure.figsize'] = 20, 12
                    #mosaic(df_plot, [x_field, y_field], gap=0.01, title=x_field + " vs " + y_field)
                    #plt.savefig(directory + "/" + x_field + "_vs_" + y_field + ".png", bbox_inches='tight')
                    #plt.clf()
                    #plt.close()

                    #Heat Map
                    plt.figure(figsize=(25,20))
                    plot = sns.heatmap(pd.crosstab([df[y_field]], [df[x_field]]), cmap="Blues", annot=False, cbar=True)
                    plot.set_title("Heatmap of " + str(x_field) + " vs " + str(y_field), {'fontsize':30, 'fontweight':1, 'verticalalignment':'baseline', 'horizontalalignment':'center'})
                    plot.tick_params(axis='x',labelsize=20, labelrotation=45)
                    plot.tick_params(axis='y',labelsize=20, labelrotation=45)
                    plot.set_xlabel(str(x_field), fontsize=30)
                    plot.set_ylabel(str(y_field), fontsize=30)
                    plt.savefig(directory + "/" + x_field + "_vs_" + y_field + "_Heatmap.png", bbox_inches='tight')
                    plt.clf()
                    plt.close()
                    charts_created = charts_created + 1

                except:
                    #print("Error on " + x_field + " vs " + y_field)
                    charts_errored = charts_errored + 1
                    pass             

    print("Run Time: " + str(datetime.datetime.now() - start_time) + "  Created: ", str(charts_created) +           "  Skipped: " + str(charts_skipped) + "  Errors: " + str(charts_errored))
    
    
def internal_measure_binary_analysis(df, directory, measure_fields, unique_cutoff, sample_size):
    
    start_time = datetime.datetime.now()
    print("Internal measure binary analysis - " + str(directory))
    c = 0
    charts_created = 0
    charts_errored = 0
    
    #Check to see if directory exists
    check_directory(directory)
    
    #Find all unique combinations of the fields
    unique_field_pairs, charts_skipped = find_unique_combos(df, measure_fields, measure_fields, unique_cutoff)
    
    #Loop through the fields
    for x_field,y_field in unique_field_pairs:
        
        #Increment the counter
        c = c + 1
        
        #Check to make sure the field is in the dataframe
        if x_field in df.columns and y_field in df.columns:
       
            #remove the blank values, 0 values, and convert numbers to strings (otherwise it will error)
            df_plot = df[[x_field,y_field]].copy().dropna()
            df_plot = df_plot[df_plot[y_field] > 0]
            df_plot[x_field] = df_plot[x_field].astype(str)
            df_plot[y_field] = df_plot[y_field].astype(str)

            #Make sure there are records left after filtering out bad data
            if len(df_plot) > 0:

                #randomly sample the data (to improve run time)
                df_plot = sample_data(df_plot, len(df), sample_size)
                #print(str(c) + "/" + str(len(unique_field_pairs)) + ": " + x_field + " vs " + y_field + \
                #" - TR:" + str(len(df_plot)) + " UX:" + str(df[x_field].nunique()) + " UY:" + str(df[y_field].nunique()))

                try:

                    #Mosaic Plot
                    #rcParams['figure.figsize'] = 20, 12
                    #mosaic(df_plot, [x_field, y_field], gap=0.01, title=x_field + " vs " + y_field)
                    #plt.savefig(directory + "/" + x_field + "_vs_" + y_field + "_Mosaic.png", bbox_inches='tight')
                    #plt.clf()
                    #plt.close()

                    #Heat Map
                    plt.figure(figsize=(25,20))
                    sns.heatmap(pd.crosstab([df[x_field]], [df[y_field]]), cmap="Blues", annot=False, cbar=True)
                    plt.savefig(directory + "/" + x_field + "_vs_" + y_field + "_Heatmap.png", bbox_inches='tight')
                    plt.clf()
                    plt.close()

                    #Scatter plot
                    plt.scatter(df_plot[x_field],df_plot[y_field])
                    plt.xlabel(x_field)
                    plt.ylabel(y_field)
                    rcParams['figure.figsize'] = 20, 10
                    plt.savefig(directory + "/" + x_field + "_vs_" + y_field + "_Scatter.png", bbox_inches='tight')
                    plt.clf()
                    plt.close()
                    charts_created = charts_created + 1

                except:
                    #print("Error on " + x_field + " vs " + y_field)
                    charts_errored = charts_errored + 1                        
                    pass
                    
    print("Run Time: " + str(datetime.datetime.now() - start_time) + "  Created: ", str(charts_created) +           "  Skipped: " + str(charts_skipped) + "  Errors: " + str(charts_errored))
    
    
def internal_date_binary_analysis(df, directory, date_fields, unique_cutoff, sample_size):
    
    start_time = datetime.datetime.now()
    print("Internal date binary analysis - " + str(directory))
    c = 0
    charts_created = 0
    charts_errored = 0
    
    #Check to see if directory exists
    check_directory(directory)
    
    #Find all unique combinations of the fields
    unique_field_pairs, charts_skipped = find_unique_combos(df, date_fields, date_fields, unique_cutoff)
    
    #Loop through the fields
    for x_field,y_field in unique_field_pairs:
        
        #Increment the counter
        c = c + 1
        
        #Check to make sure the field is in the dataframe
        if x_field in df.columns and y_field in df.columns:
                        
            #Filter out erroneous dates (prior to 1900)
            df_plot = df[(df[x_field]>pd.Timestamp(1900,1,1)) & (df[y_field]>pd.Timestamp(1900,1,1))]

            #Make sure there are records left after filtering out bad data
            if len(df_plot) > 0:

                #randomly sample the data (to improve run time)
                df_plot = sample_data(df_plot, len(df), sample_size)
                #print("df_plot: ", len(df_plot))
                #print(str(c) + "/" + str(len(unique_field_pairs)) + ": " + x_field + " vs " + y_field + \
                #" - TR:" + str(len(df_plot)) + " UX:" + str(df[x_field].nunique()) + " UY:" + str(df[y_field].nunique()))

                try:

                    #Scatter plot
                    rcParams['figure.figsize'] = 12, 10
                    plt.plot_date(df_plot[x_field],df_plot[y_field])
                    plt.xlabel(x_field)
                    plt.ylabel(y_field)
                    plt.title(x_field + "_vs_" + y_field)
                    plt.savefig(directory + "/" + x_field + "_vs_" + y_field + "_Scatter.png", bbox_inches='tight')
                    plt.clf()
                    plt.close() 
                    charts_created = charts_created + 1

                except:
                    #print("Error on " + x_field + " vs " + y_field)
                    charts_errored = charts_errored + 1
                    pass


    print("Run Time: " + str(datetime.datetime.now() - start_time) + "  Created: ", str(charts_created) +           "  Skipped: " + str(charts_skipped) + "  Errors: " + str(charts_errored))
    
    
def external_category_measure_binary_analysis(df, directory, category_fields, measure_fields, unique_cutoff, sample_size):
    
    start_time = datetime.datetime.now()
    print(directory)
    c = 0
    charts_created = 0
    charts_errored = 0
    
    #Check to see if directory exists
    check_directory(directory)
    
    #Find all unique combinations of the fields
    unique_field_pairs, charts_skipped = find_unique_combos(df, category_fields, measure_fields, unique_cutoff)
    
    #Loop through the fields
    for x_field,y_field in unique_field_pairs:
        
        #Increment the counter
        c = c + 1
        
        #Check to make sure the field is in the dataframe
        if x_field in df.columns and y_field in df.columns:
                        
            #Category Plots
            #if 0 < (df[x_field].nunique() + df[y_field].nunique()) < unique_cutoff:

            #remove the blank and 0 values and convert numbers to strings (otherwise it will error)
            df_plot = df[[x_field,y_field]].copy().dropna()
            df_plot = df_plot[df_plot[y_field] > 0]
            df_plot[x_field] = df_plot[x_field].astype(str)
            df_plot[y_field] = df_plot[y_field].astype(str)

            #Make sure there are records left after filtering out bad data
            if len(df_plot) > 0:

                #randomly sample the data (to improve run time)
                df_plot = sample_data(df_plot, len(df), sample_size)
                #print(str(c) + "/" + str(len(unique_field_pairs)) + ": " + x_field + " vs " + y_field + \
                #" - TR:" + str(len(df_plot)) + " UX:" + str(df[x_field].nunique()) + " UY:" + str(df[y_field].nunique()))

                try:

                    #Mosaic Plot
                    #rcParams['figure.figsize'] = 20, 12
                    #mosaic(df_plot, [x_field, y_field], gap=0.01, title=x_field + " vs " + y_field)
                    #plt.savefig(directory + "/" + x_field + "_vs_" + y_field + ".png", bbox_inches='tight')
                    #plt.clf()
                    #plt.close()

                    #Heatmap
                    plt.figure(figsize=(25,20))
                    sns.heatmap(pd.crosstab([df[x_field]], [df[y_field]]), cmap="Blues", annot=False, cbar=True)
                    plt.savefig(directory + "/" + x_field + "_vs_" + y_field + "_Heatmap.png", bbox_inches='tight')
                    plt.clf()
                    plt.close()
                    charts_created = charts_created + 1

                except:
                    #print("Error on " + x_field + " vs " + y_field)
                    charts_errored = charts_errored + 1
                    pass
                    


    print("Run Time: " + str(datetime.datetime.now() - start_time) + "  Created: ", str(charts_created) +           "  Skipped: " + str(charts_skipped) + "  Errors: " + str(charts_errored))
    
    
def external_date_measure_binary_analysis(df, directory, date_fields, measure_fields, unique_cutoff, sample_size):
    
    start_time = datetime.datetime.now()
    print(directory)
    c = 0
    charts_created = 0
    charts_errored = 0
    
    #Check to see if directory exists
    check_directory(directory)
    
    #Find all unique combinations of the fields
    unique_field_pairs, charts_skipped = find_unique_combos(df, date_fields, measure_fields, unique_cutoff)
    
    #Loop through the fields
    for x_field,y_field in unique_field_pairs:
        
        #Increment the counter
        c = c + 1
        
        #Check to make sure the field is in the dataframe
        if x_field in df.columns and y_field in df.columns:
                        
            #print(x_field + " vs " + y_field)

            #Filter out null values
            df_plot = df[[x_field,y_field]].copy().dropna()
            
            #Filter out erroneous dates (prior to 1900)
            df_plot = df_plot[df_plot[x_field]>pd.Timestamp(1900,1,1)]

            
            #Make sure there are records left after filtering out bad data
            if len(df_plot) > 0:

                #randomly sample the data (to improve run time)
                df_plot = sample_data(df_plot, len(df), sample_size)
                #print(str(c) + "/" + str(len(unique_field_pairs)) + ": " + x_field + \
                #" vs " + y_field + " - TR:" + str(len(df_plot)) + " UX:" + str(df[x_field].nunique()) + \
                #" UY:" + str(df[y_field].nunique()))

                try:
            
                    #Scatter plot
                    rcParams['figure.figsize'] = 12, 10
                    plt.plot_date(df_plot[x_field],df_plot[y_field])
                    plt.xlabel(x_field)
                    plt.ylabel(y_field)
                    plt.title(x_field + "_vs_" + y_field)
                    plt.savefig(directory + "/" + x_field + "_vs_" + y_field + "_Scatter.png", bbox_inches='tight')
                    plt.clf()
                    plt.close() 
                        
                    charts_created = charts_created + 1 

                except:
                    #print("Error on " + x_field + " vs " + y_field)
                    charts_errored = charts_errored + 1
                    pass

    print("Run Time: " + str(datetime.datetime.now() - start_time) + "  Created: ", str(charts_created) +            "  Skipped: " + str(charts_skipped) + "  Errors: " + str(charts_errored))
    
    
#######################################
#FUNCTIONS FOR DATA VISUALIZATION
def data_correlation_graph(df):
    f, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.35)
    plt.autoscale()
    
    corr_data = df.corr()
    sns.heatmap(corr_data,
                mask=np.zeros_like(corr_data, dtype=np.bool), 
                cmap=sns.diverging_palette(10, 220, as_cmap=False),
                vmin=-1, vmax=1,
                square=True, 
                ax=ax)
    
    fig_name = 'fig_cor_plot.png'
    f.savefig('./figures/' + str(fig_name),  dpi=70)
    plt.close('all')

    
def numeric_column_plot(df, directory, fields):
#create boxplot, histogram, violin of numeric data columns
    start_time = datetime.datetime.now()

    print("Numeric Analysis - Measures - " + directory)
    
    #Check to see if directory exists
    check_directory(directory)
    
    #For each numeric column in the list
    for x, col_name in enumerate(fields):
        #Create a copy of the column values without nulls or NA
        no_null_col = df[col_name].dropna()
        
        #Calculate the 95 percentile of the values
        q25 = np.percentile(no_null_col, 25)
        q75 = np.percentile(no_null_col, 75)    
        q95 = np.percentile(no_null_col, 95)
        
        # Plot the graphs
        fig3 = plt.figure(figsize=(20,15))
        plt.subplots_adjust(wspace=0.4, hspace=0.35)
    
        ax1 = fig3.add_subplot(3,2,1)
        ax1.set_title("Box plot for all the values", fontsize=20)
        plt.setp(ax1.get_xticklabels(), ha="right", rotation=35)
        plt.setp(ax1.get_yticklabels(), ha="right", fontsize=15)
        ax1.boxplot(no_null_col)
    
        ax1 = fig3.add_subplot(3,2,2)
        ax1.set_title("Distribution of all values", fontsize=20)
        plt.setp(ax1.get_xticklabels(), ha="right", rotation=35, fontsize=15)
        plt.setp(ax1.get_yticklabels(), ha="right", fontsize=15)
        ax1.hist(no_null_col)
    
        ax1 = fig3.add_subplot(3,2,3)
        ax1.set_title("Box plot without outliers", fontsize=20)
        plt.setp(ax1.get_xticklabels(), ha="right", rotation=35, fontsize=15)
        plt.setp(ax1.get_yticklabels(), ha="right", fontsize=15)
        ax1.boxplot(no_null_col, showfliers=False)
    
        ax1 = fig3.add_subplot(3,2,4)
        ax1.set_title("Violin plot (<95% percentile)", fontsize=20)
        plt.setp(ax1.get_xticklabels(), ha="right", rotation=35, fontsize=15)
        plt.setp(ax1.get_yticklabels(), ha="right", fontsize=15)
        ax1.violinplot(no_null_col[no_null_col <= q95])
        
        ax1 = fig3.add_subplot(3,2,5)
        if len(no_null_col.value_counts()) >= 4: #Ensure at least 4 non-null values exist in the column
            df[u'quartiles'] = pd.qcut(df[col_name], 4, duplicates = 'drop') #Will crash if duplicates aren't dropped
            df.boxplot(column= col_name, by=u'quartiles', ax = ax1)
            plt.suptitle('')
        else:
            print("Insufficient data in column " + str(col_name) + " for a quartile box plot")
        ax1.set_title("Box plot for quartiles (all values)", fontsize=20)
        plt.setp(ax1.get_xticklabels(), ha="right", rotation=35, fontsize=15)
        plt.setp(ax1.get_yticklabels(), ha="right", fontsize=15)
        
        #Histogram with bin ranges, counts and percentile color
        ax1 = fig3.add_subplot(3,2,6)
        ax1.set_title("Histogram (<95% percentile)", fontsize=20)
        plt.setp(ax1.get_xticklabels(), ha="right", rotation=35, fontsize=15)
        plt.setp(ax1.get_yticklabels(), ha="right", fontsize=15)
    
        #Take only the data less than 95 percentile
        data = no_null_col[no_null_col <= q95]
    
        #Colours for different percentiles
        perc_25_colour = 'gold'
        perc_50_colour = 'green'
        perc_75_colour = 'deepskyblue'
    
        #counts  = numpy.ndarray of count of data ponts for each bin/column in the histogram
        #bins    = numpy.ndarray of bin edge/range values
        #patches = a list of Patch objects.
        #        each Patch object contains a Rectnagle object. 
        #        e.g. Rectangle(xy=(-2.51953, 0), width=0.501013, height=3, angle=0)
        counts, bins, patches = ax1.hist(data, bins=10, facecolor=perc_50_colour, edgecolor='gray')
    
        #Set the ticks to be at the edges of the bins.
        ax1.set_xticks(bins.round(2))
        plt.xticks(rotation=70, fontsize=15)
    
        #Change the colors of bars at the edges
        for patch, leftside, rightside in zip(patches, bins[:-1], bins[1:]):            
            if leftside < q25:
                patch.set_facecolor(perc_25_colour)
            elif leftside > q75:
                patch.set_facecolor(perc_75_colour)
    
        #Calculate bar centre to display the count of data points and %
        bin_x_centers = 0.5 * np.diff(bins) + bins[:-1]
        bin_y_centers = ax1.get_yticks()[1] * 0.25
    
        #Display the the count of data points and % for each bar in histogram
        for i in range(len(bins)-1):
            bin_label = "{0:,}".format(counts[i]) + "  ({0:,.2f}%)".format((counts[i]/counts.sum())*100)
            plt.text(bin_x_centers[i], bin_y_centers, bin_label, rotation=90, rotation_mode='anchor')
    
        #Create legend + title
        fig3.suptitle("Profile of column  " + col_name, fontsize=25)  #Title for the whole figure
        handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [perc_25_colour, perc_50_colour, perc_75_colour]]
        labels = ["0-25 Percentile","25-75 Percentile", "75-95 Percentile"]
        plt.legend(handles, labels, bbox_to_anchor=(0.4, 0, 0.85, 0.99))
        
        #Save file
        fig3.savefig(directory + "/" + col_name + ".png", dpi = 50)
        plt.close('all')
        
    print("Run Time: ", datetime.datetime.now() - start_time)
    
    
def corrplot(df, directory, combolist):
    #Create scatterplots for correlated numerical columns
    # Combolist is a list of lists.  Each inner list should have 2 column names in dataframe df
    start_time = datetime.datetime.now()

    print("Correlation Plots - " + directory)
    
    #Check to see if directory exists
    check_directory(directory)
    
    for combo in combolist:
        #Drop numerical outliers (more than 3 standard deviations away)
        newdf = df[combo]
        newdf['col0zscore'] = (newdf[combo[0]] - newdf[combo[0]].mean())/newdf[combo[0]].std(ddof=0)
        newdf['col1zscore'] = (newdf[combo[1]] - newdf[combo[1]].mean())/newdf[combo[1]].std(ddof=0)
        newdf = newdf[(np.abs(newdf['col0zscore']) <= 3)]
        newdf = newdf[(np.abs(newdf['col1zscore']) <= 3)]
        
        #Plot the remaining data
        plot = newdf.plot(kind = 'scatter', x=combo[0], y=combo[1], c = 'DarkBlue', legend = False)
        plot.set_title("Scatterplot of " +str(combo[0]) + ' vs. ' + str(combo[1]), {'fontsize':25, 'fontweight':1, 'verticalalignment':'baseline', 'horizontalalignment':'center'})
        plot.set_xlabel(str(combo[0]), fontsize=25)
        plot.set_ylabel(str(combo[1]), fontsize=25)
        plot.tick_params(axis='y',labelsize=20)
        plot.tick_params(axis='x',labelsize=20)
        fig = plot.get_figure()
        fig.set_size_inches(10,10)
        fig.savefig(directory + "/" + str(combo[0]) + "vs" + str(combo[1]) + ".png", bbox_inches = "tight")
        plt.clf()
        plt.close()
    
    print("Run Time: ", datetime.datetime.now() - start_time)
    
    
#Functions for creating the categorical correlation plot
def replace_nan_with_value(x, y):
    #This function fills missing values for the conditional_entropy function and the theils_u function
    x = [v if v == v and v is not None else 0.0 for v in x] #NaN != NaN
    y = [v if v == v and v is not None else 0.0 for v in y]
    return x, y

    
def conditional_entropy(x, y):
    #Calculates the conditional entropy of x given y: S(x|y)
    # Wikipedia:  https://en.wikipedia.org/wiki/Conditional_entropy
    # Github reference:  https://github.com/shakedzy/dython
    # Parameters:
    # x: list/NumPy ndarray/Pandas Series (A sequence of measurements)
    # y: list/NumPy ndarray/Pandas Series (A sequence of measurements)
    x, y = replace_nan_with_value(x, y)
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy)
    return entropy


def theils_u(x, y):
    #Calculates Theil's U statistic (Uncertainty coefficient) for categorical-categorical association.
    # This is the uncertainty of x given y: The value is on the range of [0,1]
    # 0 means y provides no information about x, and 1 means y provides full information about x.
    # This is an asymmetric coefficient: U(x,y) != U(y,x)
    # Wikipedia:  https://en.wikipedia.org/wiki/Uncertainty_coefficient
    # Github reference:  https://github.com/shakedzy/dython
    # This returns a float in the range of [0,1]
    # Parameters:
    # x: list/NumPy ndarray/Pandas Series (A sequence of categorical measurements)
    # y: list/NumPy ndarray/Pandas Series (A sequence of categorical measurements)
    x, y = replace_nan_with_value(x, y)
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return(s_x - s_xy) / s_x
        

def cat_associations(df, categorical_columns, **kwargs):
    #Calculates the strength-of-association of features in the data-set using
    # Theil's U for categorical-categorical cases (defined above).
    # Returns a dataframe of the strength-of-association between the submitted features.
    # Parameters:
    # dataset: NumPy ndarray/Pandas DataFrame (the dataset we're working with)
    # categorical_columns: string/list/NumPy ndarray.  Names of the columns with categorical values
    
    print('Calculating Category-Category Associations')
    
    #Add new category to columns
    for column in df:
        df[column] = df[column].astype('category')
        df[column] = df[column].cat.add_categories('Unknown')
    df.fillna('Unknown', inplace=True)
    if categorical_columns is None: #If no categorical columns, cannot create the plot
        print("Cannot define categorical-categorical associations, no submitted categorical columns.")
        return df
    
    #Calculate Thiel U strength-of-associations for our categorical values
    columns = categorical_columns
    corr = pd.DataFrame(index=columns, columns=columns) #Dataframe to track correlations
    for i in range(0, len(columns)):
        for j in range(i, len(columns)):
            MemUsage = df.memory_usage().sum()
            print('Col1: ' + str(i) + '    Col2: ' + str(j)) ##$
            if i == j: #Columns perfectly correlate with themselves, should have value of 1.0
                corr[columns[i]][columns[j]] = 1.0
            else:
                corr[columns[j]][columns[i]] = theils_u(df[columns[i]],df[columns[j]])
                corr[columns[i]][columns[j]] = theils_u(df[columns[j]],df[columns[i]])
    corr.fillna(value=np.nan, inplace=True) #If some values are somehow missed, fill missing values with np.nan
    
    #Create plot of correlation values
    f, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.35)
    plt.autoscale()
    if len(categorical_columns) > 10:  #Less detailed map for large numbers of categories
        sns.heatmap(corr, 
                    mask=np.zeros_like(corr, dtype=np.bool), 
                    #cmap=sns.diverging_palette(10, 220, as_cmap=False), #Makes color palette range from red to blue
                    vmin=0, vmax=1,
                    square=True, 
                    ax=ax)
    else:  #More detailed plot for small numbers of categories
        sns.heatmap(corr,
                annot=kwargs.get('annot', True),
                fmt=kwargs.get('fmt', '.2f'),
                ax=ax)
    fig_name = 'fig_cat_cor_plot.png'
    f.savefig('./figures/' + str(fig_name), dpi=70)
    plt.close('all')
    print('Categorical Associations Completed')
    return corr
    

#Functions for creating the categorical-numeric correlation plot
def convert(data, to): #Helps with various data conversions
    converted = None
    if to == 'array':
        if isinstance(data, np.ndarray):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values
        elif isinstance(data, list):
            converted = np.array(data)
        elif isinstance(data, pd.DataFrame):
            converted = data.as_matrix()
    elif to == 'list':
        if isinstance(data, list):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values.tolist()
        elif isinstance(data, np.ndarray):
            converted = data.tolist()
    elif to == 'dataframe':
        if isinstance(data, pd.DataFrame):
            converted = data
        elif isinstance(data, np.ndarray):
            converted = pd.DataFrame(data)
    else:
        raise ValueError("Unknown data conversion: {}".format(to))
    if converted is None:
        raise TypeError('cannot handle data conversion of type: {} to {}'.format(type(data), to))
    else:
        return converted

def correlation_ratio(categories, measurements):
    #Calculates the Correlation Ratio (sometimes marked by the greek letter Eta)
    # for categorical-continuous association.    
    # Demonstrates if given a continous value of a measurement if the associated category is determinable.
    # Values range from 0 to 1.  0 indicates category can't be determined, 1 means it's determinable with 100% certainty.
    # Wikipedia:  https://en.wikipedia.org/wiki/Correlation_ratio
    # Github reference:  https://github.com/shakedzy/dython
    # Returns: A float ranging from 0 to 1
    # Parameters:
    # categories: list/NumPy ndarray/Pandas Series.  A sequence of categorical measurements.
    # measurements: list/NumPy ndarray/Pandas Series.  A sequence of continuous measurements.
    
    categories, measurements = replace_nan_with_value(categories, measurements)
    categories = convert(categories, 'array')
    measurements = convert(measurements, 'array')
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) /np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def catnum_associations(df, categorical_columns, numeric_columns, **kwargs):
    #Calculates the strength-of-association of features in the data-set using
    # Correlation Ratio for continuous-categorical cases (defined above).
    # Parameters: 
    # dataset: NumPy ndarray/Pandas DataFrame (the dataset we're working with)
    # categorical_columns: string/list/NumPy ndarray.  Names of the columns with categorical values
    # numeric_columns: string/list/NumPy ndarray.  Names of the columns with numeric values
    
    if categorical_columns is None: #If no categorical columns, cannot create the plot
        print("Cannot define numeric-categorical associations, no submitted categorical columns.")
        return df
    if numeric_columns is None: #If no numeric columns, cannot create the plot
        print("Cannot define numeric-categorical associations, no submitted numeric columns.")
        return df
    
    print('Calculating Category-Measure Associations')
    
    #Calculate Correlation Ratio values for our numeric-categorical pairs
    corr = pd.DataFrame(index=categorical_columns, columns=numeric_columns)
    for i in categorical_columns:
        for j in numeric_columns:
            MemUsage = df.memory_usage().sum()
            print('Col1: ' + str(i) + '    Col2: ' + str(j)) ##$
            eta = correlation_ratio(df[i], df[j])
            corr.at[i, j] = eta
    corr.fillna(value=np.nan, inplace=True)
    
    #Create plot of correlation values
    f, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.35)
    plt.autoscale()
    if len(categorical_columns + numeric_columns) > 10:  #Less detailed map for large numbers of category/column pairs
        sns.heatmap(corr, 
                    mask=np.zeros_like(corr, dtype=np.bool), 
                    #cmap=sns.diverging_palette(10, 220, as_cmap=False), #Makes color palette range from red to blue
                    vmin=0, vmax=1,
                    square=True, 
                    ax=ax)
    else:  #More detailed plot for small numbers of category/column pairs
        sns.heatmap(corr,
                annot=kwargs.get('annot', True),
                fmt=kwargs.get('fmt', '.2f'),
                ax=ax)
    fig_name = 'fig_catnum_cor_plot.png'
    f.savefig('./figures/' + str(fig_name), dpi=70)
    plt.close('all')
    
    print('Category-Measure Associations Completed')
    return corr


def cat_cat_heatmap(df, cols):
    #Parameters:
    # df = input dataframe
    # cols = list with 2 values to use as axes (ex: ['COL1', 'COL2'])
    directory = "figures" #Directory to save plots in
    
    #Get category names
    cat1 = cols[0]
    cat2 = cols[1]
    
    #Find top 15 values for each column
    cat1values = df[cat1].value_counts()[:15].index.tolist()
    cat2values = df[cat2].value_counts()[:15].index.tolist()
    
    #Remove rows that don't contain top 15 values from the columns
    df = df[df[cat1].isin(cat1values)]
    df = df[df[cat2].isin(cat2values)]
    df = df.loc[:, [cat1, cat2]] #Only keep the 2 columns we need
    
    #Attempt to create the heatmap
    try:       
        f, ax = plt.subplots(figsize=(25, 20))
        
        crosstab = df.groupby([cat1, cat2])[cat2].count().unstack().fillna(0) #Reorganize data for heatmap
        sns.heatmap(crosstab, cmap="Blues", annot=False, cbar=True)
        
        ax.set_title("Heatmap of " + str(cat1) + " vs " + str(cat2), {'fontsize':30, 'fontweight':1, 'verticalalignment':'baseline', 'horizontalalignment':'center'})
        ax.tick_params(axis = 'x', labelsize=20, labelrotation=90)
        ax.tick_params(axis = 'y', labelsize=20, labelrotation=0) 
        ax.set_xlabel(str(cat2), fontsize=25)
        ax.set_ylabel(str(cat1), fontsize=25)
        
        xticklabels = cat2values
        yticklabels = cat1values
        for i in range(len(xticklabels)):
            xticklabels[i] = xticklabels[i][:20] #Only take first 20 characters
        for i in range(len(yticklabels)):
            yticklabels[i] = yticklabels[i][:20] #Only take first 20 characters
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)
        
        plt.subplots_adjust(bottom=0.35)
        plt.autoscale()
        
        f.savefig(directory + "/" + cat2 + "_vs_" + cat1 + "_Heatmap.png", bbox_inches='tight')
        plt.close('all')
        
    except:
        pass
    

def cat_num_heatmap(df, cols):
    #Parameters: 
    # df = input dataframe
    # cols = list with 2 values to use as axes (ex: ['COL1', 'COL2'])
    directory = "figures" #Directory to save plots in
    
    #Get categorical and numeric column names
    cat = cols[0]
    num = cols[1]
    
    #Find top 15 values for each categorical column
    catvalues = df[cat].value_counts()[:15].index.tolist()
    
    #Remove rows that don't contain top 15 values from the categorical column
    df = df[df[cat].isin(catvalues)]
    df = df.loc[:, [cat,num]] #Only keep the two columns we need
    
    #Orient the dataframe correctly for making a boxplot
    df = df.pivot(columns=cat, values=num) 
    
    #Attempt to create the plot
    try:
        
        ax = df.plot(kind='box', showfliers=False, rot=90, fontsize=20, figsize=(25,20))
        ax.set_xlabel(cat, fontsize=20)
        
        #Potentially shorten x-tick labels
        xticklabels = catvalues
        for i in range(len(xticklabels)):
            xticklabels[i] = xticklabels[i][:20] #Only take first 20 characters
        ax.set_xticklabels(xticklabels) 
        
        ax.set_title("Plot of " + str(num) + " vs " + str(cat), {'fontsize':30, 'fontweight':1, 'verticalalignment':'baseline', 'horizontalalignment':'center'})
        ax.set_xlabel(cat, fontsize=25)
        ax.set_ylabel(num, fontsize=25)
    
        fig = ax.get_figure()
        fig.savefig(directory + "/" + num + "_vs_" + cat + "_Plot.png", bbox_inches='tight')
        plt.clf()
        plt.close()
        
    except:
        pass
    
    
#Determing Data Encoding
import chardet
def determine_encoding(folder, filename, extension):
    #Parameters:
    # folder, filename, and extension of the file
    # ex: 'C:/SHARE/Data/', 'iISDDC_ADHOC_DATA_CY06_to_04-30-19', '.csv'
    
    file = folder + filename + extension #Get overall file name+location
    data_object = open(file, 'rb') #Open file in binary mode
    
    #Only read first 1 million bytes as a bytearray (improves speed; chardet, by default, reads large files in entirety and will be very slow)
    ba = bytearray(data_object.read(1000000)) 
    
    encode_dict = chardet.detect(ba) #Detects filetype and accuracy (example output:  {'encoding': 'UTF-16', 'confidence': 1.0, 'language': ''})
    return str(encode_dict['encoding']) #Returns a string with encoding type (e.g. 'UTF-16')

#Convert chardet's output string to appropriate string for pandas
def encoding_string(chardet_output):
    #Parameters:
    # chardet_output: string output by chardet naming the encoding type (e.g. 'ascii', 'EUC-JP', 'UTF-16', etc.)
    #Returns:
    # pandas_input: string for the encoding that pandas will interpret correctly (e.g. 'latin1', 'euc_jp', 'utf_16', etc.)
    
    #Use a dictionary to link the chardet outputs and the pandas inputs
    encode_dict = {'Big5':'big5', 'GB2312':'gb2312', 'GB18030':'gb18030', 'HZ-GB-2312':'hz', 'EUC-JP':'euc_jp', 'SHIFT_JIS':'shift_jis',
                   'ISO-2022-JP':'iso-2022-jp', 'EUC-KR':'euc_kr', 'ISO-2022-KR':'iso-2022-kr', 'KOI8-R':'koi8_r', 
                   'MacCyrillic':'maccyrillic', 'IBM855':'IBM855', 'IBM866':'IBM866', 'ISO-8859-5':'iso-8859-5', 
                   'windows-1251':'windows-1251', 'ISO-8859-2':'iso-8859-2', 'windows-1250':'windows-1250',
                   'ISO-8859-1':'iso-8859-1', 'windows-1252':'windows-1252', 'ISO-8859-7':'iso-8859-7',
                   'windows-1253':'windows-1253', 'ISO-8859-8':'iso-8859-8', 'windows-1255':'windows-1255',
                   'UTF-16':'utf_16', 'UTF-8':'utf_8', 'ascii':'latin1'}
    try:
        pandas_input = encode_dict[chardet_output]
        return pandas_input
    except:
        print('Encoding type used is unknown.  Manually input encoding type or convert to a more common type (e.g. latin1, utf_16, etc.).')
    
    
################################################################################
#COMMON FUNCTIONS FOR GEODESIC DISTANCE

#This is from the "Mario's Entangled Bank" blog (http://pineda-krch.com) of 
#Mario Pineda-Krch, a theoretical biologist at the University of Alberta.

###CONVERT DEGREES TO RADIANS
#Note that for the decimal degrees positive latitudes are north of the equator, 
#negative latitudes are south of the equator. Positive longitudes are east of 
#Prime Meridian, negative longitudes are west of the Prime Meridian.
def deg2rad(deg):
    return(deg*np.pi/180)


###CONVERT KM TO MILES
def kmtomi(km):
    return(km / 1.60934)


###CALCULATE DISTANCES
#Calculates the geodesic distance between two points specified by radian 
# latitude/longitude using the Spherical Law of Cosines (slc)
# NOTE - very fast, but doesn't reflect actual planetary shape

def slc(long1, lat1, long2, lat2):
    R = 6371 #Earth mean radius (km)
    d = np.arccos(np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2) * np.cos(long2-long1)) * R
    return(d) #Distance in km

#Calculates the geodesic distance between two points specified by radian 
# latitude/longitude using the Haversine formula (hf)
# NOTE - very fast, but doesn't reflect actual planetary shape
def hf(long1, lat1, long2, lat2):
    R = 6371 #Earth mean radius (km)
    delta_long = (long2 - long1)
    delta_lat = (lat2 - lat1)
    a = np.sin(delta_lat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_long/2)**2
    c = 2 * np.arcsin(min(1,np.sqrt(a)))
    d = R * c
    return(d) #Distance in km

#Calculates the geodesic distance between two points specified by radian 
# latitude/longitude using Vincenty inverse formula for ellipsoids (VIF)
# NOTE - slow, but more accurate
def vif(long1, lat1, long2, lat2):
    # WGS-84 ellipsoid parameters
    a = 6378137           # length of major axis of the ellipsoid (radius at equator)
    b = 6356752.314245    # length of minor axis of the ellipsoid (radius at the poles)
    f = 1/298.257223563   # flattening of the ellipsoid
    
    L = long2 - long1                       # difference in longitude
    U1 = np.arctan((1-f) * np.tan(lat1))   # reduced latitude
    U2 = np.arctan((1-f) * np.tan(lat2))   # reduced latitude
    sinU1 = np.sin(U1)
    cosU1 = np.cos(U1)
    sinU2 = np.sin(U2)
    cosU2 = np.cos(U2)
    
    cosSqAlpha = None
    sinSigma = None
    cosSigma = None
    cos2SigmaM = None
    sigma = None
    
    lambda_ = L #'lambda' is a reserved term in python
    lambdaP = 0
    iterLimit = 100
    if lambda_ == 0: #If L=0, the longitudes are identical, AKA same port
        return(0) #If the above if statement is not used, code will fail; while loop not performed->undefined var
    else:
        while ((abs(lambda_-lambdaP) > 1e-12) & (iterLimit>0)):
            sinLambda = np.sin(lambda_)
            cosLambda = np.cos(lambda_)
            sinSigma = np.sqrt( (cosU2*sinLambda) * (cosU2*sinLambda) + 
                             (cosU1*sinU2-sinU1*cosU2*cosLambda) *
                             (cosU1*sinU2-sinU1*cosU2*cosLambda) )
            if (sinSigma==0): return(0) #Co-incident points
            cosSigma = sinU1*sinU2 + cosU1*cosU2*cosLambda
            sigma = np.arctan2(sinSigma, cosSigma)
            sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma
            cosSqAlpha = 1 - sinAlpha*sinAlpha
            cos2SigmaM = cosSigma - 2*sinU1*sinU2/cosSqAlpha
            if pd.isnull(cos2SigmaM) == True: cos2SigmaM = 0 #Equatorial line: cosSqAlpha=0
            C = f/16*cosSqAlpha*(4+f*(4-3*cosSqAlpha))
            lambdaP = lambda_
            lambda_ = L + (1-C) * f * sinAlpha * (sigma + C*sinSigma*(cos2SigmaM + C*cosSigma*(-1+2*cos2SigmaM*cos2SigmaM)))
            iterLimit = iterLimit - 1
    
        if (iterLimit==0): return(np.nan) # formula failed to converge #option:  also if iterLimit==100 (no iter)
        uSq = cosSqAlpha * (a*a - b*b) / (b*b)
        A = 1 + uSq/16384*(4096+uSq*(-768+uSq*(320-175*uSq)))
        B = uSq/1024 * (256+uSq*(-128+uSq*(74-47*uSq)))
        deltaSigma = B*sinSigma*(cos2SigmaM+B/4*(cosSigma*(-1+2*cos2SigmaM**2) - 
                                             B/6*cos2SigmaM*(-3+4*sinSigma**2) * (-3+4*cos2SigmaM**2)))
        s = b*A*(sigma-deltaSigma) / 1000
    
        return(s) # Distance in km
            
    
def vincenty(long1, lat1, long2, lat2):
    #Convert degrees to radians
    long1 = deg2rad(long1)
    lat1 = deg2rad(lat1)
    long2 = deg2rad(long2)
    lat2 = deg2rad(lat2)
    
    return(kmtomi(vif(long1, lat1, long2, lat2)))


def spherical(long1, lat1, long2, lat2):
    #Convert degrees to radians
    long1 = deg2rad(long1)
    lat1 = deg2rad(lat1)
    long2 = deg2rad(long2)
    lat2 = deg2rad(lat2)
    
    return(kmtomi(slc(long1, lat1, long2, lat2)))


###ALL METHODS EXECUTED
# Calculates the geodesic distance between two points specified by degrees (DD) 
# latitude/longitude using Haversine formula (hf), Spherical Law of Cosines 
#(slc) and Vincenty inverse formula for ellipsoids (vif)

def gcd(long1, lat1, long2, lat2):
    # Convert degrees to radians
    long1 = deg2rad(long1)
    lat1 = deg2rad(lat1)
    long2 = deg2rad(long2)
    lat2 = deg2rad(lat2)
    
    return(list(haversine = kmtomi(hf(long1, lat1, long2, lat2)),
               sphere = kmtomi(slc(long1, lat1, long2, lat2)),
               vincenty = kmtomi(vif(long1, lat1, long2, lat2))))
        
print("derivitizationUtils script loaded")
