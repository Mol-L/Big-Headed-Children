# =============================================================================
# The data analysis for the MSc thesis Big headed Children for Archaeological Sciences
#  at Leiden University
# Written by: 
#     Lara Molenaar
# MSc thesis supervisor: 
#     Dr. Rachel Schats
# 
# Data gathered by Lara Molenaar from the Middenbeemster collection of
# Leiden University
# Used datafiles: Age_and_accuracy.csv & Combined_siding.csv 
# =============================================================================
# =============================================================================
# Import of the packages and the data
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy as stats
from scipy import stats
from scipy.stats import norm
import numpy as np
import statsmodels.api as sm
from collections import namedtuple
import warnings
warnings.filterwarnings('ignore')


Measures = pd.read_csv(input("Enter here the Age_and_accuracy.csv path: "))
Measures = Measures.set_index("Feat_number")

# =============================================================================
# =============================================================================
# Section 1: descriptive statistics
MainData = Measures.loc[:,Measures.columns.str.contains("estim")==False]
MainData = MainData.loc[:,MainData.columns.str.contains("accuracy")==False]
MainData = MainData.loc[:,MainData.columns.str.contains("AgeRange")==False]
#locating only the actual measurements, not the accuracy nor the estimation of age

CentralMeasures = MainData.loc[:,MainData.columns.str.contains("L_")==False]
CentralMeasures = CentralMeasures.loc[:,CentralMeasures.columns.str.contains("R_")==False]
CentralMeasures = CentralMeasures.loc[:,CentralMeasures.columns.str.contains("number")==False]
#locating all the measures that don't have a siding focus and no find numbers

LeftMeasures = MainData.loc[:,MainData.columns.str.contains("L_")]
#locating the measures that only have a siding of left

RightMeasures = MainData.loc[:,MainData.columns.str.contains("R_")]
#locating the measures that only have a siding of right

def descriptives(dataframe):
    """
    A dataframe with the descriptives that are useful for this research

    Parameters
    ----------
    dataframe : DataFrame
        A dataframe that only contains number values.

    Returns
    -------
    DataFrame
        A dataframe with the count, median, mean, standard deviation, standard error,
        minimum measures and maximum measures.

    """
    DescrMeasures = {}
    for measure in dataframe:
        #create a Dictionary where the main important descriptives are shown 
            #per column
        measureCount = dataframe[measure].count() #the count
        measureMedian = round(dataframe[measure].median(),2) #the median
        measureMean = round(dataframe[measure].mean(),2) #the mean 
        measureSD = round(dataframe[measure].std(),2) #the Standard Deviation
        measureSEM = round(dataframe[measure].sem(),2) #the standard error of mean
        minmeasure = dataframe[measure].min() #the minimum measurement
        maxmeasure = dataframe[measure].max() #the maximum measurement
        DescrMeasures[measure] = {
            "Count": measureCount,
            "Median": measureMedian,
            "Mean":  measureMean,
            "Standard Deviation": measureSD,
            "Standard Error": measureSEM, 
            "Minimum": minmeasure,
            "Maximum": maxmeasure,
            }
    return pd.DataFrame(DescrMeasures) 
        #to make this into a dataframe i.o. dictionary, now it's pretty
            #and easily usable

CentralDescr = descriptives(CentralMeasures)
LeftDescr = descriptives(LeftMeasures)
RightDescr = descriptives(RightMeasures)
# =============================================================================
# =============================================================================
# Section 2: statistical analysis of the two side measurements


#first a function or the Wilcoxon signed rank test is created, as no version in scipy
    #also gives a ranked table, nor more then the W and p-value, and more must be reported
def Wilcoxon_Rank(dataset, x, y):
    """
    Function applying the Wilcoxon signed rank statistical test, two-taild

    Parameters
    ----------
    dataset : DataFrame
        The name of the dataframe where x and y are from.
    x : Series
        The first measure on which the Wilcoxon signed rank is preformed.
    y : Series
        The second measure on which the Wilcoxon signed rank is preformed

    Returns
    -------
    TUPLE
        A tuple collection which shows:
            Data; a DataFrame
                showing the x value, y-value, difference, absolute difference, 
                    rank, positive rank and negative rank
            RankedTable; A DataFrame 
            showing the information of the positive, negative and tied values
                including sample size, mean rank and sum of rank for each
            sample_size; a interger value 
                of the sample size used
            W; float
                The W value of the wilcoxon signed rank test
            z_score; float
                The Z-score of the Wilcoxon signed rank test
            pval; float
                The p-vale of the Wilcoxon signed rank test
    """
    difference = dataset[x] - dataset[y] # calculates the difference of x to y per row
    absdiff = abs(difference) # gives an absolute of the difference
    rank = absdiff.rank() # this ranks the absolute differences from 
        #lowest being 1 and heighest being the sample size
    positiverank = np.where(difference > 0, rank, np.nan)
        #this creates a new column where all rank values are copied into if the difference was positie
    negativerank = np.where(difference < 0, rank, np.nan)
        #this creates a new column where all rank values are copied into if the difference was negtive
    Data = pd.DataFrame({'measure 1': dataset[x], 
                         'measure 2': dataset[y], 
                         'difference': difference,
                         'absolute difference': absdiff,
                         'rank': rank,
                         'positive rank': positiverank,
                         'negative rank': negativerank}) #creates a dataset of the previous information
    n = len(absdiff) - Data['absolute difference'].isna().sum()  # calculates the number of samples
    pos_sum = Data['positive rank'].sum() #sums all rank values which are of a positive difference
    neg_sum = Data['negative rank'].sum() #sums all rank values which are of a negative difference
    W = min([pos_sum, neg_sum]) #establishes the W value, the lowest sum of ranks
    meanrank = (n*(n+1))/4 # calculates the mean of the sum of the rank values
    variance = (n*(n+1)*(2*n+1))/24 #calcutlates the variance
    std = variance**(1/2) # standard deviation of the vairance test
    zscore = abs(W-meanrank)/std # calculates the Z score of the test
    pval = 2*(1-norm.cdf(zscore)) # calculates the p-value
    posrank = len(positiverank) - Data['positive rank'].isna().sum() 
        #this shows the number of positive differences
    negrank = len(negativerank) - Data['negative rank'].isna().sum() 
        #this shows the number of negative differences
    meanpos = pos_sum/posrank #this gives the mean rank number of the positive differences
    meanneg = neg_sum/negrank #this gives the mean rank number of the negative differences
    ties = n - posrank - negrank # this shos the number of neither negative nor positive differences
    RankedTable = pd.DataFrame({'ranks': ["Negative Rank", "Positive Rank", "Ties", "Total"], 
                                'N': [negrank, posrank, ties, n],
                                'Mean Rank': [meanneg, meanpos, np.nan , np.nan],
                                'Sum of Ranks': [neg_sum, pos_sum, np.nan, np.nan]})
    Wilcoxon = namedtuple('wilcoxonsigned', ['Dataframe', 'RankedTable', 'sample_size', 'statistic', 'z_score', 'pval'])
        #this creates a tuple names Wilcoxon signed, where the dataframe which shows the differences and ranks
        # the sample size, the W score, the z-score and the p value are entered into by name
    return  Wilcoxon(Data, RankedTable, n, W, zscore, pval)
        #this returns this information as described by the names of the together named tuple
    
print()
WilcoxZygWidth = Wilcoxon_Rank(Measures, "L_ZygomaticWidth", "R_ZygomaticWidth")
print("The p-value is:",WilcoxZygWidth.pval.round(3), 
      "Which means that H0 cannot be rejected",
      "and there is no significant difference between", 
      "the right and the left measures of the Zygomatic.")
print("The W is:", WilcoxZygWidth.statistic.round(3))
print("The z-value is:", WilcoxZygWidth.z_score.round(3))
print()


WilcoxMRH = Wilcoxon_Rank(Measures, "L_MandibleRamusHeight", "R_MandibleRamusHeight")
print("The p-value is:", WilcoxMRH.pval.round(3), 
      "Which means that H0 cannot be rejected",
      "and there is no significant difference between", 
      "the right and the left measures of the Zygomatic.")
print("The W is:", WilcoxMRH.statistic.round(3))
print("The z-value is:", WilcoxMRH.z_score.round(3))
print()

# =============================================================================
# =============================================================================
# Section 3: Accuracy test
    #the combination of the two was done in excel because it's faster
    #as it would take a lot more coding else

CombinedMeasures = pd.read_csv(input("Enter here the Combined_siding.csv file path: "))
accuracy =  CombinedMeasures.loc[:,CombinedMeasures.columns.str.contains("accuracy")]
    #creates a dataset which only has the accuracy information of the dataset
    
    
YoungAccuracy = accuracy.loc[:,accuracy.columns.str.contains("old")==False]
    #all measures that don't have the word old are for indiviuals 2 or younger
OldAccuracy = accuracy.loc[:,accuracy.columns.str.contains("old")]
    #all measures that have the word old are for 2+ individuals

def ValueCount(dataframe):
    """
    Creating a dataframe which shows the value count of all columns

    Parameters
    ----------
    dataframe : DataFrame
        a dataframe which has the accuracy categorical values.

    Returns
    -------
    DataFrame
        A dataframe showing the number of categorical values

    """
    Accuracy = {}
    for measure in dataframe:
        #create a Dictionary where the main important descriptives are shown
        ValueCount = dataframe[measure].value_counts()
        Accuracy[measure] = pd.concat([ValueCount], axis=0)
    return pd.DataFrame(Accuracy)

AccurateListYoung = ValueCount(YoungAccuracy)
AccurateListOld = ValueCount(OldAccuracy)


estimages = CombinedMeasures.loc[:,CombinedMeasures.columns.str.contains("estim")]
    #creates a dataset which holds all the estimation numbers
YoungEstim = estimages.loc[:,estimages.columns.str.contains("old")==False]
OldEstim = estimages.loc[:,estimages.columns.str.contains("old")]

YoungEstimation = descriptives(YoungEstim)
OldEstimation = descriptives(OldEstim)

plt.figure(figsize=(7,5)) #creates a histplot with all chronological ages
sns.histplot(data=Measures, x="ArchivalAge", hue="AgeRange", 
             palette="Dark2", bins=12, shrink=.8, hue_order=("<2", "2-6", "6-12"))
plt.xlabel('Archival Age') # label labels
plt.ylabel('Count')
plt.title('Histplot counting the Archival Age of the dataset') 
plt.show()

plt.figure(figsize=(7,5)) #creates histplot with invidiuals 2 or younger
sns.histplot(data=YoungEstim, bins=12, shrink=.8, legend=False)
plt.xlabel('Estimatd Age of individuals <2') # label labels
plt.ylabel('Count')
plt.title('Histplot counting the estimated Age of the dataset') 
plt.show()

plt.figure(figsize=(7,5)) # creates a histplor with invidiuals 2+
sns.histplot(data=OldEstim, bins=12, shrink=.8, legend=False)
plt.xlabel('Estimatd Age of individuals 2-12') # label labels
plt.ylabel('Count')
plt.title('Histplot counting the estimated Age of the dataset') 
plt.show()

YoungMain = Measures.loc[Measures['AgeRange'] ==  '<2']
    #collects the data of the young individuals, with the dataset i.o. 
        #combined measures as both sides are still separate here
YoungEstimMain = YoungMain.iloc[:,YoungMain.columns.str.contains("estim")]
    #gives only the estimated age of the individuals
YoungEstimMain['mean'] = YoungEstimMain.mean(axis=1)
    #this creates a new column with the subject mean, and averages out the mean
        #estimated age based on the estim columns, per row. 


OldMain = Measures.loc[Measures['AgeRange'] != '<2']
OldEstimMain  = OldMain.iloc[:, OldMain.columns.str.contains("estim")]
OldEstimMain['mean'] = OldEstimMain.mean(axis=1)
    # same as before, looks only at the individuals that are not 2 or younger
    
    
plt.figure(figsize=(7,5))
sns.histplot(data=YoungEstimMain, x="mean", bins=12, shrink=.8,
             legend=False, color="#1b9e77")
plt.xlabel('Estimatd Age of individuals <2') # label labels
plt.ylabel('Count')
plt.title('Histplot counting the mean estimated Age') 
plt.show() 
    #a histplot that counts the mean estimated ages for individuals <2

plt.figure(figsize=(7,5))
sns.histplot(data=OldEstimMain, x="mean", bins=12, shrink=.8,
             legend=False, color="#7570b3")
plt.xlabel('Estimatd Age of individuals 2-12') # label labels
plt.ylabel('Count')
plt.title('Histplot counting the mean estimated Age') 
plt.show()
    #a histplot that counts the mean estimated ages for individuals 2-12

# =============================================================================
# =============================================================================
# Section 4: Wilcoxon signed rank tests to compare the estimations to archival age        

#first we estimate for all the individuals
MeanEstim = pd.concat([OldEstimMain['mean'], YoungEstimMain['mean']])
MeanAge = pd.concat([MeanEstim, Measures["ArchivalAge"]], axis=1)
    #combines the mean ages and the archival age based on the index; 
        #their feature number
youngmean = pd.concat([YoungEstimMain['mean'], YoungMain["ArchivalAge"]], axis=1)


TillSix = Measures.loc[Measures['AgeRange'] ==  '2-6'] 
TillSixEstim = TillSix.iloc[:,TillSix.columns.str.contains("estim")]
TillSixEstim['mean'] = TillSixEstim.mean(axis=1)
oldmean = pd.concat([TillSix['ArchivalAge'], TillSixEstim['mean']], axis=1)

TillTwelve = Measures.loc[(Measures['AgeRange'] ==  '<2') == False]
TillTwelveEstim = TillTwelve.iloc[:,TillTwelve.columns.str.contains("estim")]
TillTwelveEstim['mean'] = TillTwelveEstim.mean(axis=1)
twelvemean = pd.concat([TillTwelve['ArchivalAge'], TillTwelveEstim['mean']], axis=1)
descriptives(twelvemean)

#this part looks at all the ages
print()
AverageWillCox = Wilcoxon_Rank(MeanAge, "ArchivalAge", "mean")
print("The p-value is:", AverageWillCox.pval.round(3), 
      "Which means that H0 can be rejected and there is a",
      "significant difference between archival age and mean age")
print("The W is:", AverageWillCox.statistic.round(3))
print("The z-value is:", AverageWillCox.z_score.round(3))
print(AverageWillCox.RankedTable)
print()

meandescriptives = descriptives(MeanAge)
#the boxplot + mean shows that the main part is due to the young individuals
    #so what happens when we separate with age = 2
    
#this part focuses on the <2 individuals
YoungWillCox = Wilcoxon_Rank(youngmean, "ArchivalAge", "mean")
print("The p-value is:",YoungWillCox.pval.round(3), 
      "Which means that H0 cannot be rejected",
      "and there is no significant difference between", 
      "the archival age and the average estimated age.")
print("The W is:", YoungWillCox.statistic.round(3))
print("The z-value is:", YoungWillCox.z_score.round(3))
print(YoungWillCox.RankedTable)
print()

#This part focuses on the 2-6 individuals
OldWilcox = Wilcoxon_Rank(oldmean, "ArchivalAge", "mean")
print("The p-value is:",OldWilcox.pval.round(3), 
      "Which means that H0 can be rejected and there is a",
      "significant difference between archival age and mean age")
print("The W is:", OldWilcox.statistic.round(3))
print("The z-value is:", OldWilcox.z_score.round(3))
print(OldWilcox.RankedTable)
print()

#this part focuses on individuals 2-12
TwelveWilcox = Wilcoxon_Rank(twelvemean, "ArchivalAge", "mean")
print("The p-value is:",TwelveWilcox.pval.round(3), 
      "Which means that H0 can be rejected and there is a",
      "significant difference between archival age and mean age")
print("The W is:", TwelveWilcox.statistic.round(3))
print("The z-value is:", TwelveWilcox.z_score.round(3))
print(TwelveWilcox.RankedTable)
print()


#first we create a new DataFrame that looks only at the pinpoint age estimation and archival age
AgesOnly = CombinedMeasures.loc[:,CombinedMeasures.columns.str.contains("accuracy")==False]
AgesOnly = AgesOnly.loc[:,AgesOnly.columns.str.contains("number")==False]
AgesOnly = AgesOnly.loc[:,AgesOnly.columns.str.contains("Length")==False]
AgesOnly = AgesOnly.loc[:,AgesOnly.columns.str.contains("Width")==False]
AgesOnly = AgesOnly.loc[:,AgesOnly.columns.str.contains("Height")==False]
AgesOnly = AgesOnly.loc[:,AgesOnly.columns.str.contains("Mandible")==False]
AgesOnly = AgesOnly.loc[:,AgesOnly.columns.str.contains("Range")==False]

#now there's a new DataFrame which looks at the estimations for <2
YoungAgeEstim = AgesOnly.loc[:,AgesOnly.columns.str.contains("old")==False]

#Now we'll plot all the Wilcoxon signed rank tests for individuals <2
Wilcoxon_Rank(YoungAgeEstim, "ArchivalAge", "FH_estim")
Wilcoxon_Rank(YoungAgeEstim, "ArchivalAge", "FW_estim")
Wilcoxon_Rank(YoungAgeEstim, "ArchivalAge", "OsL_estim_young")
Wilcoxon_Rank(YoungAgeEstim, "ArchivalAge", "OsW_estim_young")
Wilcoxon_Rank(YoungAgeEstim, "ArchivalAge", "OBL_estim_young")
Wilcoxon_Rank(YoungAgeEstim, "ArchivalAge", "OBW_estim_young")
Wilcoxon_Rank(YoungAgeEstim, "ArchivalAge", "OBML_estim_young")
Wilcoxon_Rank(YoungAgeEstim, "ArchivalAge", "OLL_estim")
Wilcoxon_Rank(YoungAgeEstim, "ArchivalAge", "OLW_estim")
Wilcoxon_Rank(YoungAgeEstim, "ArchivalAge", "ZL_estim")
Wilcoxon_Rank(YoungAgeEstim, "ArchivalAge", "ZW_estim")
Wilcoxon_Rank(YoungAgeEstim, "ArchivalAge", "ML_estim_young")
Wilcoxon_Rank(YoungAgeEstim, "ArchivalAge", "MW_estim_young")
Wilcoxon_Rank(YoungAgeEstim, "ArchivalAge", "MH_estim_young")
Wilcoxon_Rank(YoungAgeEstim, "ArchivalAge", "ChinH_estim")
Wilcoxon_Rank(YoungAgeEstim, "ArchivalAge", "BW_estim_young")
Wilcoxon_Rank(YoungAgeEstim, "ArchivalAge", "RH_estim_young")
Wilcoxon_Rank(YoungAgeEstim, "ArchivalAge", "MML_estim_young")


#now there's a new DataFram which looks at the estimatios for >2
OldAgeEstim = AgesOnly.loc[:,AgesOnly.columns.str.contains("young")==False]
OldAgeEstim = OldAgeEstim.loc[:,OldAgeEstim.columns.str.endswith("estim")==False]

#now we'll plot all the Wilcoxon signed ranked tests for individuals >2
Wilcoxon_Rank(OldAgeEstim, "ArchivalAge", "OsL_estim_old")
Wilcoxon_Rank(OldAgeEstim, "ArchivalAge", "OsW_estim_old")
Wilcoxon_Rank(OldAgeEstim, "ArchivalAge", "OBL_estim_old")
Wilcoxon_Rank(OldAgeEstim, "ArchivalAge", "OBW_estim_old")
Wilcoxon_Rank(OldAgeEstim, "ArchivalAge", "OBML_estim_old")
Wilcoxon_Rank(OldAgeEstim, "ArchivalAge", "ML_estim_old")
Wilcoxon_Rank(OldAgeEstim, "ArchivalAge", "MW_estim_old")
Wilcoxon_Rank(OldAgeEstim, "ArchivalAge", "MH_estim_old")
Wilcoxon_Rank(OldAgeEstim, "ArchivalAge", "BB_estim_old")
Wilcoxon_Rank(OldAgeEstim, "ArchivalAge", "BW_estim_old")
Wilcoxon_Rank(OldAgeEstim, "ArchivalAge", "MML_estim_old")

# =============================================================================
# =============================================================================
# Section 5: Regression estimation
RegressionBase = CombinedMeasures.loc[:,CombinedMeasures.columns.str.contains("estim")==False]
RegressionBase = RegressionBase.loc[:,RegressionBase.columns.str.contains("accuracy")==False]
RegressionBase = RegressionBase.loc[:,RegressionBase.columns.str.contains("number")==False]
    #creates a new measure where only the actual measures are present
        #so no estim accuracy or feature number, but the Archival Age and AgeRange is present


def plot(dataframe):
    """
    Creating plots for the regressio models

    Parameters
    ----------
    dataframe : DataFrame
        The dataframe with the data that contains the "Age Range" column

    Returns
    -------
    plot : plot
        This creates a plot with all the points of individual until 2 years old.
    plot1 : plot
        This creates a plot with all the points of individual 2 until 6 years old.
    plot2 : plot
        This creates a plot with all the points of individual 6 until 12 years old.
        All plots are shown together
    """
    for measure in dataframe:
        #create a new scatter plot where the three age groups are shown in different colours
        plt.figure(figsize=(10,8))
        baby = dataframe.loc[dataframe['AgeRange'] ==  '<2']
        toddler = dataframe.loc[dataframe['AgeRange'] == '2-6']
        child = dataframe.loc[dataframe['AgeRange'] == '6-12']
        plot = plt.scatter(baby["ArchivalAge"],baby[measure], s=100,
                           c="#1b9e77", marker="^")
        plot1 = plt.scatter(toddler["ArchivalAge"],toddler[measure], 
                            s=100, c="#d95f02", marker="*")
        plot2 = plt.scatter(child["ArchivalAge"],child[measure], 
                            s=100, c="#7570b3", marker="d")
        plt.title("Scatterplot of " + measure + " and age", fontsize=14)
        plt.xlabel("Age-at-death (years)", fontsize=14)
        plt.xticks(fontsize=14) 
        plt.ylabel("bone measurements (mm)", fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()
    return plot, plot1, plot2

plot(RegressionBase)
    #every column will get their own plot with all ages in, to establish the linear connection

Below2 = RegressionBase.loc[RegressionBase['AgeRange'] ==  '<2']
TwotillSix = RegressionBase.loc[RegressionBase['AgeRange'] == '2-6']
SixtillTwelve = RegressionBase.loc[RegressionBase['AgeRange'] == '6-12']
TwotillTwelve = pd.concat([TwotillSix, SixtillTwelve])

    
def rangecount(dataframe):
    """
    Function creating a dataframe indicating the ranges of age and measures for
    each individual measure method

    Parameters
    ----------
    dataframe : DataFrame
        The name of the dataframe which you want to calculate the age and measure
        ranges from.

    Returns
    -------
    DataFrame
        A Dataframe which shows the name of measure method, the minimum of measure,
        the maximum of the measures, the minimum age of the individual, 
        and the maximum of the individuals. For each measure method

    """
    Range = {}
    for measure in dataframe:
        #create a Dictionary where the main important descriptives are shown 
            #per column
        rangeset = dataframe[[measure, "ArchivalAge"]].dropna()
        measureCount = rangeset[measure].count() #the count
        minmeasure = rangeset[measure].min() #the minimum measurement
        maxmeasure = rangeset[measure].max()
        measurerange = "{} - {}".format(minmeasure, maxmeasure)
        minage = rangeset["ArchivalAge"].min() #the minimum measurement
        maxage = rangeset["ArchivalAge"].max() #the maximum measurement
        Range[measure] = {
            "Count": measureCount,
            "Minimum measure": minmeasure,
            "Maximum measure": maxmeasure,
            "range": measurerange,
            "Minimum age": minage,
            "Maximum age": maxage,
            }
    return pd.DataFrame(Range) 

Below2Range = rangecount(Below2)
TwotillSixRange = rangecount(TwotillSix)
SixtillTwelveRange = rangecount(SixtillTwelve)
TwotillTwelveRange = rangecount(TwotillTwelve)

# starting with the below 2 group: 
def SEE(dataset, x, y, b0, b1):
    '''
    A function which estimates the Standard Estimation of Error (SEE) for a
    single regression formula 
    
    Parameters
    ----------
    dataset : DataFrame
        The dataset from which the regression formula was created
    x : string
        The column which the independent value is from
    y : string
        The column which the dependent value is from.
    b0 : float
        The coefficient of the intercept.
    b1 : float
        The coefficient of the slope of the corresponding independent value.

    Returns
    -------
    Float value that shows the Standard Estimation of Error.

    '''
    #this is the one for one variable (x)
    yhat = b0 + dataset[x]*b1
        #this creates the y^ of the dataframe based on the slope and intercept
    epsilon = dataset[y] - yhat
        #this creates the epsilon, which when there are no epsilon 
        #value's creates NaN values
    epsilon = epsilon.dropna()
    eps2 = epsilon*epsilon
    SEE = (sum(eps2)/(len(epsilon)-2))**(1/2)
        #this creates the Standard Error of Estimate for the specified ols
    return(SEE)


#now we do all regression models for individuals under 2
    #to show F9 the entire 5 lines
x = Below2["Frontal Height"]
y = Below2["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = Below2["Frontal Width"]
y = Below2["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = Below2["Occipital Squamous Length"]
y = Below2["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = Below2["Occipital Squamous Width"]
y = Below2["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = Below2["Occipital Basilar Length"]
y = Below2["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = Below2["Occipital Basilar Width"]
y = Below2["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = Below2["Occipital Basilar Max Length"]
y = Below2["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = Below2["Occipital Lateral Length"]
y = Below2["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = Below2["Occipital Lateral Width"]
y = Below2["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = Below2["Zygomatic Length"]
y = Below2["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = Below2["Zygomatic Width"]
y = Below2["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = Below2["Maxilla Length"]
y = Below2["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = Below2["Maxilla Width"]
y = Below2["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = Below2["Maxilla Height"]
y = Below2["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = Below2["Mandible Chin Height"]
y = Below2["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = Below2["Mandible Bigonial Width"]
y = Below2["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

# x = Below2["Mandible Bicondylar Breadth"]
    #this version has no measurements below 2

x = Below2["Mandible Ramus Height"]
y = Below2["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = Below2["Mandible Max Length"]
y = Below2["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

#calculating the SEE of each Regression: 
SEE(Below2, "Frontal Height", "ArchivalAge", -2.4265, 0.0478)
SEE(Below2, "Frontal Width", "ArchivalAge", -1.8930, 0.0397)
SEE(Below2, "Occipital Squamous Length", "ArchivalAge", -5.0430, 0.0948)
SEE(Below2, "Occipital Squamous Width", "ArchivalAge", -4.5269 , 0.0776)
SEE(Below2, "Occipital Basilar Length", "ArchivalAge", -2.7705, 0.2237 )
SEE(Below2, "Occipital Basilar Width", "ArchivalAge", -2.2483, 0.1532 )
SEE(Below2, "Occipital Basilar Max Length", "ArchivalAge", -2.8944, 0.1773 )
SEE(Below2, "Occipital Lateral Length", "ArchivalAge", -1.0873, 0.0439)
SEE(Below2, "Occipital Lateral Width", "ArchivalAge", -1.4505, 0.1014)
SEE(Below2, "Zygomatic Length", "ArchivalAge", -2.4606, 0.1035)
SEE(Below2, "Zygomatic Width", "ArchivalAge", -2.7226, 0.1582 )
SEE(Below2, "Maxilla Length", "ArchivalAge", -6.2837, 0.2748)
SEE(Below2, "Maxilla Width", "ArchivalAge", -4.3488 , 0.1752 )
SEE(Below2, "Maxilla Height", "ArchivalAge", -1.4457, 0.0681)
SEE(Below2, "Mandible Chin Height", "ArchivalAge", -1.6166 , 0.158)
SEE(Below2, "Mandible Bigonial Width", "ArchivalAge", -6.2031, 0.1140 )
SEE(Below2, "Mandible Ramus Height", "ArchivalAge", -1.4646 , 0.0879)
SEE(Below2, "Mandible Max Length", "ArchivalAge", -3.0289, 0.0824)


#We continue with the group 2-6; same as before
x = TwotillSix["Frontal Height"]
y = TwotillSix["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

# x = TwotillSix["Frontal Width"]
    #there are no 2-6 measures for this mearure

x = TwotillSix["Occipital Squamous Length"]
y = TwotillSix["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillSix["Occipital Squamous Width"]
y = TwotillSix["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillSix["Occipital Basilar Length"]
y = TwotillSix["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillSix["Occipital Basilar Width"]
y = TwotillSix["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillSix["Occipital Basilar Max Length"]
y = TwotillSix["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

#x = TwotillSix["Occipital Lateral Length"]
    #there are no 2-6 measures for this mearure
    
x = TwotillSix["Occipital Lateral Width"]
y = TwotillSix["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillSix["Zygomatic Length"]
y = TwotillSix["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillSix["Zygomatic Width"]
y = TwotillSix["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillSix["Maxilla Length"]
y = TwotillSix["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillSix["Maxilla Width"]
y = TwotillSix["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillSix["Maxilla Height"]
y = TwotillSix["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillSix["Mandible Chin Height"]
y = TwotillSix["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillSix["Mandible Bigonial Width"]
y = TwotillSix["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillSix["Mandible Bicondylar Breadth"]
y = TwotillSix["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillSix["Mandible Ramus Height"]
y = TwotillSix["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillSix["Mandible Max Length"]
y = TwotillSix["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

#calculating the SEE of these 
SEE(TwotillSix, "Frontal Height", "ArchivalAge", 4.1116, -0.0046)
SEE(TwotillSix, "Occipital Squamous Length", "ArchivalAge", 13.6548, -0.1084)
SEE(TwotillSix, "Occipital Squamous Width", "ArchivalAge", - 1.9100 , 0.0568)
SEE(TwotillSix, "Occipital Basilar Length", "ArchivalAge", 0.0458, 0.2130)
SEE(TwotillSix, "Occipital Basilar Width", "ArchivalAge", 2.3534 , 0.0562)
SEE(TwotillSix, "Occipital Basilar Max Length", "ArchivalAge", 0.04 , 0.1469)
SEE(TwotillSix, "Occipital Lateral Width", "ArchivalAge", - 4.6322, 0.2882)
SEE(TwotillSix, "Zygomatic Length", "ArchivalAge", 3.8597 , 0.0019 )
SEE(TwotillSix, "Zygomatic Width", "ArchivalAge", 4.5894 , -0.0283)
SEE(TwotillSix, "Maxilla Length", "ArchivalAge", - 1.2559 , 0.1663)
SEE(TwotillSix, "Maxilla Width", "ArchivalAge", - 3.4182 , 0.2026)
SEE(TwotillSix, "Maxilla Height", "ArchivalAge", 0.2605 , 0.0742 )
SEE(TwotillSix, "Mandible Chin Height", "ArchivalAge", 1.4323, 0.1116 )
SEE(TwotillSix, "Mandible Bigonial Width", "ArchivalAge", - 4.1476 , 0.097)
SEE(TwotillSix, "Mandible Bicondylar Breadth", "ArchivalAge", - 7.7738, 0.1252 )
SEE(TwotillSix, "Mandible Ramus Height", "ArchivalAge", - 2.5744 , 0.1792)
SEE(TwotillSix, "Mandible Max Length", "ArchivalAge", - 6.5759, 0.1718)


#when the measures of 2-6 and 2-12 are combined
x = TwotillTwelve["Frontal Height"]
y = TwotillTwelve["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

#x = TwotillTwelve["Frontal Width"]
    #there are no 2-12 measures for this measure  

x = TwotillTwelve["Occipital Squamous Length"]
y = TwotillTwelve["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillTwelve["Occipital Squamous Width"]
y = TwotillTwelve["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillTwelve["Occipital Basilar Length"]
y = TwotillTwelve["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

# x = TwotillTwelve["Occipital Basilar Width"]
    #there are no 6-12 measures for this measure
        #will be the same as the 2-6 estimates
# x = TwotillTwelve["Occipital Basilar Max Length"]
    #there are no 6-12 measures for this measure
# x = TwotillTwelve["Occipital Lateral Length"]
    #there are no 2-12 measures for this measure
        #will be the same as the 2-6 estimates

x = TwotillTwelve["Occipital Lateral Width"]
y = TwotillTwelve["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillTwelve["Zygomatic Length"]
y = TwotillTwelve["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillTwelve["Zygomatic Width"]
y = TwotillTwelve["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillTwelve["Maxilla Length"]
y = TwotillTwelve["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillTwelve["Maxilla Width"]
y = TwotillTwelve["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillTwelve["Maxilla Height"]
y = TwotillTwelve["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillTwelve["Mandible Chin Height"]
y = TwotillTwelve["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillTwelve["Mandible Bigonial Width"]
y = TwotillTwelve["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillTwelve["Mandible Bicondylar Breadth"]
y = TwotillTwelve["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillTwelve["Mandible Ramus Height"]
y = TwotillTwelve["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()

x = TwotillTwelve["Mandible Max Length"]
y = TwotillTwelve["ArchivalAge"]
x_sm = sm.add_constant(x)
mod_sm = sm.OLS(y,x_sm, missing="drop").fit()
mod_sm.summary()


#Calculating the SEE of these regressions:
SEE(TwotillTwelve, "Frontal Height", "ArchivalAge",1.5761 , 0.0484)
SEE(TwotillTwelve, "Occipital Squamous Length", "ArchivalAge",-1.1875 , 0.0808)
SEE(TwotillTwelve, "Occipital Squamous Width", "ArchivalAge", -24.5738, 0.2898)
SEE(TwotillTwelve, "Occipital Basilar Length", "ArchivalAge", -12.5395, 0.9950)
SEE(TwotillTwelve, "Occipital Lateral Width", "ArchivalAge", -7.6041, 0.4752)
SEE(TwotillTwelve, "Zygomatic Length", "ArchivalAge", -1.0224, 0.1706)
SEE(TwotillTwelve, "Zygomatic Width", "ArchivalAge",5.8815 , -0.0013)
SEE(TwotillTwelve, "Maxilla Length", "ArchivalAge",-18.3485 , 0.7627)
SEE(TwotillTwelve, "Maxilla Width", "ArchivalAge",-25.4851 , 0.8452)
SEE(TwotillTwelve, "Maxilla Height", "ArchivalAge",-13.9913 , 0.3983 )
SEE(TwotillTwelve, "Mandible Chin Height", "ArchivalAge", -15.9512, 0.9777)
SEE(TwotillTwelve, "Mandible Bigonial Width", "ArchivalAge",  -18.0809, 0.3183)
SEE(TwotillTwelve, "Mandible Bicondylar Breadth", "ArchivalAge",-14.9201 , 0.2174 )
SEE(TwotillTwelve, "Mandible Ramus Height", "ArchivalAge",-10.6612 , 0.4413)
SEE(TwotillTwelve, "Mandible Max Length", "ArchivalAge", -22.0700, 0.4390)

# =============================================================================
# Scatterplot which are possibly useful for the results chapter
    #visualising the spread and the actual regression formulae 
        #for several differen measurements
        
plt.figure(figsize=(10,8)) #added this this time for a visually appealing spread of the scatterpltos
plot = plt.scatter(Below2["Frontal Height"], Below2["ArchivalAge"],s=100,
                   c="#1b9e77", marker="^")
plot1 = plt.scatter(SixtillTwelve["Frontal Height"],SixtillTwelve["ArchivalAge"], 
                    s=100, c="#d95f02", marker="*")
plot2 = plt.scatter(TwotillSix["Frontal Height"],TwotillSix["ArchivalAge"], 
                    s=100, c="#7570b3", marker="d")
plt.plot(np.arange(40, 100), 0.0478*np.arange(40, 100)-2.4265, c="#1b9e77")
plt.plot(np.arange(90, 110), 0.0436*np.arange(90, 110)+4.6876, c="#d95f02")
plt.title("Scatterplot of Frontal Height and age", fontsize=14)
plt.xlabel("bone measurements (mm)", fontsize=14)
plt.xticks(fontsize=14) 
plt.ylabel("Age-at-death (years)", fontsize=14)
plt.yticks(fontsize=14)
plt.show()


plt.figure(figsize=(10,8)) #added this this time for a visually appealing spread of the scatterpltos
plot = plt.scatter(Below2["Zygomatic Width"], Below2["ArchivalAge"],s=100,
                   c="#1b9e77", marker="^")
plot1 = plt.scatter(SixtillTwelve["Zygomatic Width"],SixtillTwelve["ArchivalAge"], 
                    s=100, c="#d95f02", marker="*")
plot2 = plt.scatter(TwotillSix["Zygomatic Width"],TwotillSix["ArchivalAge"], 
                    s=100, c="#7570b3", marker="d")
plt.plot(np.arange(12,30), 0.1582*np.arange(12,30)-2.7226, c="#1b9e77")
plt.plot(np.arange(24,34), -0.3380*np.arange(24, 34)+18.9375, c="#d95f02")
plt.title("Scatterplot of Zygomatic Width and age", fontsize=14)
plt.xlabel("bone measurements (mm)", fontsize=14)
plt.xticks(fontsize=14) 
plt.ylabel("Age-at-death (years)", fontsize=14)
plt.yticks(fontsize=14)
plt.show()

plt.figure(figsize=(10,8)) #added this this time for a visually appealing spread of the scatterpltos
plot = plt.scatter(Below2["Maxilla Length"], Below2["ArchivalAge"],s=100,
                   c="#1b9e77", marker="^")
plot1 = plt.scatter(SixtillTwelve["Maxilla Length"],SixtillTwelve["ArchivalAge"], 
                    s=100, c="#d95f02", marker="*")
plot2 = plt.scatter(TwotillSix["Maxilla Length"],TwotillSix["ArchivalAge"], 
                    s=100, c="#7570b3", marker="d")
plt.plot(np.arange(25,40), 0.7627*np.arange(25,40)-18.0809, c="#7570b3")
plt.plot(np.arange(25,40), 0.2644*np.arange(25,40)+0.0808, c="#d95f02")
plt.title("Scatterplot of Maxilla Length and age", fontsize=14)
plt.xlabel("bone measurements (mm)", fontsize=14)
plt.xticks(fontsize=14) 
plt.ylabel("Age-at-death (years)", fontsize=14)
plt.yticks(fontsize=14)
plt.show()


plt.figure(figsize=(10,8)) #added this this time for a visually appealing spread of the scatterpltos
plot = plt.scatter(Below2["Occipital Lateral Width"], Below2["ArchivalAge"],s=100,
                   c="#1b9e77", marker="^")
plot1 = plt.scatter(SixtillTwelve["Occipital Lateral Width"],SixtillTwelve["ArchivalAge"], 
                    s=100, c="#d95f02", marker="*")
plot2 = plt.scatter(TwotillSix["Occipital Lateral Width"],TwotillSix["ArchivalAge"], 
                   s=100, c="#7570b3", marker="d")
plt.plot(np.arange(10,40), 0.1014*np.arange(10,40) - 1.4505, c="#1b9e77")
# plt.plot(np.arange(10,40), 0.4752*np.arange(10,40) - 7.6041, c="#7570b3")
plt.title("Scatterplot of Occipital Lateral Width and age", fontsize=14)
plt.xlabel("Occipital lateral width (mm)", fontsize=14)
plt.xticks(fontsize=14) 
plt.ylabel("Age-at-death (years)", fontsize=14)
plt.yticks(fontsize=14)
plt.show()


