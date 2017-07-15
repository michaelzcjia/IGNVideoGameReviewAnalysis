#Process data into dataframe
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
sys.path.append('../Data')

#Predict the highest rated game
class GamesDataFrame():

    def __init__ (self, file_loc = '../data/ign.csv'):
        '''initialize dataframe from csv'''

        rawData = pd.read_csv('../data/ign.csv')
        rawData.drop(['Unnamed: 0', 'url'], axis=1, inplace=True)  # Gets rid of the url and unnamed columns
        rawData.replace(to_replace='Y', value=1, inplace=True)
        rawData.replace(to_replace='N', value=0, inplace=True)
        self.dataFrame = rawData[rawData['release_year'].values != 1970] # Gets rid of the row with the outlier year

    def __str__(self):
        return str(self.dataFrame)

    def getColumn(self, columnName = 'score_phrase'):
        return self.dataFrame.loc[:,columnName]

    def plotAverageScorePerYear(self):
        score = self.getColumn("score")
        year = self.getColumn("release_year")
        scoreList = list(score)
        yearList = list(year)
        meanScores = []
        correspondingYears = sorted(list(set(yearList)))

        for year in range(1996, 2017):
            indices = [i for i, x in enumerate(yearList) if x == year] #Gets all the valid indices for the year
            meanScoreList = []
            #Appends all of the scores for that year to a list
            for index in indices:
                meanScoreList.append(scoreList[index])
            #Appends the mean score of that year to a list
            meanScores.append(np.mean(np.array(meanScoreList)))
        plt.title("Average Game Score by Year")
        plt.xlabel("Year")
        plt.ylabel("Average Game Score")
        plt.xticks(correspondingYears)
        plt.plot(correspondingYears, meanScores)
        plt.plot(correspondingYears, meanScores, 'r*')
        plt.show()

    def plotScorePerGenreYear(self):
        plt.subplots(figsize=(19, 15))
        dataByGenre = self.dataFrame.groupby('genre')['genre'].count() #Makes a groupby object that groups by genre count
        dataByGenre = dataByGenre[dataByGenre.values > 200] #Eliminates rows with not enough data
        dataByGenre.sort_values(ascending=True, inplace=True)

        # Only takes the rows where the relevent genres are in the column "genre"
        dataFilteredGenres = self.dataFrame[self.dataFrame['genre'].isin(dataByGenre.index)]

        # Gets the mean score for each genre for each year, the resets the groupby indexing
        meanGroupedData = dataFilteredGenres.groupby(["release_year", 'genre'])['score'].mean().reset_index()
        meanGroupedDataRect = meanGroupedData.pivot(index = 'release_year', columns = 'genre', values = 'score')

        sns.heatmap(meanGroupedDataRect, annot=True, cmap='RdYlGn', linewidths=0.2)
        plt.xticks(rotation = 90)

        plt.title('Average Score by Genre and Release Year')
        plt.show()

    def plotAverageScoreStandardDeviationGenre(self):
        sns.set_style("whitegrid", {'axes.grid': False})
        dataByGenre = self.dataFrame.groupby('genre')['genre'].count() #Makes a groupby object that groups by genre count
        dataByGenre = dataByGenre[dataByGenre.values > 50] #Eliminates rows with not enough data
        dataByGenre.sort_values(ascending=True, inplace=True)

        # Only takes the rows where the relevent genres are in the column "genre"
        dataFilteredGenres = self.dataFrame[self.dataFrame['genre'].isin(dataByGenre.index)]

        #Creating the mean score by genre dataframe and sorting it
        meanGroupedData = dataFilteredGenres.groupby('genre')['score'].mean().reset_index()
        meanGroupedData.sort_values(['score'], axis=0, ascending=True, inplace=True)
        print(meanGroupedData)

        # Creating the mean score by genre dataframe
        stdGroupedData = dataFilteredGenres.groupby('genre')['score'].std().reset_index()

        genreList = list(meanGroupedData['genre'])
        meanList = list(meanGroupedData['score'])
        stdList = list(stdGroupedData['score'])


        fig, ax1 = plt.subplots()
        position = np.arange(len(genreList))
        #print(position)
        ax1.bar(position, meanList, color = 'g', width = 0.3, tick_label= genreList )
        ax1.set_xlabel('Genre')

        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel('Average', color='g')
        ax1.tick_params('y', colors='g')

        plt.xticks(rotation=90)
        plt.ylim(5,8)

        ax2 = ax1.twinx()
        ax2.bar(np.array(position)+0.3, stdList, color = 'b', width = 0.3, tick_label = None )
        ax2.set_ylabel('Standard Deviation', color='b')
        ax2.tick_params('y', colors='b')

        plt.ylim(1.0, 2.0)
        plt.title("Average Score and Standard Deviation by Genre")
       # fig.tight_layout()
        plt.show()

    def getOneHotFeaturesLabels(self, trainingPrct = 0.7):
        #One hot encoding
        dataOneHot = pd.get_dummies(self.dataFrame, columns = ['genre', 'platform', 'release_year'])
        dataOneHot = dataOneHot.drop(labels = ["score_phrase", "title", "release_month", 'release_day'], axis = 1)
        shuffle(dataOneHot)

        #Split into training ang testing data
        prctNum = int(np.floor(trainingPrct * len(dataOneHot)))
        training = dataOneHot.iloc[:prctNum]
        testing = dataOneHot.iloc[prctNum:]

        trainLabels = training["score"]
        testLabels = testing["score"]
        trainFeatures = training.drop("score", axis = 1)
        testFeatures = testing.drop("score", axis=1)

        return [(trainFeatures, trainLabels), (testFeatures, testLabels)]

    def plotAvgScoreVsReleaseQuantity(self):

        fig, ax1 = plt.subplots(figsize=(13, 8))
        dataAverageScore = self.dataFrame.groupby('release_year')['score'].mean()
        ax1.plot(dataAverageScore, label = "Average Score", color = "r")
        ax1.tick_params('y', colors='r')
        plt.xticks(list(set(self.dataFrame["release_year"])))
        plt.ylabel("Average Score")
        plt.legend()

        ax2 = ax1.twinx()
        dataReleaseYear = self.dataFrame.groupby('release_year')['release_year'].count()  # Makes a groupby object that groups by genre count
        ax2.plot(dataReleaseYear, label = "Number of Releases")
        ax2.tick_params('y', colors='b')
        plt.ylabel("Number of Releases")

        plt.xlabel("Year")
        plt.legend()
        plt.show()



    # INVERSE RELATIONSHIP BETWEEN SCORE AND # OF RELEASES

if __name__ == "__main__":
    df = GamesDataFrame()
    # df.plotAverageScorePerYear()
    # df.plotScorePerGenreYear()
    # df.plotAverageScoreStandardDeviationGenre()
    df.plotAvgScoreVsReleaseQuantity()
    print(df.dataFrame)

#print(list(x))
#INVERSE RELATIONSHIP BETWEEN SCORE AND # OF RELEASES
#Plot average score vs number of releases
#print(rawData.head())#Checks the first few rows of the dataframe
#print(rawData.isnull().sum())   #Finds the null values
#d = dict(zip(list1, list2)) dictionary of 2 lists
""

