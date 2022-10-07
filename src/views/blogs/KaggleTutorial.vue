<template>
    <n-space vertical>
        <n-h1 style="color: #e4572e; font-size: 40px;">Kaggle Titanic Tutorial</n-h1>
        <n-grid cols="1" responsive="screen" :x-gap="100">
            <n-gi>
                <n-space>
                    <n-h2 style="color: white">
                        I followed through a tutorial by Alexis Cook that went through 
                        the process of completing in a Kaggle competition. The practice 
                        competition uses a passenger dataset from the Titanic. The dataset 
                        specifies a passengers attributes such as age, sex, ticket class, 
                        etc. The dataset also indicates if they survived or not. The challenge 
                        is to create a prediction model that predicts whether a given passenger 
                        would survive. 
                    </n-h2>
                    <n-h2 style="color: white">Going into the directory with the training data,
                        testing data and example prediction .csv files.</n-h2>
                    <n-image src="assets/KaggleTutorialSnip1.png" width="800" />
                    <n-h2 style="color: white">Assigning training data and testing data. Also
                    displaying the 1st five passengers with their respective data for each dataset.</n-h2>
                    <n-image src="assets/KaggleTutorialSnip2.png" width="800" />
                    <n-h2 style="color: white">Testing if gender_submission.csv prediction of all
                        female passengers survived and all male passengers died. Since the numbers are
                        74% women and 18% men we can say that this prediction is significantly inaccurate.
                    </n-h2>
                    <n-image src="assets/KaggleTutorialSnip3.png" width="800" />
                    <n-h2 style="color: white">Using the Random Tree Model to make predictions based
                    on a given passenger's data.</n-h2>
                    <n-image src="assets/KaggleTutorialSnip4.png" width="800" />
                    <n-h2 style="color: white">
                        After following the tutorial from Alexis Cook we built a prediction model
                        that uses a random forest model. A random forest model produces trees that
                        take each attribute of the passenger's data into account. The model uses
                        a majority vote with the outcome of the trees to predict if the passenger
                        survived. The model we built only accounted for the passenger's class,
                        sex, amount of siblings and spouses, amount of parents and children. This
                        model produced a score of .77511 accuracy.
                    </n-h2>
                    <n-image src="assets/KaggleTutorialSnip5.png" width="800" />
                </n-space>
            </n-gi>
        </n-grid>
    </n-space>
</template>

<style>

</style>