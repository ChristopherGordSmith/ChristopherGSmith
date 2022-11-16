<template>
    <n-space vertical>
        <n-grid cols="1" responsive="screen" :x-gap="100">
            <n-gi>
                <n-space>
                    <n-h1 style="color: #e4572e">Motivation</n-h1>
                    <n-h2 style="color: white">
                        For my Data Mining class we were tasked with building a Naive Bayes Classifier using the dataset from the 

                        <a style="color: #e07a5f" target="_blank"
                            href="https://www.kaggle.com/datasets/gaveshjain/ford-sentence-classifiaction-dataset">Kaggle
                            hosted by Ford</a>. I am using the concepts I learned in class and applying them to build a model for this
                            specific dataset.
                    </n-h2>
                    <n-h1 style="color: #e4572e">Objective</n-h1>
                    <n-h2 style="color: white">
                        The object of this blog is to explain the concepts behind Naïve Bayes, walk through my Naïve Bayes Classifier model,
                        show the results of my model, and walk through some pros and cons of using a Naïve Bayes model. I will also explain any
                        obstacles during this process and how I overcame them as well as any shortcomings in the objective.
                    </n-h2>
                    <n-h1 style="color: #e4572e">Naive Bayes</n-h1>
                    <n-h2 style="color: white">
                        Naïve Bayes is an algorithm that produces probabilities, in my case classes for a dataset, based on the data given.
                        After training a Naive Bayes classifier one simply needs prompt the model with an unknown class that has a set of
                        features. Then it will return the probability of it being a given class. If a binary classification is being used one
                        can assume if the probability of a given class is lower than 50 percent, it is probably the other class. In the case of
                        using multiple classes, you would have to test for each class and choose the highest probability. Naïve Bayes is broken
                        down into two main concepts Bayesian Theorem and Naïve assumption.
                    </n-h2>
                    <n-h1 style="color: #e4572e">Bayesian Theorem</n-h1>
                    <n-h2 style="color: white">
                        Before going into Bayesian Theorem, I would like to explain the difference between a Frequentest and a Bayest. A
                        Frequentest and a Bayest consider the outcomes of past events to make a prediction of possible outcomes of an event.
                        Where these two differ is how they weigh past events that affect their predictions. A Frequentest will take the outcomes
                        of past events and directly applies them to their prediction. In other words, a Frequentest will weigh outcomes of past
                        events heavily and possibly ignores what’s realistic. A Bayest will take the outcomes of past events and apply a small
                        weight on each event that will slightly alter the prediction based on past events. This will be more transparent with an
                        example.
                    </n-h2>
                    <n-h2 style="color: white">
                        Let’s say we have a coin, and we flip it 20 times and it lands on tails 15 times. What’s the probability that the next
                        coin flip will result in tails? Well, if a Frequentest were to guess they would say the probability of tails given a
                        fair coin would be a 75 percent. A Bayest would say the probability of tails given a fair coin would be 50 percent. The
                        Bayest would then look at the outcomes of past events and apply Bayesian Theorem to come up with a probability.
                    </n-h2>
                    <n-h2 style="color: white">
                        Bayesian Theorem is the formula that Naïve Bayes uses to produce a probability for a given class based on the data of
                        past events and their outcomes. It uses a Hypothesis (H) and Evidence (E) to made a prediction. Hypothesis is what 
                        the algorithm is querying for and Evidence is prior knowledge given.
                        <n-image style="margin-bottom: 0px;margin-left: 30%; margin-right: 30%" src="assets/BayesTheorem.jpg" width="500" />
                        <n-h5 style="margin-top: 0px; color: white; margin-right:45%; margin-left:45%">Bayes formula</n-h5>
                        This formula can be broken down into P(H and E)/P(E), where P(H and E) is when both H and E are true.
                        P(H and E) is the joint probability and P(E) is the Margin probability.
                        <n-image style="margin-left: 30%; margin-right: 30%" src="assets/BayesTheoremEvidence.jpg" width="500" />
                        <n-h5 style="margin-top: 0px; color: white; margin-right:45%; margin-left:45%">Marginalization</n-h5>
                        P(E) can be broken down into the summation of the products of the probability of the Hypothesis with the Evidence
                        given the Hypothesis and the opposite of the Hypothesis with Evidence given the opposite of the Hypothesis.
                    </n-h2>
                    <n-h1 style="color: #e4572e">Naive Assumption</n-h1>
                    <n-h2 style="color: white">
                        The Naive part of the Naive Bayes algorithm is the Naive Assumption this algorithm makes about the data given.
                        The algorithm assumes conditional independence between each feature of the dataset. This means each attribute of
                        a given row of data will not affect the outcome of the other features of the same given row of data. This does
                        mean that if the given data's features are dependent on one another the algorithm will fail during the prediction process.
                    </n-h2>
                    <n-h1 style="color: #e4572e">Source code</n-h1>
                    <n-h2 style="color: white">
                        In the code below I setup two classes FeedBack and Diction. FeedBack would represent each class
                        and would hold an array of words (Dictions) the FeedBack class contained. I then made each Diction
                        reside in the Feedback array of a given class. Diction would contain the word and the occurrence of
                        that word for the given class. This was so I could make a table like figure making coming up with 
                        probabilities easier.
                        <n-image style="margin-left: 30%; margin-right: 30%" src="assets/NaiveClasses.png" width="500" />
                        <n-h5 style="margin-top: 0px; color: white; margin-right:45%; margin-left:45%">Document Classes</n-h5>
                        After setting up the classes I then took the data from the Kaggle challenge and had it separated into
                        training, development, and testing datasets. They were split 60 percent : 20 percent : 20 percent
                        respectively. I did have to throw away the data that had absent sentences so I could use the data properly.
                        <n-image style="margin-left: 30%; margin-right: 30%" src="assets/NaiveDataSetup.png" width="500" />
                        <n-h5 style="margin-top: 0px; color: white; margin-right:45%; margin-left:45%">Naive Data Setup</n-h5>
                        I setup a Dictionary of all the words utilized in the dataset from the Kaggle challenge. I used the python library
                        're' to split the sentences by word. I did have to
                        learn Regex from
                        <a style="color: #e07a5f" target="_blank"
                            href="https://learn.microsoft.com/en-us/dotnet/standard/base-types/regular-expression-language-quick-reference">Microsoft Learn</a>
                        so I could gather as many words without special characters as possible while using re.split(). Also I made every word lower
                        case so the I could include words at the start of the sentences. After words any words that were below 5 occurrences
                        were dropped from the dictionary.
                        <n-image style="margin-left: 30%; margin-right: 30%" src="assets/DictionaryInitialize.png" width="500" />
                        <n-h5 style="margin-top: 0px; color: white; margin-right:45%; margin-left:45%">Dictionary Initialization</n-h5>
                        I then used the FeedBack class to get my classes for the dataset. I would store them all in results with each node
                        being a different class it would find when exploring the dataset. If the class already was known it would just add
                        data to the node's array. Additionally, if the word was new to the node's array then I would initialize a new Diction
                        and append it to the end. I also used the 're' python library to split my data and applied the same Regex expression
                        used on the Dictionary.
                        <n-image style="margin-left: 30%; margin-right: 30%" src="assets/ResultsInitialize.png" width="500" />
                        <n-h5 style="margin-top: 0px; color: white; margin-right:45%; margin-left:45%">Results Initialization</n-h5>
                        At this point I was ready to try calculating a probability using my formed dataset. I calculated the probability of
                        getting the word "the" out of the entire dataset which just was simply the occurrences of the word "the" divided by
                        the total words in the dataset. The occurrence of "the" was 8095 and the probability of getting "the" was 1.8 percent.
                        <n-image style="margin-left: 30%; margin-right: 30%" src="assets/theProbs.png" width="500" />
                        <n-h5 style="margin-top: 0px; color: white; margin-right:45%; margin-left:45%">"the" probability</n-h5>
                        After that I decided to try using Naive Bayes to come up with a probability of getting "the" given that
                        the class was "Requirement". I broke up the math and printed out the result of each operation. Starting with P(Hypothesis),
                        P(Evidence|Hypothesis), P(Evidence|notHypothesis), then P(Evidence) my margin. I then applied the Bayes Theorem formula
                        and came out 3.3 percent chance. This means my chances of getting "the" with the prior knowledge of the word coming
                        from the class "Requirement" was pretty low. 
                        <n-image style="margin-left: 30%; margin-right: 30%" src="assets/theGivenReq.png" width="500" />
                        <n-h5 style="margin-top: 0px; color: white; margin-right:45%; margin-left:45%">"the" given requirement probability
                        </n-h5>
                        I then designed a function using the same logic of the code above that could take in any word and class dynamically.
                        This would allow for readability and a nicer look to the function. I could also now call this function multiple times
                        and save time.
                        <n-image style="margin-left: 30%; margin-right: 30%" src="assets/probsFunction.png" width="500" />
                        <n-h5 style="margin-top: 0px; color: white; margin-right:45%; margin-left:45%">probs Function</n-h5>
                        In the code below I sought out to get the top 10 likely words chosen for each class in my dataset using the training
                        dataset. I would loop through each word in the dictionary and use my probs() function to return the probability
                        of getting that word for the given class. If the word didn't occur in the dataset I then just returned 0.0 probability.
                        The result would then be stored with the word being checked in a node in each classes respective list.
                        After getting all the words and their probabilities for each given class. I sorted each class of words in ascending
                        order and took the first 10 words.
                        <n-image style="margin-left: 30%; margin-right: 30%" src="assets/MakingTop10.png" width="500" />
                        <n-h5 style="margin-top: 0px; color: white; margin-right:45%; margin-left:45%">Top 10 setup</n-h5>
                        I below is a result of each class's top 10 words and their probabilities given their respective class.
                        This information is holds value as words such as: 'knowledge', 'communication', and 'degree' are present
                        in the results. Which means these words are most likely what recruiters would be looking for on a resume
                        for the given class. However, it is made apparent in the results words such as: "a", "and", "of" are a waste
                        of information as it logically doesn't contribute to the prediction.
                        <n-image style="margin-left: 30%; margin-right: 30%" src="assets/top10words.png" width="500" />
                        <n-h5 style="margin-top: 0px; color: white; margin-right:45%; margin-left:45%">Top 10 results</n-h5>
                        This is the results of the code I wrote for this challenge. To conclude I wasn't able to get as far as I
                        would like with my code, but I was able to come up with a function that could be used generically to gather
                        probabilities.
                    </n-h2>
                    <n-h1 style="color: #e4572e">Pros & Cons for Naive Bayes</n-h1>s
                    <n-h2 style="font-weight: 700;">
                       <n-h2 style="color: white">Below are the pros and cons of using the Naive Bayes approach which I learned in my classes.</n-h2>
                        <n-table :bordered="false" :single-line="false" style="margin-left: 20%; margin-right: 20%">
                            <thead>
                                <tr>
                                    <th>Pros</th>
                                    <th>Cons</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>The algorithm is relatively simple to implement for a model
                                    </td>
                                    <td>Fails when features are not independent
                                    </td>
                                </tr>
                                <tr>
                                    <td>Training time is fast O(n)
                                    </td>
                                    <td>Fails when data is missing (Can be fixed with smoothing or dropping the feature)
                                    </td>
                                </tr>
                                <tr>
                                    <td>Testing time is fast O(1)
                                    </td>
                                    <td>
                                    </td>
                                </tr>
                            </tbody>
                        </n-table>
                    </n-h2>
                    <h2>
                    <n-h1 style="color: #e4572e">Contributions</n-h1>
                        <n-table :bordered="false" :single-line="false">
                            <thead>
                                <tr>
                                    <th>Contribution</th>
                                    <th>The value of the contribution</th>
                                    <th>Technical Challenges</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Used regex to take off unneeded special characters</td>
                                    <td>Allowed the data to be used for raw words that didn't have any extra unneeded information</td>
                                    <td>Had to allow sum words to be missed since the the special characters such as single quotes needed to remain.
                                        I also had to learn regex and how to utilize it for the splitting.
                                    </td>
                                </tr>
                                <tr>
                                    <td>Made a dictionary and class table of the given words</td>
                                    <td>I made a dictionary of all the words used with their occurrences and then made an array
                                        that acted as a class table for the class, the word, and their occurrences given the class.
                                    </td>
                                    <td>Had to take words off that occurred less than 5 times.
                                    </td>
                                </tr>
                                <tr>
                                    <td>Made a probability function "probs()"</td>
                                    <td>I made a dynamic probability function that given the word and the class would output
                                        a probability of the word given the class using the Naive Bayes formula.
                                    </td>
                                    <td>Only could make probabilities of words given classes, but not classes given words.
                                    </td>
                                </tr>
                                <tr>
                                    <td>Explained Naive Bayes and core principles of the math behind it</td>
                                    <td>I gave an explanation of what Naive Bayes is and gave an explanation to core principles
                                        such as: Bayes vs. Frequentest, Bayes Theorem, and Naive Assumption.
                                    </td>
                                    <td>No issues.
                                    </td>
                                </tr>
                                <tr>
                                    <td>Made illustrations of the math myself</td>
                                    <td>All images and explanations used were made by myself using what I learned in class.
                                    </td>
                                    <td>No issues.
                                    </td>
                                </tr>
                            </tbody>
                        </n-table>
                    </h2>


                    <n-h1 style="color: #e4572e">References</n-h1>
                    <h2>
                        <n-table :bordered="false" :single-line="false">
                            <thead>
                                <tr>
                                    <th>Reference</th>
                                    <th>How I used the reference</th>
                                    <th>What value I made over the reference</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>"https://learn.microsoft.com/en-us/dotnet/standard/base-types/regular-expression-language-quick-reference"
                                    </td>
                                    <td>I used this website to learn Regex
                                    </td>
                                    <td>This allowed me to split my sentences using spaces and other delimiters for the "re" python library.
                                    </td>
                                </tr>
                            </tbody>
                        </n-table>
                    </h2>
                    <n-h1 style="color: #e4572e">Repository:</n-h1>
                    <n-h2>
                    <a style="color: #e07a5f" href="https://github.com/ChristopherGordSmith/NaiveBayesClassifier" target="_blank">
                        This Blogs Repository</a>
                    </n-h2>
                    
                    <n-h1 style="color: #e4572e">Downloadable Code:</n-h1>
                    <n-h2>
                        <a style="color: #e07a5f" href="assets/Data-Mining-Assignment-2.ipynb" download>
                        My Naive Bayes Model</a>
                    </n-h2>
                </n-space>
            </n-gi>
        </n-grid>
    </n-space>
</template>

<style>

</style>