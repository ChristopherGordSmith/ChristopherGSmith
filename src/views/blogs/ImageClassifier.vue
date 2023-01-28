<template>
    <n-space vertical>
        <n-grid cols="1" responsive="screen" :x-gap="100">
            <n-gi>
                <n-space>
                    <n-h1 style="color: #e4572e">Motivation</n-h1>
                    <n-h2 style="color: white">
                        For my Data Mining class we were assigned to 
                    use a dataset from

                    <a style="color: #e07a5f" target="_blank" href="https://www.kaggle.com/datasets/maricinnamon/caltech101-airplanes-motorbikes-schooners">caltech101 airplanes-motorbikes-schooners competition</a>
                    
                    . The objective was to derive a model to predict
                    the classes of each image and through experimentation increase the performance
                    of the model. This is my attempt to increase the performance of the model.
                    </n-h2>

                    <n-h1 style="color: #e4572e">Source code and background</n-h1>
                    <n-h2 style="color: white">
                        The dataset used is from 

                    <a style="color: #e07a5f" target="_blank" href="https://www.kaggle.com/datasets/maricinnamon/caltech101-airplanes-motorbikes-schooners">caltech101 airplanes-motorbikes-schooners competition</a>
                        
                    . It contains 3 folders of airplanes, motorbikes and schooners. To start the assignment, I looked online for source code that
                    implemented Convolution Neural Networks for the Machine Learning process while being able to process the images from the
                    dataset. That’s when I stumbled upon Geeks for Geeks

                    <a style="color: #e07a5f" target="_blank" href="https://www.geeksforgeeks.org/python-image-classification-using-keras/">“Python Image Classification using Keras”</a>
                    
                    article. In the article they use the Keras library for processing the images, creating, and training a model, and
                    predicting the images. However, the approach they used was an 80:20 split for their training and testing. Additionally,
                    it was only concerned with classification for 2 classes.</n-h2>

                    <n-h2 style="color: white">
                        I decided to implement a part of the source code from this article and would look for source code that implemented
                    prediction and training for more than 2 classes. I found a medium article

                    <a style="color: #e07a5f" target="_blank" href="https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720">“Tutorial on using Keras flow_from_directory and generators”</a>

                    , that gave a tutorial for the training and prediction process. This article also only
                    classified 2 classes for cats and dogs, but implemented a training, validation, and test datasets using Keras model
                    again.
                    </n-h2>

                    <n-h2 style="color: white">
                        In both models they used “binary cross entropy” for loss, “rmsprop” for optimization, and their metrics were “accuracy”.
                    Binary cross entropy’s loss function is great for binary classification and a great article I found is medium’s
                    
                    <a style="color: #e07a5f" target="_blank" href="https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a">“Understanding binary cross-entropy / logs loss: a visual explanation”</a>
                     
                    . To give a brief explanation what loss functions do in convolutional neural networks. The output is evaluated at the end of forward propagation to
                    evaluate how far off the model is from the desired goal/output of the given input by using a given loss function. In the
                    case of using multiple classes binary cross entropy isn’t best suited.
                    </n-h2>

                    <n-h2 style="color: white">
                        Therefore, in my model I used categorical cross entropy as the loss function. This loss function is ideal as it takes
                    the probabilities of each class and adjusts the weights in favor of the actual class. As opposed to binary cross entropy which is concerned about the probability of the class being a class or not. 
                    This means it isn’t really concerned about identifying other images if it can classify one and say it is or is not that class.
                    </n-h2>
                    
                    <n-h2 style="color: white">
                        They used “rmsprop” for optimization and after reading over an article from TowardsDataScience, 
                    
                    <a style="color: #e07a5f" target="_blank" href="https://towardsdatascience.com/a-look-at-gradient-descent-and-rmsprop-optimizers-f77d483ef08b">“A look at gradient descent and rmsprop optimizers”</a>

                    it seems that they used this in order to increase the learning rate
                    with larger steps without worrying about overstepping during the descent. For my model I decided to use “sgd” or
                    Stochastic Gradient Descent. I read an article by geeks for geeks 

                    <a style="color: #e07a5f" target="_blank" href="https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/">SGD</a>

                     on how the optimizer works. I
                    simply used this to try and derive different results that could be compared to using rmsprop.
                    </n-h2>
                    
                    <n-h2 style="color: white">
                        Lastly, their model used simply “accuracy” as their metric, which gives the difference between the prediction and the
                    actual. However, in the case of multi-classification, we need a metric that gives us a sum of the correctly matched/total
                    images. This is more realistic since we aren’t measuring the probability of it being a class, but if the class matched
                    was correct or not. Therefore, I used "categorical accuracy" as my metric for the model.
                    </n-h2>

                    <n-h1 style="color: #e4572e">What is forward and backward propagation?</n-h1>
                    <n-h2 style="color: white">
                        Forward propagation is passing input data through layers that reach an output layer. The input goes through layers known
                    as “Node layers”, and they have set weights on each node that performs an operation on the input. These layers are the
                    hidden layers of a neural network and for this network to exist it must possess at least one hidden layer. Once the
                    input reaches the output layer the output is then evaluated based on how far off the desired output was. After the
                    output is assessed, backward propagation begins. Backward propagation goes backward going through each layer and
                    adjusting each node of each layer in hopes to close the gap between the output and the desired output or decrease the
                    “loss”.
                    </n-h2>
                    
                    <n-h1 style="color: #e4572e">Experiments & results</n-h1>
                    <n-h2 style="color: white">
                        In the experiments I would adjust the hyper parameters of the model to derive different results.
                        Each result was recording on an 
                         <a style="color: #e07a5f" href="assets/ImageClassifier/Assignment_1_Experiment_Data.xlsx" download>excel sheet</a>
                        with each row representing an experiment. The columns representing the results:
                        training time (seconds), accuracy, batch size, epochs, node layers, loss(percent).
                    </n-h2>
                    <n-h2 style="color: white">
                        This is the result of the initial state of the program, after applying changes to make it work
                        with the new dataset and a 60:20:20 ratio for training, validation, and testing. Through out
                        the experiments I would monitor the training vs. validation performance to make sure that
                        over fitting wouldn't occur and if so I would report it in the experiment.
                    </n-h2>
                    <n-image style="margin-left: 30%; margin-right: 30%" src="assets/ImageClassifier/initial_state_loss_graph.png" width="800" />
                    <n-h2 style="color: white">
                        For accuracy I would use this prediction output for my accuracy that I calculated myself.
                        Note that the above image does have an accuracy for training and validation, but I used
                        the prediction model output as it was tested on the test dataset. Also my accuracy throughout
                        these experiments never changed as I was unsuccessful in even changing the result of the
                        accuracy.
                    </n-h2>
                    <n-h2 style="color: white">
                        From the initial state to the 1st experiment I added an extra layer to the training model to see
                        if this could possibly improve performance. This did not improve performance and actually resulted
                        in a negative effect in loss and accuracy.
                    </n-h2>
                    <n-image style="margin-left: 30%; margin-right: 30%" src="assets/ImageClassifier/Extra_layer.png" width="800" />
                    <n-image style="margin-left: 30%; margin-right: 30%" src="assets/ImageClassifier/experiment_1_loss_graph.png" width="800" />
                    <n-h2 style="color: white">
                        In this experiment I kept the initial state and doubled the epochs from 10 to 20 in hopes that
                        more runs in the training model would result in increased performance. This actually did increase
                        the accuracy, but at the cost of increased training time. However, according to the line graph
                        my training had over fitting occur.
                    </n-h2>
                    <n-image style="margin-left: 30%; margin-right: 30%" src="assets/ImageClassifier/double_epochs.png" width="800" />
                    <n-image style="margin-left: 30%; margin-right: 30%" src="assets/ImageClassifier/experiment_2_loss_graph.png" width="800" />
                    <n-h2 style="color: white">
                        In this experiment I kept the initial state and doubled the batch size from 16 to 32. The result of this 
                        was a slightly faster training time. The training model did take a slight decrease in accuracy and a significant
                        increase in loss.
                    </n-h2>
                    <n-image style="margin-left: 30%; margin-right: 30%" src="assets/ImageClassifier/double_batch.png" width="800" />
                    <n-image style="margin-left: 30%; margin-right: 30%" src="assets/ImageClassifier/experiment_3_loss_graph.png" width="800" />
                    
                    <n-image style="margin-left: 30%; margin-right: 30%" src="assets/ImageClassifier/all_experiments.png" width="800" />
                    
                    <n-h2 style="color: white">
                        <n-h1 style="color: #e4572e">Conclusion</n-h1>
                        To conclude this blog I found that increasing the epochs did increase the accuracy of the model.
                        However, it did result in over fitting as the training performance exceeded the validation performance.
                        Increasing the layers did not result in better performance and actually increased loss and accuracy
                        significantly. Doubling the batch size also did not yield any not worth results. Lastly, the initial state
                        of the model from the geeks for geeks tutorial also suffered from over performance. This means experiment #1's result
                        yielded the best outcome.
                    </n-h2>
                    <n-h1 style="color: #e4572e">Contributions</n-h1>
                    <h2>
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
                                    <td>Explanation of forward and backward propagation</td>
                                    <td>I explained at a high level the process of forward and backward propagation in my own words.</td>
                                    <td>No technical difficulties</td>
                                </tr>
                                <tr>
                                    <td>Model metric</td>
                                    <td>I added into the model a different metric rather than using accuracy I used categorical accuracy.</td>
                                    <td> No technical difficulties </td>
                                </tr>
                                <tr>
                                    <td>Experiment #1</td>
                                    <td>
                                        In this experiment I changed the hyperparameter of neural layers from 3 to 4 and had the output filters
                                        increase and then decrease in the layers. The outcome of this experiment resulted in a decrease in accuracy
                                        and increase in loss. Showing that I was possibly filtering the inputs too much to the point that nothing
                                        accurate could be derived from it.
                                    </td>
                                    <td>No technical difficulties</td>
                                </tr>
                                <tr>
                                    <td>Experiment #2</td>
                                    <td>
                                        In this experiment I changed the hyperparameter of epochs from 10 to 20. The outcome of this experiment resulted
                                        in a increase in accuracy and decrease in loss. Show that increasing the model epochs correlate to an increased
                                        performance, but longer training time.
                                    </td>
                                    <td>No technical difficulties</td>
                                </tr>
                                <tr>
                                    <td>Experiment #3</td>
                                    <td>
                                        In this experiment I changed the hyperparameter of batch sizes from 16 to 32. The outcome of this experiment resulted
                                        in a decrease in accuracy and increase in loss. A positive of this is the training time did decrease. This shows that
                                        the batch size increase in this scenario had little to no difference on the model.
                                    </td>
                                    <td>No technical difficulties</td>
                                </tr>
                                <tr>
                                    <td>Converted the binary classification model to a multi-classification model</td>
                                    <td>I was able to convert the training model from the geeks for geeks tutorial to a multi-classification model.</td>
                                    <td>
                                        I spent 3 hours trying to figure out how to solve an error I would get when using "categorical cross-entropy".
                                        Turns out the model was pushing out only 1 node and not the same amount of nodes as there were classes in the training
                                        model. It also explained why my accuracy was also very consistent.
                                    </td>
                                </tr>
                            </tbody>
                        </n-table>
                    </h2>
                    

                    <n-h1 style="color: #e4572e">Reference</n-h1>
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
                                    <td>"https://www.kaggle.com/datasets/maricinnamon/caltech101-airplanes-motorbikes-schooners"</td>
                                    <td>I used their dataset for my experiments</td>
                                    <td>
                                        I converted each of these folders into a training, validation, and test folder. I used a 60:20:20 ratio
                                        respectively.
                                        Each folder also needed to have a subfolder for motorbikes, airplanes, and schooners that contained
                                        their respective images.
                                        This was done so the Keras model would work with this dataset. This took me an hour to implement.
                                    </td>
                                </tr>
                                <tr>
                                    <td>"https://www.geeksforgeeks.org/python-image-classification-using-keras/"</td>
                                    <td>I used their training model in my code, but not their data generator and prediction code.</td>
                                    <td>This allowed me to experiment with the hyper parameters of the data.</td>
                                </tr>
                                <tr>
                                    <td>"https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720"
                                    </td>
                                    <td>I used their data generator and prediction code.</td>
                                    <td>This allowed me to derive results from the experiments I conducted on the model using various hyper
                                        parameters.</td>
                                </tr>
                                <tr>
                                    <td>"https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a"
                                    </td>
                                    <td>I read this article to give me an understanding of the binary cross entropy loss function.</td>
                                    <td>I used what I learned from this article to explain in my blog why using this loss function wouldn't work
                                        for multi-classification.
                                    </td>
                                </tr>
                                <tr>
                                    <td>"https://towardsdatascience.com/a-look-at-gradient-descent-and-rmsprop-optimizers-f77d483ef08b"
                                    </td>
                                    <td>I read this article to give me an understanding of rmsprop optimizers.</td>
                                    <td>I got an understanding that the reason the code I referenced used and rmsprop optimizer was to allow for
                                        greater learning rates.
                                        This optimizer allows for greater learning rates without the fear of over stepping on the descent.
                                    </td>
                                </tr>
                                <tr>
                                    <td>"https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/"
                                    </td>
                                    <td>I read this article to give me an understanding of stochastic gradient descent.</td>
                                    <td>I was able to derive that rmsprop optimizer and sgd optimizer are very similar, but rmsprop allows for a
                                        faster learning rate.
                                        I would like to experiment with these different optimizer to see if I get different results.
                                    </td>
                                </tr>
                            </tbody>
                        </n-table>
                    </h2>
                    
                    <n-h1 style="color: #e4572e">Downloadable Code:</n-h1>
                    <n-h2>
                        <a style="color: #e07a5f" href="assets/ImageClassifier/Reference_Code.ipynb" download>Original code from geeks for geeks</a>
                    </n-h2>
                    <n-h2>
                        <a style="color: #e07a5f" href="assets/ImageClassifier/data-mining-assignment-1.ipynb" download>My implementation with experiments</a>
                    </n-h2>
                </n-space>
            </n-gi>
        </n-grid>
    </n-space>
</template>

<style>

</style>