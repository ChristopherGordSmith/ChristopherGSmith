<template>
    <n-space vertical>
        <n-grid cols="1" responsive="screen" :x-gap="100">
            <n-gi>
                <n-space>
                    <n-h1 style="color: #e4572e">Motivation</n-h1>
                    <n-h2 style="color: white">
                        For my Data Mining class we were tasked with comparing classification algorithms. The task
                        was to be carried out by implementing the code for these algorithms and test them against
                        our choice of a 
                        <a style="color: #e07a5f" target="_blank"
                            href="httpshttps://www.kaggle.com/datasets/vivekgediya/ecommerce-product-review-data">text classifier dataset</a>
                        or an 
                        <a style="color: #e07a5f" target="_blank" href="https://www.kaggle.com/datasets/whenamancodes/wild-animals-images">image classifier dataset</a>. 
                        I chose the latter and decided to compare the algorithms Convolutional Neural Networks, 
                        Support Vector Machines, and K Nearest Neighbors.
                    </n-h2>
                    <n-h1 style="color: #e4572e">Objective</n-h1>
                    <n-h2 style="color: white">
                        The object of this blog is to do a comparison of each algorithm and show their performances on the
                        <a style="color: #e07a5f" target="_blank" href="https://www.kaggle.com/datasets/whenamancodes/wild-animals-images">wild animals dataset</a>.
                        I'll also give a walk through of each algorithm I'm comparing to give you an understanding of each one.
                        I will also explain any obstacles during this process and how I overcame them as well as any shortcomings in the
                        objective.
                    </n-h2>
                    <n-h1 style="color: #e4572e">Convolutional Neural Networks</n-h1>
                    <n-h2 style="color: white">
                        I have a blog post already going over Convolutional Neural Networks above under the "Image Classifier" post.
                        I'll be going over it again, but in less detail. Convolutional Neural Networks are algorithm with layers of nodes
                        that a receive inputs and slightly adjust the given inputs for a desired output. A Neural Networks must have at least
                        one hidden layer that separates the input layer from the output layer. This layer performs an adjustment on the inputs
                        to an extent. This extent is a weight that depending on the weight will dramatically alter it or slightly alter it.
                        These networks learn by measuring the difference in the result and the desired output and have the weights readjusted
                        accordingly. I go over the concept of Forward Propagation and Backward Propagation in my blog post "Image Classifier".
                        The amount of outputs of this network can vary, but for my purposes the outputs were 6 with each output node representing
                        the probability of the image belonging to a given class.
                    </n-h2>
                    <n-h1 style="color: #e4572e">Support Vector Machines</n-h1>
                    <n-h2 style="color: white">
                        Support Vector Machines use matrices to map out objects based on their features. The dimensions of the matrix is based
                        on the amount of features the object has. In our case the dimensions are based on how many pixels the image has. This
                        is because the code I used was specifically looking for the intensity at each pixel. After plotting each object or in our
                        case each image. It then measures each image from one to another based on their features and class. With these measurements
                        it attempts to create a line that will segregate the classes from one another. This line will attempt to form itself to give
                        each class as great of a margin of error as it can. This is an attempt to give the classes breathing room to be slightly different
                        from one another so when predicting a class we have a better chance at being correct. This line is called a support vector
                        which is where this algorithm gets its name from.
                    </n-h2>
                    <n-h1 style="color: #e4572e">K Nearest Neighbors</n-h1>
                    <n-h2 style="color: white">
                        K Nearest Neighbors also uses matrices to map out object based on their features. The dimensions are also based on the 
                        amount of features the object has. This algorithm will also use each image's pixel intensities to determine uniqueness.
                        However, where Support Vector Machines and K Nearest Neighbors differs is how they train and predict. K Nearest Neighbors
                        doesn't have a training time as it doesn't need to create a vector/line to separate each class. Instead when asked to predict
                        the class of an object/image they simply grab their K Nearest Neighbors. They measure each feature from one another using 
                        Euclidean Distance.
                        <n-image style="margin-left: 30%; margin-right: 30%" src="assets/EuclideanDistance.png" width="500" />
                        <n-h5 style="margin-top: 0px; color: white; margin-right:44%; margin-left:44%">Euclidean Distance
                        </n-h5>
                        Where x and y are two objects and i is a feature of the objects. Once comparing the distance of each object to the object to
                        be predicted it then grabs its nearest K neighbors. This is where K Nearest Neighbors gets its name. K is a hyper parameter
                        that must be adjusted to give better results. You have to make sure the K is not grabbing too many neighbors and not too little.
                        After grabbing the neighbors the majority class wins the prediction. Normally you want your k to be an odd number to prevent ties.
                    </n-h2>
                    <n-h1 style="color: #e4572e">Source code for Convolutional Neural Networks</n-h1>
                    <n-h2 style="color: white">
                        The code for this blog is the same as the code used in the "Image Classifier" blog post. The code came from two tutorials, 
                    <a style="color: #e07a5f" target="_blank" href="https://www.geeksforgeeks.org/python-image-classification-using-keras/">“Python Image Classification using Keras”</a>
                    and 
                    <a style="color: #e07a5f" target="_blank"
                        href="https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720">“Tutorial
                        on using Keras flow_from_directory and generators”</a>.
                    The former allowed me to create and modify the hyper parameters of the model. While the latter allowed me to implement predictions
                    so I could measure accuracy.
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/FinalCNNCode1.png" width="500" />
                    In the code above I'm getting a count of all the files to reconfirm the amount in each directory for each dataset. The dataset was split 60:20:20
                    Train:Validation:Test.
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/FinalCNNCode2.png" width="500" />
                    In this snippet of code I set up the epochs which tells the model how many times to run through and the batch size. I then edited
                    the number of layers and the number of node/parameters of each layer till my accuracy rose enough. Generally high epochs will increase
                    the accuracy, but this is also prone to overfitting where the model is too tuned to the training that it can't identify other objects.
                    In other words it can be too perfect. During testing this can hurt the accuracy of the model.
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/FinalCNNCode3.png" width="500" />
                    At this point we are getting the images of the models and turning them into arrays that the program can understand. Then we fit the model
                    with our training data and hyper parameters.
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/FinalCNNCode4.png" width="500" />
                    Lastly we test our model and measure the accuracy of the model. Below is the output of the model with a layout of the layers and
                    the result of running the dataset with this model. This model did suffer a little bit of over fitting, but I don't think it was enough
                    to give bad results.
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/FinalCNNCode5.png" width="500" />
                    </n-h2>
                    <n-h1 style="color: #e4572e">Source code for Support Vector Machines</n-h1>
                    <n-h2 style="color: white">
                        The code for this algorithm was from a tutorial on medium called,
                    <a style="color: #e07a5f" target="_blank" href="https://medium.com/analytics-vidhya/image-classification-using-machine-learning-support-vector-machine-svm-dc7a0ec92e01">“Image Classification Using Machine Learning-Support Vector Machine(SVM)”</a>.
                    I modified the code to have it work with the dataset we are using for this dataset. In the code below I set up what the categories and directories being used were.
                    Since their code using sklearn's train_test_split I had images merge together and then adjusted the test size on the function. In for loop you can see the dimensions of the
                    image being resized to a set image. Since the code was using pixel intensities to measure, I treated the dimension sizes there as a hyper parameter and adjusted how many pixels
                    would be used.
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/FinalSVMCode1.png" width="500" />
                    After this we set up our SVM with a Gamma for influence, C for cost of misclassifications, and the Kernel. Afterwards we train our model and fit it. Then we start
                    predicting images with our test data and record the accuracy.
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/FinalSVMCode2.png" width="500" />
                    Below is a result of each dimensions size I used as a hyper parameter to derive a better accuracy.
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/100by100.png" width="500" />
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/100by100time.png" width="500" />
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/150by150.png" width="500" />
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/150by150time.png" width="500" />
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/fiftybyfifty.png" width="500" />
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/fiftybyfiftytime.png" width="500" />
                    In these graphs below you can see that as the dimensions of the images got bigger the training time dramatically increased, however the accuracy
                    of the model did not. I felt that maybe since the features are pixels it would actually become too complex for the SVM to simply use distances.
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/SVMTrainTime.png" width="500" />
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/SVMAccuracy.png" width="500" />
                    In the final comparison of the algorithms I'll be using the 100 by 100 dimensions result.
                    </n-h2>
                    <n-h1 style="color: #e4572e">Source code for K Nearest Neighbors</n-h1>
                    <n-h2 style="color: white">
                        The code for the algorithm was from a tutorial on medium called,
                        <a style="color: #e07a5f" target="_blank"
                            href="https://medium.com/swlh/image-classification-with-k-nearest-neighbours-51b3a289280">“Image Classification with K Nearest Neighbours”</a>.
                        I modified the code to have it work with the dataset we are using for this dataset. In the code below I set up the categories and once again mashed
                        all of the data together since they were also using the sklearn's train_test_split and I simply adjusted the test size.
                        In the load_image_files you can see a dimension parameter which I treated as a hyper parameter to achieve better accuracies.
                        These images also treated pixel intensities as features.
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/FinalKNNCode1.png" width="500" />
                        In this part of the code we are setting up the testing and training data and to be fitted into the KNeighborsClassifier model.
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/FinalKNNCode2.png" width="500" />
                        At this point we are fitting the model and evaluating the score. In their code they wrote a for loop that would find the best K
                        for the model and report that K with their accuracy. I extended the range to 200 to see if checking more Ks would possibly improve accuracy.
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/FinalKNNCode3.png" width="500" />
                        Below you can see the results of each dimension I tried for the images. My theory was adding a greater dimension would increase the accuracy
                        since more data would be utilized, however it seems the higher the dimension worse it got.
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/KNN150by150.png" width="500" />
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/KNN200by200time.png" width="500" />
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/KNN250by250.png" width="500" />
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/KNNAccuracy.png" width="500" />
                        I believe the results correlated with the same issue that SVM had with too many features being used therefore being too complex for SVM or KNN
                        to be accurate. Additionally, you can see the more K increased the worse the model did. In the final comparison I used the 150 by 150 dimension result.
                    </n-h2>
                    <n-h1 style="color: #e4572e">Performance Comparison</n-h1>
                    <n-h2 style="color: white">
                        Below is a comparison of each model below with their training times and accuracies. The Convolutional Neural Network proved to be the best for accuracy,
                        but had a long training time. Did well behind CNN in second, but also had a very long training time. KNN came in last, but had a dramatically lesser training time.
                        <n-image style="margin-left: 35%; margin-right: 35%" src="assets/AlgorithmComparisons.png" width="500" />
                        I believe SVM and KNN suffered due to major over fitting of the models to their training data. They are designed to rely on the training data solely, but
                        as the complexity or features increased the model quickly grew less accurate. CNN did suffer from over fitting, but not as bad as KNN and SVM.
                    </n-h2>
                    <n-h1 style="color: #e4572e">Pros & Cons For Each Classifier</n-h1>s
                    <n-h2 style="font-weight: 700;">
                        <n-h2 style="color: white">Convolutional Neural Networks</n-h2>
                        <n-table :bordered="false" :single-line="false" style="margin-left: 20%; margin-right: 20%">
                            <thead>
                                <tr>
                                    <th>Pros</th>
                                    <th>Cons</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Works well with complex objects for classification
                                    </td>
                                    <td>Training Time can be long based on how many layers, epochs, and dataset size
                                    </td>
                                </tr>
                                <tr>
                                    <td>Super effective with image classification
                                    </td>
                                    <td>The algorithm is a little complex and needs strong math basis
                                    </td>
                                </tr>
                            </tbody>
                        </n-table>
                        <n-h2 style="color: white">Support Vector Machines</n-h2>
                        <n-table :bordered="false" :single-line="false" style="margin-left: 20%; margin-right: 20%">
                            <thead>
                                <tr>
                                    <th>Pros</th>
                                    <th>Cons</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Faster training time than Convolutional Neural Networks
                                    </td>
                                    <td>The algorithm becomes more complex as the dimensions increase
                                    </td>
                                </tr>
                                <tr>
                                    <td>More accurate than KNN
                                    </td>
                                    <td>Can't handle super complex / high number of features
                                    </td>
                                </tr>
                            </tbody>
                        </n-table>
                        <n-h2 style="color: white">K Nearest Neighbors</n-h2>
                        <n-table :bordered="false" :single-line="false" style="margin-left: 20%; margin-right: 20%">
                            <thead>
                                <tr>
                                    <th>Pros</th>
                                    <th>Cons</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Super fast training time
                                    </td>
                                    <td>Fails when object is super complex / features are high
                                    </td>
                                </tr>
                                <tr>
                                    <td>Algorithm is very simple
                                    </td>
                                    <td>Very Memory hungry
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                    </td>
                                    <td>Least Accurate
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
                                    <td>Allowed the data to be used for raw words that didn't have any extra unneeded
                                        information</td>
                                    <td>Had to allow sum words to be missed since the the special characters such as
                                        single quotes needed to remain.
                                        I also had to learn regex and how to utilize it for the splitting.
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
                                    <td>This allowed me to split my sentences using spaces and other delimiters for the
                                        "re" python library.
                                    </td>
                                </tr>
                            </tbody>
                        </n-table>
                    </h2>
                    <n-h1 style="color: #e4572e">Repository:</n-h1>
                    <n-h2>
                        <a style="color: #e07a5f" href="https://github.com/ChristopherGordSmith/final_project_Data_mining"
                            target="_blank">
                            This Blog's Repository</a>
                    </n-h2>

                    <n-h1 style="color: #e4572e">Downloadable Code:</n-h1>
                    <n-h2>
                        <a style="color: #e07a5f" href="assets/final-data-mining-project.ipynb" download>
                            CNN Classifier</a>
                        <a style="color: #e07a5f" href="assets/SVMImageClassifier.ipynb" download>
                            SVM Classifier</a>
                        <a style="color: #e07a5f" href="assets/knn-classifier.ipynb" download>
                            KNN Classifier</a>
                    </n-h2>
                </n-space>
            </n-gi>
        </n-grid>
    </n-space>
</template>

<style>

</style>