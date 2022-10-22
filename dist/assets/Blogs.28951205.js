import{_ as p,r as i,o as d,c as h,w as o,a as t,b as a,d as e,e as _}from"./index.8d6ba077.js";const y={},w=a(" I followed through a tutorial by Alexis Cook that went through the process of completing in a Kaggle competition. The practice competition uses a passenger dataset from the Titanic. The dataset specifies a passengers attributes such as age, sex, ticket class, etc. The dataset also indicates if they survived or not. The challenge is to create a prediction model that predicts whether a given passenger would survive. "),b=a("Going into the directory with the training data, testing data and example prediction .csv files."),v=a("Assigning training data and testing data. Also displaying the 1st five passengers with their respective data for each dataset."),k=a("Testing if gender_submission.csv prediction of all female passengers survived and all male passengers died. Since the numbers are 74% women and 18% men we can say that this prediction is significantly inaccurate. "),x=a("Using the Random Tree Model to make predictions based on a given passenger's data."),I=a(" After following the tutorial from Alexis Cook we built a prediction model that uses a random forest model. A random forest model produces trees that take each attribute of the passenger's data into account. The model uses a majority vote with the outcome of the trees to predict if the passenger survived. The model we built only accounted for the passenger's class, sex, amount of siblings and spouses, amount of parents and children. This model produced a score of .77511 accuracy. "),T=e("h1",{style:{color:"white"}},"Reference:",-1),z=e("a",{href:"https://www.kaggle.com/code/alexisbcook/titanic-tutorial/notebook"},"Alexis Cook's Titanic Kaggle Tutorial (Click me)",-1);function C(u,f){const n=i("n-h2"),s=i("n-image"),r=i("n-space"),l=i("n-gi"),c=i("n-grid");return d(),h(r,{vertical:""},{default:o(()=>[t(c,{cols:"1",responsive:"screen","x-gap":100},{default:o(()=>[t(l,null,{default:o(()=>[t(r,null,{default:o(()=>[t(n,{style:{color:"white"}},{default:o(()=>[w]),_:1}),t(n,{style:{color:"white"}},{default:o(()=>[b]),_:1}),t(s,{src:"assets/KaggleTutorialSnip1.png",width:"800"}),t(n,{style:{color:"white"}},{default:o(()=>[v]),_:1}),t(s,{src:"assets/KaggleTutorialSnip2.png",width:"800"}),t(n,{style:{color:"white"}},{default:o(()=>[k]),_:1}),t(s,{src:"assets/KaggleTutorialSnip3.png",width:"800"}),t(n,{style:{color:"white"}},{default:o(()=>[x]),_:1}),t(s,{src:"assets/KaggleTutorialSnip4.png",width:"800"}),t(n,{style:{color:"white"}},{default:o(()=>[I]),_:1}),t(s,{src:"assets/KaggleTutorialSnip5.png",width:"800"})]),_:1})]),_:1})]),_:1}),T,z]),_:1})}const A=p(y,[["render",C]]),K={},$=a("Motivation"),N=a(" For my Data Mining class we were assigned to use a dataset from "),S=e("a",{style:{color:"#e07a5f"},target:"_blank",href:"https://www.kaggle.com/datasets/maricinnamon/caltech101-airplanes-motorbikes-schooners"},"caltech101 airplanes-motorbikes-schooners competition",-1),E=a(" . The objective was to derive a model to predict the classes of each image and through experimentation increase the performance of the model. This is my attempt to increase the performance of the model. "),j=a("Source code and background"),B=a(" The dataset used is from "),D=e("a",{style:{color:"#e07a5f"},target:"_blank",href:"https://www.kaggle.com/datasets/maricinnamon/caltech101-airplanes-motorbikes-schooners"},"caltech101 airplanes-motorbikes-schooners competition",-1),M=a(" . It contains 3 folders of airplanes, motorbikes and schooners. To start the assignment, I looked online for source code that implemented Convolution Neural Networks for the Machine Learning process while being able to process the images from the dataset. That\u2019s when I stumbled upon Geeks for Geeks "),F=e("a",{style:{color:"#e07a5f"},target:"_blank",href:"https://www.geeksforgeeks.org/python-image-classification-using-keras/"},"\u201CPython Image Classification using Keras\u201D",-1),G=a(" article. In the article they use the Keras library for processing the images, creating, and training a model, and predicting the images. However, the approach they used was an 80:20 split for their training and testing. Additionally, it was only concerned with classification for 2 classes."),H=a(" I decided to implement a part of the source code from this article and would look for source code that implemented prediction and training for more than 2 classes. I found a medium article "),R=e("a",{style:{color:"#e07a5f"},target:"_blank",href:"https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720"},"\u201CTutorial on using Keras flow_from_directory and generators\u201D",-1),L=a(" , that gave a tutorial for the training and prediction process. This article also only classified 2 classes for cats and dogs, but implemented a training, validation, and test datasets using Keras model again. "),V=a(" In both models they used \u201Cbinary cross entropy\u201D for loss, \u201Crmsprop\u201D for optimization, and their metrics were \u201Caccuracy\u201D. Binary cross entropy\u2019s loss function is great for binary classification and a great article I found is medium\u2019s "),O=e("a",{style:{color:"#e07a5f"},target:"_blank",href:"https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a"},"\u201CUnderstanding binary cross-entropy / logs loss: a visual explanation\u201D",-1),U=a(" . To give a brief explanation what loss functions do in convolutional neural networks. The output is evaluated at the end of forward propagation to evaluate how far off the model is from the desired goal/output of the given input by using a given loss function. In the case of using multiple classes binary cross entropy isn\u2019t best suited. "),W=a(" Therefore, in my model I used categorical cross entropy as the loss function. This loss function is ideal as it takes the probabilities of each class and adjusts the weights in favor of the actual class. As opposed to binary cross entropy which is concerned about the probability of the class being a class or not. This means it isn\u2019t really concerned about identifying other images if it can classify one and say it is or is not that class. "),P=a(" They used \u201Crmsprop\u201D for optimization and after reading over an article from TowardsDataScience, "),q=e("a",{style:{color:"#e07a5f"},target:"_blank",href:"https://towardsdatascience.com/a-look-at-gradient-descent-and-rmsprop-optimizers-f77d483ef08b"},"\u201CA look at gradient descent and rmsprop optimizers\u201D",-1),J=a(" it seems that they used this in order to increase the learning rate with larger steps without worrying about overstepping during the descent. For my model I decided to use \u201Csgd\u201D or Stochastic Gradient Descent. I read an article by geeks for geeks "),Q=e("a",{style:{color:"#e07a5f"},target:"_blank",href:"https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/"},"SGD",-1),X=a(" on how the optimizer works. I simply used this to try and derive different results that could be compared to using rmsprop. "),Y=a(' Lastly, their model used simply \u201Caccuracy\u201D as their metric, which gives the difference between the prediction and the actual. However, in the case of multi-classification, we need a metric that gives us a sum of the correctly matched/total images. This is more realistic since we aren\u2019t measuring the probability of it being a class, but if the class matched was correct or not. Therefore, I used "categorical accuracy" as my metric for the model. '),Z=a("What is forward and backward propagation?"),ee=a(" Forward propagation is passing input data through layers that reach an output layer. The input goes through layers known as \u201CNode layers\u201D, and they have set weights on each node that performs an operation on the input. These layers are the hidden layers of a neural network and for this network to exist it must possess at least one hidden layer. Once the input reaches the output layer the output is then evaluated based on how far off the desired output was. After the output is assessed, backward propagation begins. Backward propagation goes backward going through each layer and adjusting each node of each layer in hopes to close the gap between the output and the desired output or decrease the \u201Closs\u201D. "),te=a("Experiments & results"),oe=a(" In the experiments I would adjust the hyper parameters of the model to derive different results. Each result was recording on an "),ae=e("a",{style:{color:"#e07a5f"},href:"assets/Assignment_1_Experiment_Data.xlsx",download:""},"excel sheet",-1),se=a(" with each row representing an experiment. The columns representing the results: training time (seconds), accuracy, batch size, epochs, node layers, loss(percent). "),ne=a(" This is the result of the initial state of the program, after applying changes to make it work with the new dataset and a 60:20:20 ratio for training, validation, and testing. Through out the experiments I would monitor the training vs. validation performance to make sure that over fitting wouldn't occur and if so I would report it in the experiment. "),ie=a(" For accuracy I would use this prediction output for my accuracy that I calculated myself. Note that the above image does have an accuracy for training and validation, but I used the prediction model output as it was tested on the test dataset. Also my accuracy throughout these experiments never changed as I was unsuccessful in even changing the result of the accuracy. "),re=a(" From the initial state to the 1st experiment I added an extra layer to the training model to see if this could possibly improve performance. This did not improve performance and actually resulted in a negative effect in loss and accuracy. "),le=a(" In this experiment I kept the initial state and doubled the epochs from 10 to 20 in hopes that more runs in the training model would result in increased performance. This actually did increase the accuracy, but at the cost of increased training time. However, according to the line graph my training had over fitting occur. "),ce=a(" In this experiment I kept the initial state and doubled the batch size from 16 to 32. The result of this was a slightly faster training time. The training model did take a slight decrease in accuracy and a significant increase in loss. "),de=a("Conclusion"),he=a(" To conclude this blog I found that increasing the epochs did increase the accuracy of the model. However, it did result in over fitting as the training performance exceeded the validation performance. Increasing the layers did not result in better performance and actually increased loss and accuracy significantly. Doubling the batch size also did not yield any not worth results. Lastly, the initial state of the model from the geeks for geeks tutorial also suffered from over performance. This means experiment #1's result yielded the best outcome. "),ue=a("Contributions"),fe=e("thead",null,[e("tr",null,[e("th",null,"Contribution"),e("th",null,"The value of the contribution"),e("th",null,"Technical Challenges")])],-1),pe=e("tbody",null,[e("tr",null,[e("td",null,"Explanation of forward and backward propagation"),e("td",null,"I explained at a high level the process of forward and backward propagation in my own words."),e("td",null,"No technical difficulties")]),e("tr",null,[e("td",null,"Model metric"),e("td",null,"I added into the model a different metric rather than using accuracy I used categorical accuracy."),e("td",null," No technical difficulties ")]),e("tr",null,[e("td",null,"Experiment #1"),e("td",null," In this experiment I changed the hyperparameter of neural layers from 3 to 4 and had the output filters increase and then decrease in the layers. The outcome of this experiment resulted in a decrease in accuracy and increase in loss. Showing that I was possibly filtering the inputs too much to the point that nothing accurate could be derived from it. "),e("td",null,"No technical difficulties")]),e("tr",null,[e("td",null,"Experiment #2"),e("td",null," In this experiment I changed the hyperparameter of epochs from 10 to 20. The outcome of this experiment resulted in a increase in accuracy and decrease in loss. Show that increasing the model epochs correlate to an increased performance, but longer training time. "),e("td",null,"No technical difficulties")]),e("tr",null,[e("td",null,"Experiment #3"),e("td",null," In this experiment I changed the hyperparameter of batch sizes from 16 to 32. The outcome of this experiment resulted in a decrease in accuracy and increase in loss. A positive of this is the training time did decrease. This shows that the batch size increase in this scenario had little to no difference on the model. "),e("td",null,"No technical difficulties")]),e("tr",null,[e("td",null,"Converted the binary classification model to a multi-classification model"),e("td",null,"I was able to convert the training model from the geeks for geeks tutorial to a multi-classification model."),e("td",null,' I spent 3 hours trying to figure out how to solve an error I would get when using "categorical cross-entropy". Turns out the model was pushing out only 1 node and not the same amount of nodes as there were classes in the training model. It also explained why my accuracy was also very consistent. ')])],-1),ge=a("Reference"),me=e("thead",null,[e("tr",null,[e("th",null,"Reference"),e("th",null,"How I used the reference"),e("th",null,"What value I made over the reference")])],-1),_e=e("tbody",null,[e("tr",null,[e("td",null,'"https://www.kaggle.com/datasets/maricinnamon/caltech101-airplanes-motorbikes-schooners"'),e("td",null,"I used their dataset for my experiments"),e("td",null," I converted each of these folders into a training, validation, and test folder. I used a 60:20:20 ratio respectively. Each folder also needed to have a subfolder for motorbikes, airplanes, and schooners that contained their respective images. This was done so the Keras model would work with this dataset. This took me an hour to implement. ")]),e("tr",null,[e("td",null,'"https://www.geeksforgeeks.org/python-image-classification-using-keras/"'),e("td",null,"I used their training model in my code, but not their data generator and prediction code."),e("td",null,"This allowed me to experiment with the hyper parameters of the data.")]),e("tr",null,[e("td",null,'"https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720" '),e("td",null,"I used their data generator and prediction code."),e("td",null,"This allowed me to derive results from the experiments I conducted on the model using various hyper parameters.")]),e("tr",null,[e("td",null,'"https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a" '),e("td",null,"I read this article to give me an understanding of the binary cross entropy loss function."),e("td",null,"I used what I learned from this article to explain in my blog why using this loss function wouldn't work for multi-classification. ")]),e("tr",null,[e("td",null,'"https://towardsdatascience.com/a-look-at-gradient-descent-and-rmsprop-optimizers-f77d483ef08b" '),e("td",null,"I read this article to give me an understanding of rmsprop optimizers."),e("td",null,"I got an understanding that the reason the code I referenced used and rmsprop optimizer was to allow for greater learning rates. This optimizer allows for greater learning rates without the fear of over stepping on the descent. ")]),e("tr",null,[e("td",null,'"https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/" '),e("td",null,"I read this article to give me an understanding of stochastic gradient descent."),e("td",null,"I was able to derive that rmsprop optimizer and sgd optimizer are very similar, but rmsprop allows for a faster learning rate. I would like to experiment with these different optimizer to see if I get different results. ")])],-1),ye=a("Downloadable Code:"),we=e("a",{style:{color:"#e07a5f"},href:"assets/Reference_Code.ipynb",download:""},"Original code from geeks for geeks",-1),be=e("a",{style:{color:"#e07a5f"},href:"assets/data-mining-assignment-1.ipynb",download:""},"My implementation with experiments",-1);function ve(u,f){const n=i("n-h1"),s=i("n-h2"),r=i("n-image"),l=i("n-table"),c=i("n-space"),g=i("n-gi"),m=i("n-grid");return d(),h(c,{vertical:""},{default:o(()=>[t(m,{cols:"1",responsive:"screen","x-gap":100},{default:o(()=>[t(g,null,{default:o(()=>[t(c,null,{default:o(()=>[t(n,{style:{color:"#e4572e"}},{default:o(()=>[$]),_:1}),t(s,{style:{color:"white"}},{default:o(()=>[N,S,E]),_:1}),t(n,{style:{color:"#e4572e"}},{default:o(()=>[j]),_:1}),t(s,{style:{color:"white"}},{default:o(()=>[B,D,M,F,G]),_:1}),t(s,{style:{color:"white"}},{default:o(()=>[H,R,L]),_:1}),t(s,{style:{color:"white"}},{default:o(()=>[V,O,U]),_:1}),t(s,{style:{color:"white"}},{default:o(()=>[W]),_:1}),t(s,{style:{color:"white"}},{default:o(()=>[P,q,J,Q,X]),_:1}),t(s,{style:{color:"white"}},{default:o(()=>[Y]),_:1}),t(n,{style:{color:"#e4572e"}},{default:o(()=>[Z]),_:1}),t(s,{style:{color:"white"}},{default:o(()=>[ee]),_:1}),t(n,{style:{color:"#e4572e"}},{default:o(()=>[te]),_:1}),t(s,{style:{color:"white"}},{default:o(()=>[oe,ae,se]),_:1}),t(s,{style:{color:"white"}},{default:o(()=>[ne]),_:1}),t(r,{style:{"margin-left":"30%","margin-right":"30%"},src:"assets/initial_state_loss_graph.png",width:"800"}),t(s,{style:{color:"white"}},{default:o(()=>[ie]),_:1}),t(s,{style:{color:"white"}},{default:o(()=>[re]),_:1}),t(r,{style:{"margin-left":"30%","margin-right":"30%"},src:"assets/Extra_layer.png",width:"800"}),t(r,{style:{"margin-left":"30%","margin-right":"30%"},src:"assets/experiment_1_loss_graph.png",width:"800"}),t(s,{style:{color:"white"}},{default:o(()=>[le]),_:1}),t(r,{style:{"margin-left":"30%","margin-right":"30%"},src:"assets/double_epochs.png",width:"800"}),t(r,{style:{"margin-left":"30%","margin-right":"30%"},src:"assets/experiment_2_loss_graph.png",width:"800"}),t(s,{style:{color:"white"}},{default:o(()=>[ce]),_:1}),t(r,{style:{"margin-left":"30%","margin-right":"30%"},src:"assets/double_batch.png",width:"800"}),t(r,{style:{"margin-left":"30%","margin-right":"30%"},src:"assets/experiment_3_loss_graph.png",width:"800"}),t(r,{style:{"margin-left":"30%","margin-right":"30%"},src:"assets/all_experiments.png",width:"800"}),t(s,{style:{color:"white"}},{default:o(()=>[t(n,{style:{color:"#e4572e"}},{default:o(()=>[de]),_:1}),he]),_:1}),t(n,{style:{color:"#e4572e"}},{default:o(()=>[ue]),_:1}),e("h2",null,[t(l,{bordered:!1,"single-line":!1},{default:o(()=>[fe,pe]),_:1})]),t(n,{style:{color:"#e4572e"}},{default:o(()=>[ge]),_:1}),e("h2",null,[t(l,{bordered:!1,"single-line":!1},{default:o(()=>[me,_e]),_:1})]),t(n,{style:{color:"#e4572e"}},{default:o(()=>[ye]),_:1}),t(s,null,{default:o(()=>[we]),_:1}),t(s,null,{default:o(()=>[be]),_:1})]),_:1})]),_:1})]),_:1})]),_:1})}const ke=p(K,[["render",ve]]),xe=a("Kaggle Titanic Tutorial"),Ie=a("Image Classifier"),ze=_({__name:"Blogs",setup(u){return(f,n)=>{const s=i("n-h1"),r=i("n-collapse-item"),l=i("n-collapse"),c=i("n-space");return d(),h(c,{style:{padding:"50px 100px 50px 100px"}},{default:o(()=>[t(l,{"default-expanded-names":"1",accordion:""},{default:o(()=>[t(r,null,{header:o(()=>[t(s,{style:{color:"#e4572e","font-size":"40px"}},{default:o(()=>[xe]),_:1})]),default:o(()=>[t(A)]),_:1}),t(r,null,{header:o(()=>[t(s,{style:{color:"#e4572e","font-size":"40px"}},{default:o(()=>[Ie]),_:1})]),default:o(()=>[t(ke)]),_:1})]),_:1})]),_:1})}}});export{ze as default};