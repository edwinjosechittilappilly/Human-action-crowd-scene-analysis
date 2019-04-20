# Human-action-crowd-scene-analysis
https://docs.google.com/document/d/e/2PACX-1vREE56B8HMJ9G0kzT4ZRAvJvlILVaGX96dRyLUIUIZHYr6Z8YjSveiQF-BOmpJaXuHFFj3sSPu0QzjV/pub

Crowd Behaviour Analysis


Edwin Jose
Department of Electronics
Cochin University of Science and Technology
Kerala, India
edwin.jose@cusat.ac.in

Greeshma Manikandan
Department of Electronics
Cochin University of Science and Technology
Kerala, India


Mithun Haridas T.P.
Department of Electronics
Cochin University of Science and Technology
Kerala, India
mithuntp@cusat.ac.in



Abstract—The study of human behavior is a subject of great scientific interest and probably an inexhaustible source of research. With the improvement of computer vision technique, several applications in this area, like video surveillance, human behavior understanding, or measurements of athletic performance have been tackled using automated or semi-automated techniques.
With the growth of crowd and population  in the real world, crowd scene understanding is becoming an important task in anomaly detection and public security. Visual ambiguities and occlusions, high density, low mobility, and scene semantics, video quality however, make this problem a great challenge.
Through this project, we would like to study and implement the various methods of crowd behavior analysis using machine learning principles. And analyse the performance of our network with the traditional methods.
Implementation
Initial Methods
VGG16 feature extraction
   The first network is a convolutional neural network with the purpose of extracting high-level features of the images and reducing the complexity of the input. Using a pre-trained model called VGG16 .VGG16 (also called OxfordNet) is a convolutional neural network architecture named after the Visual Geometry Group from Oxford, who developed it. It was used to win the ILSVR (ImageNet) competition in 2014.
  This model used  to apply  the technique of transfer learning. Modern object recognition models have millions of parameters and can take weeks to fully train. Transfer learning is a technique that optimizes a lot of this work by taking a fully trained model for a set of categories like ImageNet and retrains from the existing weights for new classes.
The first step is to extract the frames of the video. extracting using this frame, to make a prediction using the OxfordNet model. The video data preprocessing is as shown in the block diagram in Fig1.

Fig1: block diagram for extracting frames using opencv
   Considering  the transfer learning technique, the final classification layer of the inception model is not used . Instead, by extracting the result of the last pooling layer, which is a vector of 2,048 values (high-level feature map). Also instead of a single frame, a group of frames in order to classify not the frame but a segment of the video.
The model of the project is defined as shown below. The model consists of a vgg16 layer for feature extraction and followed by the time distributed input to the LSTM which is followed by a ReLU hidden layer and an output of softmax layer

MobileNetV2 feature extraction
   MobileNets are small, low-latency, low-power models parameterised to meet the resource constraints of a variety of use cases. According to the research paper, MobileNetV2 improves the state-of-the-art performance of mobile models on multiple tasks and benchmarks as well as across a spectrum of different model sizes. It is a very effective feature extractor for object detection and segmentation. For instance, for detection, when paired with Single Shot Detector Lite, MobileNetV2 is about 35 percent faster with the same accuracy than MobileNetV1.
  It builds upon the ideas from MobileNetV1, using depth-wise separable convolutions as efficient building blocks. However, Google says that the 2nd version of MobileNet has two new features:
  Linear bottlenecks between the layers: Experimental evidence suggests that using linear layers is crucial as it prevents nonlinearities from destroying too much information. Using non-linear layers in bottlenecks indeed hurts the performance by several percent, further validating our hypothesis
Shortcut connections between the bottlenecks.
Source :https://www.analyticsindiamag.com/why-googles-mobilenetv2-is-a-revolutionary-next-gen-on-device-computer-vision-network/

UCF101 is an action recognition data set
UCF101 is an action recognition data set of realistic action videos, collected from YouTube, having 101 action categories. This data set is an extension of UCF50 dataset which has 50 action categories.
With 13320 videos from 101 action categories, UCF101 gives the largest diversity in terms of actions and with the presence of large variations in camera motion, object appearance and pose, object scale, viewpoint, cluttered background, illumination conditions, etc, it is the most challenging dataset to date. As most of the available action recognition data sets are not realistic and are staged by actors, UCF101 aims to encourage further research into action recognition by learning and exploring new realistic action categories.


Fig. 2.   Sample screenshot of dataset 
The videos in 101 action categories are grouped into 25 groups, where each group can consist of 4-7 videos of an action. The videos from the same group may share some common features, such as similar background, similar viewpoint, etc.
The action categories can be divided into five types: 1)Human-Object Interaction 2) Body-Motion Only 3) Human-Human Interaction 4) Playing Musical Instruments 5) Sports.
Where this project   focus on human-object interaction and human –human interaction.


LSTM
One of the appeals of RNNs is the idea that they might be able to connect previous information to the present task, such as using previous video frames might inform the understanding of the present frame.  In cases, where the gap between the relevant information and the place that it’s needed is small, RNNs can learn to use the past information. But RNNs can’t handle “long term dependencies”.

LSTM Networks
Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by Hochreiter and Schmidhuber (1997), and were refined and popularized by many people in following work.1 They work tremendously well on a large variety of problems, and are now widely used.
LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!
All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer.
LSTMs also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.
   The core concept of LSTM’s are the cell state, and it’s various gates. The cell state act as a transport highway that transfers relative information all the way down the sequence chain. The cell state, in theory, can carry relevant information throughout the processing of the sequence. So even information from the earlier time steps can make its way to later time steps, reducing the effects of short-term memory. As the cell state goes on its journey, information gets added or removed to the cell state via gates. The gates are different neural networks that decide which information is allowed on the cell state. The gates can learn what information is relevant to keep or forget during training.

Fig. 3.   LSTM structure
   There are several architectures of LSTM units. A common architecture is composed of a cell (the memory part of the LSTM unit) and three "regulators", usually called gates: an input gate, an output gate and a forget gate. Some variations of the LSTM unit do not have one or more of these gates or maybe have other gates.
The input gate controls the extent to which a new value flows into the cell, the forget gate controls the extent to which a value remains in the cell and the output gate controls the extent to which the value in the cell is used to compute the output activation of the LSTM unit. The activation function of the LSTM gates is often the logistic function.
A sigmoid activation is similar to the tanh activation. Instead of squishing values between -1 and 1, it squishes values between 0 and 1. That is helpful to update or forget data because any number getting multiplied by 0 is 0, causing values to disappears or be “forgotten.” Any number multiplied by 1 is the same value therefore that value stays the same or is “kept.” The network can learn which data is not important therefore can be forgotten or which data is important to keep.


FORGET GATE
     After getting the output of previous state, h(t-1), Forget gate helps to take decisions about what must be removed from h(t-1) state and thus keeping only relevant stuff. It is surrounded by a sigmoid function which helps to crush the input between [0,1]. 

Fig.4 Forget gate
We multiply forget gate with previous cell state to forget the unnecessary stuff from previous state which is not needed anymore.
 
INPUT GATE
   Input gate,  decides the extent to which the new stuff from the present input is added to the  present cell state .Sigmoid layer decides which values to be updated and tanh layer creates a vector for new candidates to added to present cell state.

Fig.5 Input gate and gate-gate
   The present cell state is calculated by adding the output of ((input gate*gate gate) and forget gate).
OUTPUT GATE
  Output gate decides what to output from the cell state ,which will be done by the sigmoid function. The input is multiplied with tanh to crush the values between (-1, 1) and with the output of sigmoid function ,ensures outputting required one.
 Fig.6 output gate 

 METHOD 1
CNN MODEL USED			: INCEPTION V3, AVERAGE POOL LAYER.
LSTM INPUT SIZE			: SEQUENCE LENGTH, 2048.
SPECIAL ADD ON FEATURE  	:          
OTHER CHANGES			:
STATUS				: FAILED.
ERROR				: RESOURCE EXHAUSTED.
ANALYSIS   				: MORE RESEARCH REQUIRED FOR THE METHOD IN            
                                                             WHICH THE  DATA IS BEING PROVIDED FOR THE     
                                                             NETWORK 


METHOD 2
CNN MODEL USED			: INCEPTION V3, AVERAGE POOL LAYER.
LSTM INPUT SIZE			: SEQUENCE LENGTH,2048.
SPECIAL ADD ON FEATURE  	:
OTHER CHANGES			: REDUCED BATCH SIZE AND REQUIRED CLASSES 
                                                             FROM 16 TO 10 AND 9 TO 3 RESPECTIVELY.          
STATUS   				: FAILED.
ERROR				: RESOURCE EXHAUSTED.
ANALYSIS				: PROVIDING THE INPUT AS VECTORS OF IMAGE   	 	 	 	                          SEQUENCES PROVED WRONG SINCE THEY HAVE 
                                                             HIGHER DIMENSIONALITY AND CANNOT BE 
                                                             TRAINED SUCCESSFULLY,EVEN AFTER CHANGING
                                                             THE BATCH SIZE AND  EPOCHS.INSIGHT TO THE  
                                                             METHOD OF FEATURE VECTOR ARRAY FOR   
                                                             TRAINING.


                                                           
METHOD 3
CNN MODEL USED			: FACE CLASSIFICATION MODEL, FLATTEN1 LAYER.
LSTM INPUT SIZE			: SEQUENCE LENGTH, 2450.
SPECIAL ADD ON FEATURE 	: GPU SUPPORT.
OTHER CHANGES			: CuDNN LSTM.
STATUS				: FAILED.
ERROR				: RESOURCE EXHAUSTED.
ANALYSIS				: CHANGING THE CNN MODEL OR LSTM DIDN’T 
                                                             BRING ABOUT ANY ADVANCEMENT



METHOD 4
CNN MODEL USED			: FACE CLASSIFICATION MODEL, FLATTEN1 LAYER.
LSTM INPUT SIZE			: SEQUENCE LENGTH, 2450.
SPECIAL ADD ON FEATURE 	: GPU SUPPORT
OTHER CHANGES			: BACK TO LSTM, BATCH SIZE REDUCED TO 5.
STATUS				: FAILED
ERROR				: RESOURCE EXHAUSTED
ANALYSIS				: REDUCTION IN BATCH SIZE HAS SHOWN NO 
                                                             EFFECT.



METHOD 5
CNN MODEL USED			: FACE CLASSIFICATION MODEL, ACTIVATION3 
                                                             LAYER.
LSTM INPUT SIZE			: SEQUENCE LENGTH, 500.
SPECIAL ADD ON FEATURE 	: GPU SUPPORT
OTHER CHANGES			:
STATUS				: SHOWED AN ACCURACY OF 10% FOR 10 EPOCHS 
                                                             OF TRAINING.
ERROR				:
ANALYSIS				: ACCURACY WAS TOO LOW FOR TESTING






METHOD 6
CNN MODEL USED			: FACE CLASSIFICATION MODEL, ACTIVATION3 
                                                             LAYER.
LSTM INPUT SIZE			: SEQUENCE LENGTH, 500.
SPECIAL ADD ON FEATURE 	: GPU SUPPORT
OTHER CHANGES			: BATCH SIZE INCREASED TO 10.
STATUS				: ACCURACY = 20, EPOCHS=178
ERROR				:
ANALYSIS				: ACCURACY SEEMS TO BE NOT CONSISTENT EVEN 
                                                             AFTER 178 EPOCHS, HENCE THE PROGRAM WAS 
                                                             STOPPED BEFORE REACHING 1000 EPOCHS.



METHOD 7
CNN MODEL USED			: FACE CLASSIFICATION MODEL, ACTIVATION3 
                                                             LAYER.
LSTM INPUT SIZE			: SEQUENCE LENGTH, 500.
SPECIAL ADD ON FEATURE 	: GPU SUPPORT
OTHER CHANGES 			: BATCH SIZE AGAIN REDUCED TO 1, SELF CREATED 
                                                             DATASET WITH FOUR CLASSES
STATUS				: ACCURACY = 43,
ERROR				: BIASED OUTPUT.
ANALYSIS				: ACCURACY STILL NOT CONSISTENT AFTER 50 
                                                             EPOCHS, LEAVING WITH A BIASED OUTPUT


TRAINING LOSS AND ACCURACY  
                                                           





METHOD 8
CNN MODEL USED			: MOBILENET V3,AVERAGE POOL LAYER.
LSTM INPUT SIZE			: SEQUENCE LENGTH, 1280.
SPECIAL ADD ON FEATURE 	: GPU SUPPORT
OTHER CHANGES                           : SELF CREATED DATASET WITH FOUR CLASSES
STATUS				: ACCURACY = 81
ERROR				: BIASED OUTPUT FOR 50 EPOCHS
ANALYSIS				: CHANGING THE CNN MODEL INCREASED THE 
                                                             ACCURACY, ALSO INCREASING LSTM INPUT SIZE 
                                                             DIDN’T AFFECT THE ACCURACY .


TRAINING LOSS AND ACCURACY

                                                           



METHOD 9
CNN MODEL USED			: MOBILENET V3, AVERAGE POOL LAYER.
LSTM INPUT SIZE			: SEQUENCE LENGTH, 1280.
SPECIAL ADD ON FEATURE 	: GPU SUPPORT
OTHER CHANGES			: SELF CREATED DATASET WITH FOUR CLASSES
STATUS				: ACCURACY = 100
ERROR				: BIASED OUTPUT, FOR 100 EPOCHS
ANALYSIS				: INCREASING THE EPOCHS HAVE INCREASED THE 
                                                             ACCURACY, BUT STILL PROVIDES WITH BIASED 
                                                             OUTPUT. THUS UNDERSTOOD THAT THE AND THE 
                                                             FEATURE VECTOR NEED TO BE INCREASED .





TRAINING LOSS AND ACCURACY




METHOD 10
CNN MODEL USED			: MOBILENET V3, AVERAGE POOL LAYER.
LSTM INPUT SIZE			: SEQUENCE LENGTH, 1280.
SPECIAL ADD ON FEATURE 	: GPU SUPPORT
OTHER CHANGES			: DATASET FROM UCF101 ,WITH 9 CLASSES.
STATUS				: ACCURACY = 98
ERROR				:
ANALYSIS				: THUS PROVIDING WITH A BETTER DATASET 
                                                             INCREASED THE QUALITY OF FEATURE VECTOR 
                                                             GIVEN TO THE NETWORK,ENABLING THE SYSTEM 
                                                             TO  IDENTIFY THE ACTION.
                                               

TRAINING LOSS AND ACCURACY



 Initially the system is tried to be implemented using Vgg16 CNN and LSTM. Instead of .h5 video data file,videos from UCF101 belonging to different classes are grouped into respective individual folders with labels. From each video,30 frames has been abstracted and made into an array.All arrays are then appended to form another single array. How to input the data to the time distributed layer was unknown. The information sources describing it were also not so readily available. So a similar procedure used during image classification problem was followed to input training data, causing errors like “resource exhausted” due to the higher dimensionality of input vector. Thus the 30 frames array appending method proved wrong, leading to opt another method.

  From further research and understanding, the importance of sequence length came into picture. In this method a sequence length and a maximum sequence length has been given initially and a csv data file is created under partition:class,video name,frames. The split training data and the test data has been read frame by frame up to the range of sequence length given, 30. The features were extracted from the resized and appended frames, and with its corresponding labels it is stored on to two arrays respectively.  Finally both the arrays are stored as an h5py file. Even though the procedure for giving video data input (2048),again leaving the same error.

   One of the method known to eliminate this is to reduce batch size and required classes, but it was not  much effective to solve the problem.  Since implementing this method is beyond the computing capability of the system in hand, the need for GPU support became compulsory. Also in order to rectify the error “resource exhausted” face classification model,from the previous project has been used instead of inception. The final feature output of the face classification model has been taken to extract the features, along with CuDNN LSTM (2450)which ought to make the processing of LSTM to be run on GPU. Even with a reduction in batch size, the error remained the same. So a change in the output length (500) of CNN is introduced along with taking the activation3 layer (flatten layer) of face classification model, resulting in a low accuracy system.

    Further analysis and experimentation was executed  with a self-created head posing datasets with four classes: “head right”, “head left”, and “head down”, “head up”,five videos from each class, but the system gives a biased output .On changing the feature extracting model, to MobileNetV2 the system responded with the same biased output with increased accuracy. 

    As the  dataset and the feature vector provided to the network is  small, the UCF101 dataset with 9 classes have been given to the network , leaving with a training accuracy of 98%and validation accuracy of 74%. Thus the problem of biased output has been rectified by using appropriate dataset along with increased accuracy enabling the system to recognise action.




                                                     




