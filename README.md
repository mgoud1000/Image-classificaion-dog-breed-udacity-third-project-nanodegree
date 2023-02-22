# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. 

In this project I will train a pre trained Resnet50 neural network to identify dog breeds based on the udacity-provided dogbreed image dataset.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Run through the appropriately named sections in "train_and_deploy.ipynb" to get the desired output. 

## Dataset
I'm using the dataset supplied to this udacity project which is the dog breed classification dataset. This dataset is comprised of 133 breeds or classes. Some examples of those classes are Affenpinschers, Akitas, Basset hounds,and Poodles. There are approximately 8300 training images,8300 test images, and 8300 validation images.

### Access
For this project the data needs to be uploaded to a bucket called "dogbreedclassificationudacity". The steps for uploading are in the notebook "train_and_deploy.ipynb"

### AWS Image Settings for Notebook train_and_deploy.ipynb.
Pytorch Framework = 1.6 </br>
Python Version = 3.6</br>
## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
I chose a ResNet50 model for model training because pre-trained model is trained on a different task than the task at hand and provides a good starting point since the features learned on the old task are useful for the new task. 
Remember that your README should:
### Hyper Parameter Optimization Jobs
[!hyperparamters_tunning_in_progress](https://github.com/mgoud1000/Image-classificaion-dog-breed-udacity-third-project-nanodegree/blob/main/hyperparameter%20training%20in%20progress.png)
[!finished_hyperparameters_jobs_done]((https://github.com/mgoud1000/Image-classificaion-dog-breed-udacity-third-project-nanodegree/blob/main/hyperparameter%20training%20in%20progress.png),(https://github.com/mgoud1000/Image-classificaion-dog-breed-udacity-third-project-nanodegree/blob/main/Amazon%20SageMaker%20-%20Google%20Chrome%202_20_2023%205_06_05%20PM.png),(https://github.com/mgoud1000/Image-classificaion-dog-breed-udacity-third-project-nanodegree/blob/main/hyperparameter%20tuning%20complete.png))

I chose to tune lr (learning rate), batch_size, and epochs, because they seemed to me like they would have the largest impact on training speed and accuracy. I kept the epochs range rather small in order to avoid large cost accumulation during the course of this project. The best hyperparameters retrieved were the following:

Best Hyperparamters post Hyperparameter fine tuning are : 
 {'batch_size': 128, 'lr': '0.003253222530647985', 'weight_decay': '0.027474039481333187'}
### Model Training Jobs
[!model_training_jobs](https://github.com/mgoud1000/Image-classificaion-dog-breed-udacity-third-project-nanodegree/blob/main/training-best-hyperparameter.png)
[!model_training_jobs](https://github.com/mgoud1000/Image-classificaion-dog-breed-udacity-third-project-nanodegree/blob/main/trining-hyperparaterms-completed.png)
[!model_training_jobs](https://github.com/mgoud1000/Image-classificaion-dog-breed-udacity-third-project-nanodegree/blob/main/Amazon%20SageMaker%20-%20Google%20Chrome%202_21_2023%208_14_17%20PM.png)

Above is a screenshot of my model training jobs with the final one that was deployed circled in red.
[!model_screen_shots](https://github.com/mgoud1000/Image-classificaion-dog-breed-udacity-third-project-nanodegree/blob/main/Amazon%20SageMaker%20-%20Google%20Chrome%202_21_2023%208_13_53%20PM.png)

## Debugging and Profiling
I performed model debugging and profiling in Sagemaker by using the "smdebug" package. Through this package we can monitor the performance of the training, called debugging, as well as how effectively we utilize the resources of the instance, called profiling. We implement these tools by putting smdebug "hooks" into our functions contained in train_model.py. These hooks then report back to the overall smdebug debugging and profiling process. You can tell the debugger what are the important metrics to follow while training, called rules. The rules I used were the following:
rules = [
    Rule.sagemaker(rule_configs.loss_not_decreasing()),
    ProfilerRule.sagemaker(rule_configs.LowGPUUtilization()),
    ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    Rule.sagemaker(rule_configs.overfit()),
    Rule.sagemaker(rule_configs.overtraining()),
    Rule.sagemaker(rule_configs.poor_weight_initialization()),
]

html from profiler is included


### Results

It seems like there is anomalous behavior in my debugging output. Some examples are below.

I have an overfitting error. Overfitting can be fixed in a few ways. I can decrease the size of my network down from a resnet50 to a resnet 18, for example. I can also augment the data that I'm using for the project.
I have high GPU utilization. I could avoid this in the future by either increasing batch size or using a smaller instance.

I also learned that most of my training time is not spent in my training and validation steps of my modeling cycle. Ideally most of the training time should be spent there. That means I should optimize the data loading and outputting portions of my code.

We had set the Debugger hook to record and keep track of the Loss Criterion metrics of the process in training and validation/testing phases. The Plot of the Cross entropy loss is shown below:

## Model Deployment

The endpoint exposes the last training job I ran on my resnet50 model. In order to deploy the model I found that I had to generate a custom inference.py script which is included with this submission.
You'll need to invoke the endpoint give the endpoint "data".

[!endpoint_inferences](https://github.com/mgoud1000/Image-classificaion-dog-breed-udacity-third-project-nanodegree/blob/main/Amazon%20SageMaker%20-%20Google%20Chrome%202_21_2023%208_13_39%20PM.png)
