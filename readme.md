## MISE: Meta-knowledge Inheritance for Social Media-Based Stressor Estimation
Code and dataset of the paper:"MISE: Meta-knowledge Inheritance for Social Media-Based Stressor Estimation"

## News 🗞️

* **🔥 [2025/05/01] (update #2):** We public the new English version of the [stressor dataset](https://www.kaggle.com/datasets/xinwangcs/stressor-cause-of-mental-health-problem-dataset) in Kaggle. Let's go! 
* **🔥 [2025/04/29] (update #1):** We public the first version of the stressor dataset.  

## Dataset
### Context  
A *stressor* is a specific cause or trigger of an individual’s mental health issue, such as stress, anxiety, or suicidal thoughts. Identifying stressors is a key step toward understanding and improving mental well-being.

### Content  
This dataset contains **4,001 manually annotated online posts** collected from **June 2018 to June 2022**. Each sample includes the following fields:

- `text`: The content of the individual's post.  
- `labels`: The identified stressor(s) within the post, including their start and end positions (word-level).  
- `time_interval`: A numeric value (1-8) representing the half-year period in which the post was published.  
  - For example, `1` indicates the post was published between 2018-07-01 and 2018-12-31,  
    while `8` corresponds to 2022-01-01 to 2022-06-30.

### Data Partitioning  
By default, we divide the data based on time to simulate a real-world scenario:  1. Samples from `time_interval = 1` to `7` serve as training data (historical period). 2. Samples from `time_interval = 8` serve as test data (most recent half-year). This setup allows researchers to train models on past data and evaluate performance on future data.  

However, you are free to re-partition the dataset based on your specific task or experimental design.

### CUA
To receive access of code, you will need to read, sign, and send back the attached data and code usage agreement (CUA).

The CUA contains restrictions on how you can use the code. We would like to draw your attention to several restrictions in particular:

- No commercial use.

If your institution has issues with language in the CUA, please have the responsible person at your institution contact us with their concerns and suggested modifications.

Once the Primary Investigator has signed the CUA, the Primary Investigator should email the signed form to wangxin_6961@163.com

### Citation  
Wang, Xin, et al. *"MISE: Meta-knowledge Inheritance for Social Media-Based Stressor Estimation."* Proceedings of the ACM on Web Conference 2025.
