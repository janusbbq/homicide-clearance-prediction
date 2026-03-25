# Python Machine Learning Tutorial (Data Science) 
---

## 0. Goal of This Notebook
In this notebook, we follow the full beginner machine learning workflow shown in the video:

1. Understand what machine learning is
2. Set up Jupyter Notebook with Anaconda
3. Load data from CSV files with pandas
4. Inspect and understand a dataset
5. Build a simple music recommendation model
6. Split data into input/output sets
7. Train a Decision Tree model
8. Make predictions
9. Measure accuracy
10. Save the trained model
11. Export and visualize the decision tree

The video uses two datasets:
- `vgsales.csv` → only for learning how to load and inspect CSV data
- `music.csv` → for the actual machine learning project

---

## 1. What is Machine Learning?

Machine Learning is a subset of AI (Artificial Intelligence).  
Instead of writing lots of hard-coded rules, we give a computer many examples, and it learns patterns from data.

### Example from the video
Suppose we want a program to recognize whether an image contains:
- a cat
- a dog
- or later even a horse

With traditional programming, we would need to manually write many rules:
- detect edges
- detect curves
- detect colors
- handle angles
- handle lighting
- handle black-and-white photos

That quickly becomes too complex.

With machine learning:
- we build a model
- give it lots of example data
- let it learn patterns
- then ask it to predict new cases

The more good-quality data we provide, the better the model can become.

---

## 2. Standard Machine Learning Workflow

According to the video, a machine learning project usually follows these steps:

1. Import the data
2. Clean / prepare the data
3. Split the dataset into training and testing sets
4. Select an algorithm and create a model
5. Train the model
6. Make predictions
7. Evaluate accuracy
8. Improve the model if needed

Keep this sequence in mind. It is the backbone of the whole tutorial.

---

## 3. Tools and Libraries Used in the Video

The video introduces these Python tools/libraries:

- **NumPy** → multi-dimensional arrays and numerical work
- **pandas** → data analysis and DataFrame tables
- **matplotlib** → charts and plots
- **scikit-learn** → machine learning algorithms
- **Jupyter Notebook** → interactive environment for writing and running code cell by cell
- **Anaconda** → convenient platform that installs Python, Jupyter, and common data libraries together

---

## 4. Environment Setup (From the Video)

### Step 1: Install Anaconda
Go to the Anaconda website and install the Python 3 distribution for your operating system.

The video explains that Anaconda is convenient because it installs:
- Jupyter Notebook
- NumPy
- pandas
- and other popular data science libraries

So you do **not** need to manually install everything with `pip` at the beginning.

### Step 2: Start Jupyter Notebook
Open a terminal and run:

```python
jupyter notebook
```

This starts the notebook server and usually opens a browser automatically at something like:

```
http://localhost:8888
```

### Step 3: Create a New Notebook

In the Jupyter dashboard:

- navigate to the folder where you want to work
- click **New**
- choose **Python 3**

Rename the notebook to something meaningful.

------

## 5. Jupyter Basics from the Video

### Edit mode vs Command mode

A cell can be in:

- **Edit mode** → green bar, you can type inside the cell
- **Command mode** → blue bar, you can use notebook shortcuts

Press:

- `Enter` → go into Edit mode
- `Esc` → go into Command mode

### Useful shortcuts shown in the video

In **Command mode**:

- `A` → insert cell **above**
- `B` → insert cell **below**
- `D` `D` → delete the selected cell
- `H` → show keyboard shortcuts

### Important behavior

When you run one cell, only that cell runs.
Other cells do not automatically update unless you run them too.

------

## 6. Hello World in Jupyter

### Code Cell

```python
print("Hello World")
```

This is the first simple example shown in the video to demonstrate how notebook cells work.

------

## 7. Part A — Load and Inspect a CSV File (`vgsales.csv`)

This part is not the machine learning model yet.
It is just to learn how to load a CSV file and inspect it with pandas.

### Step 1: Put `vgsales.csv` next to your notebook

The video suggests placing the CSV file in the same folder as your notebook file.
If the CSV is not in the same folder, you must give the full path.

### Step 2: Import pandas and read the file

### Code Cell

```python
import pandas as pd

df = pd.read_csv("vgsales.csv")
df
```

### What this does

- `pd.read_csv(...)` reads a CSV file
- it returns a **DataFrame**
- a DataFrame is like a table / Excel spreadsheet with rows and columns

------

## 8. Inspect Basic Dataset Information

### 8.1 Shape of the dataset

### Code Cell

```python
df.shape
```

### Meaning

This returns:

- number of rows
- number of columns

The video explains that the video game dataset has over 16,000 rows and 11 columns.

------

### 8.2 Statistical summary

### Code Cell

```python
df.describe()
```

### Meaning of the output

For numeric columns, `describe()` gives:

- `count` → how many non-null values exist
- `mean` → average
- `std` → standard deviation
- `min` → minimum value
- `25%`, `50%`, `75%` → quartiles
- `max` → maximum value

### Example discussed in the video

If `year` has fewer values than `rank`, that means some rows are missing the `year` value.

This is an example of why data cleaning matters.

------

### 8.3 Raw values as arrays

### Code Cell

```python
df.values
```

### Meaning

This returns the table as a 2D array.

Each inner array corresponds to one row from the dataset.

------

## 9. Why Data Cleaning Matters

The video explains that before training a model, we may need to:

- remove duplicates
- remove irrelevant rows
- remove incomplete rows
- fill missing values
- convert text labels into numbers when needed

In the `vgsales.csv` dataset, the video only demonstrates inspection, not full cleaning.

------

## 10. Part B — Machine Learning Project: Music Recommendation

Now the video moves to the main project.

### Problem setup

We imagine an online music store.
When users sign up, we ask for:

- age
- gender

Then we want the model to predict which genre of music they are likely to like.

### Columns in `music.csv`

- `age`
- `gender`
- `genre`

### Meaning of `gender`

- `1` = male
- `0` = female

### Assumptions used in the video

These are made-up patterns, just for learning:

For men:

- 20–25 → hip hop
- 26–30 → jazz
- over 30 → classical

For women:

- 20–25 → dance
- 26–30 → acoustic
- over 30 → classical

------

## 11. Load `music.csv`

Put `music.csv` in the same folder as the notebook.

### Code Cell

```python
import pandas as pd

music_data = pd.read_csv("music.csv")
music_data
```

This loads the main dataset for the model.

------

## 12. Prepare the Data

The video says that this tiny dataset does not need much cleaning because:

- no duplicates are shown
- no null values are shown

But we still must separate the data into:

- **Input set (X)** → columns used to make predictions
- **Output set (y)** → the target column we want to predict

In this project:

- `X = [age, gender]`
- `y = genre`

### Code Cell

```python
X = music_data.drop(columns=["genre"])
y = music_data["genre"]

X
```

### Why `drop(columns=["genre"])`?

Because `genre` is the answer we want the model to learn to predict, so it should not be part of the input features.

### Code Cell

```python
y
```

------

## 13. Create the Model

The video uses a **Decision Tree Classifier**.

### Code Cell

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
```

### Why Decision Trees?

The video says decision trees are one of the easiest ML models to understand visually.

------

## 14. Train the Model

### Code Cell

```python
model.fit(X, y)
```

### Meaning

This is where the model learns patterns from:

- input features `X`
- correct outputs `y`

------

## 15. Make Predictions (First Simple Version)

The video first shows direct predictions before introducing train/test splitting.

### Predict for:

- a 21-year-old male → `[21, 1]`
- a 22-year-old female → `[22, 0]`

### Code Cell

```python
predictions = model.predict([[21, 1], [22, 0]])
predictions
```

### Expected interpretation

Given the assumptions in the dataset:

- 21-year-old male → likely `HipHop`
- 22-year-old female → likely `Dance`

(Exact capitalization depends on your CSV values.)

------

## 16. Why This Is Not Enough

The video explains that making a few predictions is **not enough** to know whether a model is actually good.

We need a better evaluation method:

- split the dataset into **training** and **testing** parts
- train on one part
- test on unseen data
- compare predictions to actual answers

------

## 17. Split Data into Training and Testing Sets

General rule mentioned in the video:

- 70–80% for training
- 20–30% for testing

The tutorial uses `test_size=0.2`, meaning:

- 80% train
- 20% test

### Code Cell

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)
```

### Meaning of outputs

- `X_train` → inputs for training
- `X_test` → inputs for testing
- `y_train` → expected outputs for training
- `y_test` → expected outputs for testing

------

## 18. Train the Model on Training Data Only

### Code Cell

```python
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```

------

## 19. Predict on the Test Set

### Code Cell

```python
predictions = model.predict(X_test)
predictions
```

These are the model’s predicted genres for the unseen test inputs.

------

## 20. Measure Accuracy

### Code Cell

```python
from sklearn.metrics import accuracy_score

score = accuracy_score(y_test, predictions)
score
```

### Meaning

`accuracy_score` compares:

- the correct answers: `y_test`
- the model predictions: `predictions`

It returns a value from `0` to `1`:

- `1.0` means 100% correct
- `0.8` means 80% correct

### Important note from the video

If you run the notebook again, the score may change because:

- `train_test_split()` randomly selects training/testing rows

------

## 21. Full Clean Training/Evaluation Version

This is the most useful combined version of the notebook up to this point.

### Code Cell

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv("music.csv")

X = music_data.drop(columns=["genre"])
y = music_data["genre"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)
score
```

------

## 22. Model Persistence (Save the Trained Model)

The video explains that in real applications:

- training can take time
- we do not want to retrain every time
- so we save the trained model to a file

### Important note

The video uses an older import style:

```python
from sklearn.externals import joblib
```

In modern environments, use:

```python
import joblib
```

### Code Cell

```python
import joblib

music_data = pd.read_csv("music.csv")
X = music_data.drop(columns=["genre"])
y = music_data["genre"]

model = DecisionTreeClassifier()
model.fit(X, y)

joblib.dump(model, "music-recommender.joblib")
```

This creates a binary file containing the trained model.

------

## 23. Load the Saved Model and Predict

Once a model is saved, you can load it later without retraining.

### Code Cell

```python
import joblib

model = joblib.load("music-recommender.joblib")
predictions = model.predict([[21, 1]])
predictions
```

You can also test multiple rows:

### Code Cell

```python
model = joblib.load("music-recommender.joblib")
predictions = model.predict([[21, 1], [22, 0], [35, 1], [29, 0]])
predictions
```

------

## 24. Export the Decision Tree to a `.dot` File

The video then exports the decision tree so we can visualize how it makes decisions.

### Code Cell

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

music_data = pd.read_csv("music.csv")

X = music_data.drop(columns=["genre"])
y = music_data["genre"]

model = DecisionTreeClassifier()
model.fit(X, y)

tree.export_graphviz(
    model,
    out_file="music-recommender.dot",
    feature_names=["age", "gender"],
    class_names=sorted(y.unique()),
    label="all",
    rounded=True,
    filled=True
)
```

### Meaning of important arguments

- `out_file="music-recommender.dot"` → save graph description to a dot file
- `feature_names=["age", "gender"]` → labels for input columns
- `class_names=sorted(y.unique())` → list of output classes
- `label="all"` → show useful labels in nodes
- `rounded=True` → rounded node corners
- `filled=True` → color-filled nodes

------

## 25. View the Tree in VS Code

The video suggests this process:

1. Open `music-recommender.dot` in VS Code
2. Install the extension:
   - **Graphviz (dot) language**
3. Use the preview feature to render the graph visually

This gives a tree diagram showing exactly how the model makes decisions.

------

## 26. How to Read the Decision Tree

The video explains the logic of the visualized tree like this:

### Example root rule

```
age <= 30.5
```

- If **False**, the user is older than 30
- then the predicted class becomes `Classical`

If **True**, the person is younger than about 30, so the model checks something else, such as:

```
gender <= 0.5
```

Since:

- `0 = female`
- `1 = male`

This means:

- if `gender <= 0.5` → female branch
- otherwise → male branch

Then the tree checks age again to separate:

- dance vs acoustic
- hip hop vs jazz

### Why numbers like `25.5` or `30.5`?

The video explains that these are learned split thresholds created by the model from the data.

------

## 27. Final “Video-Aligned” Notebook Version

If you want one clean notebook that follows the core ML project from start to finish, use this:

### Code Cell

```python
# Step 1: Import libraries
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

# Step 2: Load dataset
music_data = pd.read_csv("music.csv")

# Step 3: Separate input and output
X = music_data.drop(columns=["genre"])
y = music_data["genre"]

# Step 4: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# Step 5: Create model
model = DecisionTreeClassifier()

# Step 6: Train model
model.fit(X_train, y_train)

# Step 7: Make predictions on test data
predictions = model.predict(X_test)

# Step 8: Measure accuracy
score = accuracy_score(y_test, predictions)
print("Accuracy:", score)

# Step 9: Train on full data and save model
model.fit(X, y)
joblib.dump(model, "music-recommender.joblib")

# Step 10: Load model and predict new users
loaded_model = joblib.load("music-recommender.joblib")
new_predictions = loaded_model.predict([[21, 1], [22, 0], [35, 1], [29, 0]])
print("Predictions for new users:", new_predictions)

# Step 11: Export decision tree
tree.export_graphviz(
    loaded_model,
    out_file="music-recommender.dot",
    feature_names=["age", "gender"],
    class_names=sorted(y.unique()),
    label="all",
    rounded=True,
    filled=True
)

print("Decision tree exported to music-recommender.dot")
```

------

## 28. Recommended Notebook Cell Order

If you want your notebook to look clean and easy to study, organize it like this:

### Markdown Cell

```
# Python Machine Learning Tutorial Notes
```

### Markdown Cell

```
## 1. Setup and Goal
```

### Code Cell

```python
print("Notebook started")
```

### Markdown Cell

```
## 2. Load and Inspect vgsales.csv
```

### Code Cell

```python
import pandas as pd
df = pd.read_csv("vgsales.csv")
df.shape
```

### Code Cell

```python
df.describe()
```

### Code Cell

```python
df.values
```

### Markdown Cell

```
## 3. Music Recommendation Dataset
```

### Code Cell

```python
music_data = pd.read_csv("music.csv")
music_data
```

### Code Cell

```python
X = music_data.drop(columns=["genre"])
y = music_data["genre"]
```

### Code Cell

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X, y)
```

### Code Cell

```python
model.predict([[21, 1], [22, 0]])
```

### Markdown Cell

```
## 4. Train/Test Split and Accuracy
```

### Code Cell

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy_score(y_test, predictions)
```

### Markdown Cell

```
## 5. Save and Load Model
```

### Code Cell

```python
import joblib

model.fit(X, y)
joblib.dump(model, "music-recommender.joblib")

loaded_model = joblib.load("music-recommender.joblib")
loaded_model.predict([[21, 1]])
```

### Markdown Cell

```
## 6. Export the Decision Tree
```

### Code Cell

```python
from sklearn import tree

tree.export_graphviz(
    loaded_model,
    out_file="music-recommender.dot",
    feature_names=["age", "gender"],
    class_names=sorted(y.unique()),
    label="all",
    rounded=True,
    filled=True
)
```

------

## 29. Key Concepts You Should Understand After Finishing This Notebook

By the end of the video/notebook, you should understand:

- what machine learning means
- why ML is better than hard-coded rules for some problems
- why data must be cleaned/prepared
- what a DataFrame is
- how to load CSV data with pandas
- what input features and output labels are
- how to use a Decision Tree classifier
- what `fit()` does
- what `predict()` does
- why train/test split matters
- how to calculate model accuracy
- why model persistence is useful
- how to export and inspect a decision tree visually

------

## 30. One-Sentence Summary of the Whole Video

This video teaches the full beginner machine learning workflow in Python using Jupyter Notebook, pandas, and scikit-learn, ending with a working decision tree model that predicts music preferences from age and gender.

------

```

```