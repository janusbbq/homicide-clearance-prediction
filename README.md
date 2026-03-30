# 7.8: Detection of Homicide Patterns

By the end of this module assignment, you will develop a plan to use Machine Learning to detect trends and patterns in United States homicide data maintained by the Murder Accountability Project (MAP). We will merge your team's work and concepts into a master action plan for use in this course. Please follow the following guidelines and document any deviation your team feels a positive productive initiative:

1. Identify, download, and install a method to store the MAP dataset for Tennessee (Excel Format - with displayed coded variables & with only coded variables, (SHR76_22-BYNOVALSCSVTN.csv) & (SHR76_22-BYVALSCSVTN.csv) )
2. Read the database coding scheme (MAPdefinitionsSHR.pdf)
3. Outline your plan in the form of a Jupyter Notebook
4. Decide which Machine Learning (ML) tool you want to use to process the data (Excel spreadsheet, SPSS, Orange3, Python Toolkits)
5. Be prepared to present and merge your plan collectively with the entire class
6. Report back to the class in the discussion post your team's feedback on whether this GenAI tool is worthwhile.

------------------------------------------------

### **My Plan for 7.8: Detection of Homicide Patterns**

#### **1. Data Storage and Environment Method**
To fulfill the requirement to identify and install a method for storage, I have decided to use a local project directory where I've downloaded both versions of the Tennessee data. My primary engine for the analysis will be the **`SHR76_22-BYNOVALSCSVTN.csv`** file because it contains the raw coded variables needed for Machine Learning. I’ll keep the **`SHR76_22-BYVALSCSVTN.csv`** file nearby as a reference to verify that my decoded labels are correct.

**My Storage Method:** I will "install" these into my active workspace using the **Python Pandas** library. This allows me to store the data in a high-performance DataFrame for manipulation rather than just looking at a static Excel sheet.

#### **2. My Jupyter Notebook Outline**
Based on the class examples, I am structuring my Jupyter Notebook into these six distinct sections:

* **Step 1: Problem Statement** I want to see if we can use victim demographics and crime circumstances to predict whether an offender will be identified (the `WAS OFFENDER IDENTIFIED` column).
* **Step 2: Data Ingestion** I will use `pd.read_csv('SHR76_22-BYNOVALSCSVTN.csv')` to pull the Tennessee data into my notebook.
* **Step 3: Data Cleaning** Using the `MAPdefinitionsSHR.pdf`, I’ll identify and handle missing values. For instance, I need to account for codes like $999$ for unknown age or 'U' for unknown race so they don't skew my ML model.
* **Step 4: Exploratory Data Analysis (EDA)** I plan to use `seaborn` and `matplotlib` to plot homicide trends in Tennessee from 1976 to 2022 and check the distribution of weapons used.
* **Step 5: Feature Engineering** I’ll select the most relevant columns—like `VICTIM AGE`, `VICTIM SEX`, `WEAPON`, and `COUNTY NAME`—and convert them into a format the computer can understand (using techniques like One-Hot Encoding).
* **Step 6: Model Building** I will split the data into training and testing sets to build the final pattern detection model.

#### **3. My Choice of Machine Learning Tool**
I have decided to go with **Python Toolkits (Scikit-Learn)**. While I could use Excel or Orange3, Python is much better for this project because:
* It integrates perfectly with the Jupyter Notebook environment.
* It can handle the large volume of Tennessee records (dating back to 1976) much faster than Excel.
* It gives me more control over "Pattern Detection" algorithms like Random Forest or Logistic Regression.

#### **4. Class Discussion Feedback on GenAI**
For our team's report back to the class, I think the following feedback on using GenAI is worthwhile:
* **Efficiency:** The AI was incredibly helpful in quickly cross-referencing the `MAPdefinitionsSHR.pdf` dictionary with the headers in our CSV, which saved us a lot of manual lookup time.
* **Accuracy:** It helped us realize that the `BYNOVALS` file is the one we actually need for the ML algorithms, while the `BYVALS` file is just for us humans to read.
* **Value:** Overall, it’s a great "co-pilot" for setting up the technical framework of the notebook, allowing us to focus more on the actual crime pattern analysis.

---


