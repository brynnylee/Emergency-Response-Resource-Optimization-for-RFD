# Emergency Response & Resource Optimization for City of Rochester Fire Department

❗️**Notice**: The dataset and code files associated with this project are classified as confidential government assets and cannot be publicly disclosed.


## *Enhancing Efficiency and Strategic Resource Allocation Through Data Science*  
*An advanced project leveraging data science, predictive modeling, and geospatial analysis to optimize emergency response for the City of Rochester Fire Department (RFD).*


---

##  **Table of Contents**
1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Key Concepts and Background](#3-key-concepts-and-background)
4. [Methodology](#4-methodology)
    - [Data Cleaning & Preparation](#41-data-cleaning--preparation)
    - [Exploratory Data Analysis (EDA)](#42-exploratory-data-analysis-eda)
    - [Model Development](#43-model-development)
    - [Feature Selection](#44-feature-selection)
    - [Forecasting Incident Trends](#45-forecasting-incident-trends)
5. [Results & Deliverables](#5-results--deliverables)
6. [Interactive Maps](#6-interactive-maps)
7. [Conclusion & Future Work](#7-conclusion--future-work)
8. [References](#8-references)

---

## 1. Introduction  

The Rochester Fire Department (RFD) serves over 211,000 residents across 15 stations, responding to approximately 40,000 service calls annually. These calls encompass fire suppression, hazardous material handling, and emergency medical services (EMS). As urban populations grow and incident types diversify, the department identified a critical need for a data-driven strategy to maintain optimal response times and ensure equitable service delivery across all neighborhoods. Additionally, they sought a detailed supporting analysis to strengthen their request to government stakeholders for necessary resources and policy support.

### **Journey to Collaboration: How It All Began**
Recognizing the need for data-driven solutions, the RFD sought collaboration with a dedicated team of data scientists to transform its vast operational data into actionable insights. I was honored to be selected as the team lead for this pivotal project, working as part of a multidisciplinary group of five data scientists over the course of 4 months. This collaboration provided access to an extensive dataset comprising millions of records, as well as regular guidance from the RFD’s experienced data analysts. Weekly meetings with key stakeholders helped align our work with the department’s operational goals and allowed us to refine our approach using their domain expertise and advanced geospatial tools.

### **Key Milestones in Collaboration**:
- **Data Access and Preparation**: Received comprehensive incident data spanning nearly two decades (19 years), along with station-level geospatial information.
- **Stakeholder Engagement**: Conducted weekly review sessions with RFD’s planning team to incorporate real-world operational context.
- **Iterative Model Development**: Implemented and fine-tuned predictive models based on RFD’s evolving needs and feedback.

### Project Objectives:
The primary objective of this project was to enhance the operational efficiency of the RFD by leveraging data science methodologies to improve response times, optimize resource deployment, and ultimately enhance public safety across Rochester.

- **Historical Analysis**: Identify key trends and patterns in past incident data to understand historical performance and incident distribution.
- **Predictive Modeling**: Develop robust time-series forecasting models to predict future incident counts across all fire stations.
- **Resource Optimization**: Provide actionable recommendations to improve response efficiency and support strategic resource reallocation.




---

## 2. Problem Statement  

The project focuses on the following objectives:
1. **Response Time Optimization**: Analyze historical incident data to identify patterns affecting response times and optimize personnel and equipment distribution.
2. **Resource Reallocation**: Recommend reallocation of specialized units based on demand hotspots.
3. **Predictive Forecasting**: Develop a model to forecast monthly incident counts across all stations for the next 10 years.
4. **Low-Acuity Response Program**: Assess feasibility for a low-acuity response initiative aimed at reducing non-emergency EMS strain.

---

## 3. Key Concepts and Background  

### **Key Models and Techniques**  

#### **Prophet Model (Time-Series Forecasting)**  
- Developed by Facebook, the Prophet model is designed for time-series data with strong seasonal components.
- **Strengths**: Automatically handles missing data, outliers, and holiday effects.
- **Mathematical Formulation**:  
  \[
  y(t) = g(t) + s(t) + h(t) + \epsilon_t
  \]
  where \(g(t)\) models the overall trend, \(s(t)\) captures seasonality, and \(h(t)\) includes holidays or external events.

#### **SARIMAX Model (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables)**  
- A statistical model capable of handling seasonality and external factors (exogenous variables) such as temperature or COVID-19 events.
- **Key Components**:
  - \(p\): Order of the autoregressive part.
  - \(d\): Degree of differencing.
  - \(q\): Order of the moving average.
  - Seasonal components (P, D, Q).

#### **SHAP (SHapley Additive exPlanations) Values**  
- Based on cooperative game theory, SHAP values explain how much each feature contributes to a prediction.
- Provides interpretability, helping stakeholders understand why the model predicts certain outcomes.


**Why Forecasting Matters**:  
Predictive models help preemptively allocate resources, reducing lag in responses during peak periods. This project integrates domain knowledge and statistical rigor to provide both long-term forecasts and immediate operational insights.

---

## 4. Methodology  

### 4.1 Data Cleaning & Preparation  

**Data Sources**:  
- **Incident Reports (2006–2024)**: Details on location, type, response times, and personnel.
- **Station Apparatus Data**: Information on vehicle types and units stationed at different locations.
- **Geospatial Data**: Shapefiles containing the geographic locations of fire stations and incident sites.

**Key Steps**:
1. **Missing Data Imputation**:
   - Location-based missing values (e.g., ZIP codes) were imputed using mode imputation.
   - Time-related gaps in `response_time` were filled by calculating median time differentials between `alarm_time` and `arrival_time`.

   **Why Imputation Matters**: Missing data can introduce bias or reduce model accuracy. Proper imputation techniques ensure the dataset remains comprehensive and suitable for machine learning models without distorting the underlying trends.

    
2. **Feature Engineering**:
   - Created **lag features** to capture temporal dependencies in the data, crucial for time-series analysis.
   - Applied **one-hot encoding** for categorical variables (e.g., incident type, shift schedule).  
   - Created **interaction terms** to enhance spatial relationships in modeling:
     - Example: \( \text{latitude} \times \text{longitude} \) interaction terms capture geospatial nuances.
     - Severity interaction: \( \text{civilian deaths} \times \text{alarmnum (severity level 5)} \) for severe incidents.

   **New Formula Example**:
   \[
   \text{Interaction Term}_{\text{severity}} = \text{Civilian Deaths} \times \text{Severity Level Indicator}
   \]
   This helps the model capture non-linear effects of incidents that involve human casualties.

3. **External Events Handling**:  
   A binary variable (`is_covid`) was added to indicate whether an incident occurred during the pandemic period (March 2020–June 2021). This variable helps the model account for anomalies caused by the pandemic, such as changes in call volumes and response times. Statistical significance was tested using t-tests and Mann-Whitney U tests based on the distribution of the data.



### 4.2 Exploratory Data Analysis (EDA)  

EDA plays a critical role in identifying trends, distributions, and potential anomalies:

**Key Findings**:
1. **Monthly Trends**:  
   - Incident counts peak during summer (June–August), with an average increase of 17.38% compared to winter months.
   - The graph below shows that events like the Rochester International Jazz Festival and July 4th celebrations correlate with demand surges.

   **Graph (Monthly Incident Records)**:  
   
   <img width="758" alt="Screenshot 2025-01-03 at 4 42 29 PM" src="https://github.com/user-attachments/assets/f789fb76-5f5b-4987-b83d-9193d74706db" />
   
   The red dashed line in the graph represents the average monthly trend from 2006 to 2024. Significant spikes coincide with major public events, such as the Rochester Jazz Festival and Fourth of July celebrations.

2. **Incident Type Distribution**:  
   - **Rescue & EMS** incidents account for the majority (~60%) of all calls, with notable increases during weekends and holidays.
   - The average summer Rescue & EMS calls reached **1,291.4** per month.
   - The chart below emphasizes the dominance of Rescue & EMS calls, especially in high-traffic areas.

   **Graph (Incident Types During Summer)**:
   <img width="860" alt="Screenshot 2025-01-03 at 4 43 33 PM" src="https://github.com/user-attachments/assets/f5dfec55-db5f-4e83-bbf5-699b4d85ea90" />

   During summer, the average number of Rescue & EMS incidents reached over 1,290 per month, followed by False Calls (330), Good Intent Calls (251), and Hazardous Conditions (241).

3. **Hourly Distribution**:  
   - Incident counts rise sharply between **4 PM to 6 PM**, aligning with commuting hours, and decrease between **4 AM to 5 AM**.

   **Graph (Hourly Incidents)**:  
   <img width="760" alt="Screenshot 2025-01-03 at 4 43 56 PM" src="https://github.com/user-attachments/assets/df3d7262-9af6-4b67-9443-5c11971f1e1d" />
   <img width="758" alt="Screenshot 2025-01-03 at 4 44 38 PM" src="https://github.com/user-attachments/assets/04f3c75d-81b3-428c-8495-4b6f2f1991ce" />


4. **Station Workload Analysis**:  
   - Stations like **Engine 17/Rescue 11**, **Engine 13/Truck 10**, and **Engine 2** reported the highest incident volumes.

   **Graph (Monthly Incidents by Station)**:  
   <img width="919" alt="Screenshot 2025-01-03 at 4 44 59 PM" src="https://github.com/user-attachments/assets/e7812cd9-33bc-479d-832a-5567c9288434" />


**Visualization Tools**:  
- **Map1: Average Response Time by ZIP Code (Screenshot)** 
<img width="1568" alt="Screenshot 2025-01-03 at 4 02 49 PM" src="https://github.com/user-attachments/assets/1edffcef-8b42-4cb9-9cc2-22924f1774ea" />

<img width="1570" alt="Screenshot 2025-01-03 at 4 03 01 PM" src="https://github.com/user-attachments/assets/88e35e5d-a838-4783-a210-457401078619" />

- **Map2: Incident Type Distribution and Fire Station Workload (Screenshot)**  
<img width="1569" alt="Screenshot 2025-01-03 at 4 03 59 PM" src="https://github.com/user-attachments/assets/f9be8564-4b81-4f84-9e85-9ce4e02e9e44" />


### 4.3 Model Development  

The modeling process followed these key stages:  

#### 4.3.1 Handling COVID-19 Effects
A **binary indicator (`is_covid`)** was created to control for pandemic-related anomalies. **Hypothesis tests (t-tests and Mann-Whitney U tests)** showed that key features were significantly impacted during this period, necessitating their inclusion as external regressors.

#### 4.3.2 Feature Selection  
- **Random Forest & SHAP Analysis**: Used to identify the top contributing features to the model's predictions.  
  - **Random Forest**: Provided importance scores for features.  
  - **SHAP Values**: Explained the contribution of individual features, enhancing interpretability for stakeholders.  
  - **Key Interaction Terms**:  
    - `latitude * longitude` captured spatial dependencies.  
    - `civilian deaths * alarm level` captured severity-related dependencies.

**Why SHAP is Important**: SHAP values improve transparency by showing the relative impact of each feature, which helps build trust with stakeholders.


### 4.4 Forecasting Incident Trends  

The **Prophet Model** was chosen for forecasting due to its ability to model seasonality, holidays, and long-term trends. This model is highly interpretable and capable of handling missing data and outliers without extensive preprocessing.

**Mathematical Formulation**:
\[
y(t) = g(t) + s(t) + h(t) + \epsilon_t
\]
- \( g(t) \): Trend function.
- \( s(t) \): Seasonal component.
- \( h(t) \): Holiday effects.
- \( \epsilon_t \): Error term.


**Advantages of Prophet**:
- Handles multiple seasonalities (e.g., weekly, yearly).
- Incorporates holiday effects and external regressors.
- Automatically adjusts for outliers and missing data, making it robust for real-world datasets.

### Performance Metrics
The model's performance was evaluated using the following metrics:
1. **Mean Absolute Error (MAE)**: 0.34  
2. **Mean Absolute Percentage Error (MAPE)**: 19.2%  
3. **R² Score**: 85.28%  
4. **RMSE**: Used to quantify large prediction errors.  


<img width="506" alt="Screenshot 2025-01-03 at 4 55 08 PM" src="https://github.com/user-attachments/assets/9a5ddb74-6e81-4632-b41a-cc2ad2513fe7" />


---

**Why Model Performance Metrics Matter**: 
- **R² Score**: Indicates how well the model fits the data.
- **MAE**: Measures the average magnitude of prediction errors, providing a direct interpretation in the original units.
- **MAPE**: Provides a percentage-based error, making it easier to compare performance across different datasets or variables.


#### Incident Forecasts:  
The forecast predicts trends for each station up to 2034:  
**Graphs (Monthly Incident Counts Forecasts)**:  
  - Monthly incident trends showed stations `Engine 10/Truck 2` and `Engine 13/Truck 10` as high-demand locations, while `Engine 8` showed significantly lower demand.

- 2025:
  <img width="1219" alt="map_predictedCounts_2025" src="https://github.com/user-attachments/assets/3302f6e4-ee8d-47b6-8779-bbdadeb9cdba" />

- 2033:
  <img width="1219" alt="map_predictedCounts_2033" src="https://github.com/user-attachments/assets/c1b55b4c-d65b-475c-849b-98677077cd92" />

### Summary of Methodology Effectiveness  
The combination of EDA, feature engineering, and time-series modeling enabled robust and interpretable predictions:
1. **Time-Series Features**: Lag variables captured temporal dependencies effectively.
2. **External Regressors**: Variables like population density added context to predictions.
3. **Interpretability with SHAP**: Provided transparency in feature contributions, essential for stakeholder trust.

These methodological choices ensured a balance between accuracy, interpretability, and scalability for real-world application.


---

## 5. Results & Deliverables  

## Monthly Incident Counts Forecast Across RFD Stations (2024-2034)
<img width="682" alt="Screenshot 2025-01-03 at 4 56 03 PM" src="https://github.com/user-attachments/assets/8f2037eb-5fc6-49b9-a903-1e44b48fe99c" />


### Key Insights:
- **High-Volume Stations**: `Engine 10/Truck 2` and `Engine 13/Truck 10` consistently handle the highest incidents.
- **Proposed Reallocation**: Moving hazardous condition units (e.g., `E17`) to stations closer to incident hotspots (`Engine 2`) could improve response efficiency.

### Proposed Low-Acuity Response Program:  
A $400,000 initiative modeled after Seattle's Health One program:
- Deploys social workers alongside firefighters for non-emergency EMS calls.
- Targets neighborhoods with high volumes of low-acuity calls to alleviate strain on emergency responders.

---

## 6. Interactive Maps  
**See the below html files from the repository:**
- Map1_Average_Response_Time_and_Incident_Analysis_by_ZIP_2019-2024.html
- Map2_Incident_Type_Distribution_and_Fire_Station_Workload_2019-2024.html
---

## 7. Conclusion & Future Work  

### Key Accomplishments:
- Accurate long-term forecasts support proactive resource allocation.
- Interactive visualizations enhance stakeholder engagement.
- Low-acuity program proposal addresses systemic inefficiencies and supports community health initiatives.

### Recommendations:
1. **Incorporate Traffic Data**: Enhancing response time predictions by integrating real-time traffic data.
2. **Interactive Dashboard**: Develop a dynamic reporting dashboard for real-time monitoring of incident trends.
3. **Collaboration with Urban Planners**: Strengthen partnerships to better align emergency services with city planning efforts.

### Future Enhancements:
- **Dynamic Allocation Models**: Investigate reinforcement learning methods for real-time resource optimization.
- **Community Outreach**: Implement educational programs to reduce false alarms and non-emergency calls.


---

## 8. References  

1. [Seattle’s Health One Program](https://www.seattle.gov/fire/safety-and-community/mobile-integrated-health/health-one)  
2. U.S. Fire Administration (FEMA) NFIRS Reference Guide: [NFIRS Guide](https://www.usfa.fema.gov/downloads/pdf/nfirs/nfirs_complete_reference_guide_2015.pdf)  

---

*Contributors*: Brynn (Ye In) Lee, Eugene Ayonga, Homayra Tabassum, Medhini Sridharr, Nour Assili  
*Sponsor*: City of Rochester (Rochester Fire Department)
