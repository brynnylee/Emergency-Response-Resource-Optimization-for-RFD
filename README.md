# Emergency Response & Resource Optimization for City of Rochester Fire Department

❗️**Notice**: The dataset and code files associated with this project are classified as confidential government assets and cannot be publicly disclosed.


## *Enhancing Efficiency and Strategic Resource Allocation Through Data Science*  
*Project conducted for the City of Rochester Fire Department (RFD) using historical data analysis, predictive modeling, and geospatial insights to improve emergency response capabilities.*

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

The Rochester Fire Department (RFD) serves over 211,000 residents across 15 stations, responding to approximately 40,000 service calls annually. These calls range from fire suppression and hazardous material handling to emergency medical services (EMS). As urban populations grow and incident types diversify, the department faces increasing challenges in maintaining optimal response times and ensuring equitable service delivery across all neighborhoods.

### **Journey to Collaboration: How It All Began**
Recognizing the need for data-driven solutions, the RFD sought collaboration with a dedicated team of data scientists to transform its vast operational data into actionable insights. I was honored to be selected as the team lead for this pivotal project, working as part of a multidisciplinary group of five data scientists over the course of 4 months. This collaboration provided access to an extensive dataset comprising millions of records, as well as regular guidance from the RFD’s experienced data analysts. Weekly meetings with key stakeholders helped align our work with the department’s operational goals and allowed us to refine our approach using their domain expertise and advanced geospatial tools.

### **Key Milestones in Collaboration**:
- **Data Access and Preparation**: Received comprehensive incident data spanning nearly two decades (19 years), along with station-level geospatial information.
- **Stakeholder Engagement**: Conducted weekly review sessions with RFD’s planning team to incorporate real-world operational context.
- **Iterative Model Development**: Implemented and fine-tuned predictive models based on RFD’s evolving needs and feedback.

### Project Objectives:
- **Historical Analysis**: Identify key trends and patterns in past incident data to understand historical performance and incident distribution.
- **Predictive Modeling**: Develop robust time-series forecasting models to predict future incident counts across all fire stations.
- **Resource Optimization**: Provide actionable recommendations to improve response efficiency and support strategic resource reallocation.

The primary objective of this project was to enhance the operational efficiency of the RFD by leveraging data science methodologies to improve response times, optimize resource deployment, and ultimately enhance public safety across Rochester.



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
2. **Feature Engineering**:
   - Created lag features to capture temporal dependencies in the data.
   - Applied one-hot encoding for categorical variables (e.g., incident type, shift schedule).
   - Created interaction terms (e.g., `latitude * longitude`) to enhance spatial relationships in modeling.

**Handling External Events**:  
A binary variable (`is_covid`) was added to indicate whether an incident occurred during the pandemic period (March 2020–June 2021). This helped isolate the impact of COVID-19 on incident trends.


### 4.2 Exploratory Data Analysis (EDA)  

Key findings during EDA included:
- **Monthly Trends**: Incident counts show seasonal spikes, particularly during summer months due to outdoor events.
- **Incident Type Analysis**: Rescue & EMS incidents accounted for more than 60% of total calls.
- **Station Workload**: Station `Engine 17/Rescue 11` consistently reported the highest incident volumes, highlighting disparities in workload distribution.

#### Visual Insights:
- **Response Time Distribution**: A clear gap was observed between average and 90th percentile response times.
- **Incident Type by Time of Day**: EMS incidents peaked between 4 PM and 6 PM, aligning with commuting hours.

**Visualization Tools**:  
- **Map1: Average Response Time by ZIP Code (Screenshot)** 
<img width="1568" alt="Screenshot 2025-01-03 at 4 02 49 PM" src="https://github.com/user-attachments/assets/1edffcef-8b42-4cb9-9cc2-22924f1774ea" />

<img width="1570" alt="Screenshot 2025-01-03 at 4 03 01 PM" src="https://github.com/user-attachments/assets/88e35e5d-a838-4783-a210-457401078619" />

- **Map2: Incident Type Distribution and Fire Station Workload (Screenshot)**  
<img width="1569" alt="Screenshot 2025-01-03 at 4 03 59 PM" src="https://github.com/user-attachments/assets/f9be8564-4b81-4f84-9e85-9ce4e02e9e44" />


### 4.3 Model Development  

The modeling process followed these key stages:  

#### 4.3.1 Handling COVID-19 Effects
A binary indicator (`is_covid`) was created to control for pandemic-related anomalies. Hypothesis tests (t-tests and Mann-Whitney U tests) evaluated the impact of COVID-19 on key features.  

#### 4.3.2 Feature Selection  
- **Random Forest & SHAP Analysis**: Identified top contributing features to improve model interpretability.
- Interaction terms (e.g., `civilian deaths * alarm level`) highlighted critical dependencies in high-severity incidents.

### 4.4 Forecasting Incident Trends  

The **Prophet Model** was used to forecast monthly incident counts from 2024 to 2034:
- Integrated external regressors such as `population density` and `seasonality indicators`.
- Performance Metrics:
  - R² Score: 85.28%
  - Mean Absolute Error (MAE): 0.34
  - MAPE: 19.2%

#### Predicted Incident Density:  
- 2025:
  <img width="1219" alt="map_predictedCounts_2025" src="https://github.com/user-attachments/assets/3302f6e4-ee8d-47b6-8779-bbdadeb9cdba" />

- 2033:
  <img width="1219" alt="map_predictedCounts_2033" src="https://github.com/user-attachments/assets/c1b55b4c-d65b-475c-849b-98677077cd92" />

---

## 5. Results & Deliverables  

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
