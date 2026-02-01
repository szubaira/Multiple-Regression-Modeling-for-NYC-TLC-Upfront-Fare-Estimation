## **Multiple Regression Modeling for NYC TLC Upfront Fare Estimation**

### **Project Overview**

This project focuses on developing a predictive regression model to provide New York City taxi riders with accurate, upfront fare estimates. By leveraging historical trip and fare data provided by the New York City Taxi & Limousine Commission (TLC), we built a machine learning solution that considers distance, time, and traffic patterns. The final model enables a transparent user experience, allowing riders to anticipate costs before beginning their journey.

### **Business Understanding**

The New York City Taxi & Limousine Commission (TLC) sought to bridge the competitive gap between traditional street-hail taxis and app-based For-Hire Vehicles (FHVs). A primary advantage of FHVs is "price certainty"—the ability for a customer to see a binding fare before booking. Research indicates that price transparency is a key determinant of customer satisfaction and trust in urban mobility.

**Stakeholder:** NYC Taxi & Limousine Commission (TLC).

**Business Problem:** The unpredictability of metered fares can lead to passenger hesitation and disputes. The objective was to transform unused historical data into a predictive tool that enhances the taxi industry’s service quality and operational transparency.

### **Data Understanding**

The analysis utilized historical NYC TLC trip record data, which includes detailed fields such as pickup/drop-off dates and times, trip distances, itemized fare components, and passenger counts.

* **Timeframe:** Historical data spanning multiple years was processed to capture seasonal and peak-hour trends.
* **Data Limitations:** The dataset primarily records credit card tips, meaning cash tips are underreported. Additionally, as the data is provided by third-party vendors, it may contain "noisy" entries such as negative fare amounts or outliers in GPS coordinates (e.g., trips listed outside of NYC), which required rigorous cleaning.
* **Exploratory Data Analysis (EDA):** Visualizations were created using Matplotlib and Seaborn to identify correlations between trip duration, distance, and the final fare, as well as the impact of "Rush Hour" surcharges on price volatility.

### **Modeling and Evaluation**

Using the PACE framework, several regression-based machine learning models were developed and tested using Scikit-learn.

* **Models Used:** Linear Regression, Decision Tree Regressor, and Ensemble Methods (Random Forest).
* **Evaluation Metrics:** The models were evaluated using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)** to measure the average dollar deviation from the actual metered fare.
* **Results:** The ensemble approach provided the most robust results, effectively handling the non-linear relationship between traffic delays and fare increases.
<img width="448" height="331" alt="Screen Shot 2026-02-01 at 15 36 48" src="https://github.com/user-attachments/assets/01e2a8f5-f06a-41b9-b409-3232001af244" />
<img width="447" height="195" alt="Screen Shot 2026-02-01 at 15 37 32" src="https://github.com/user-attachments/assets/0795384f-7457-43c6-93d4-c698f2806e33" />

### **Conclusion**

To solve the business problem of fare uncertainty, I recommend the full integration of the Random Forest regression model into a rider-facing application. This will allow the TLC to offer "Flex Fares" or binding quotes, which have been shown to increase driver hourly earnings by approximately 6% due to higher demand for transparent pricing.

**Future Steps:**

* **Feature Expansion:** Incorporate real-time weather data and local event schedules to improve accuracy during high-demand fluctuations.
* **Model Deployment:** Develop an API for real-time inference within the existing TLC-licensed e-hail apps.
