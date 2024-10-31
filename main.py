''' This code demonstrates a Dynamic Pricing Strategy analysis using Python '''
import numpy as np
import pandas as pd
import sys
import statsmodels
import plotly
import plotly.express as px
import plotly.graph_objects as go
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

class Dynamic_Pricing:
    def __init__(self, data):
        try:
            self.data = data
            print(f"Dynamic Pricing Data Set Head Means No.of Rows & Columns")
            print(self.data.head())
            print(f"Describe Function finds Means and Counts and Few Functions and values")
            print(self.data.describe())
            print(f"Info() function is shows the Details of the Data Set Dtypes Information")
            print(self.data.info())
            self.data1 = self.data.drop(
                ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type'], axis=1)
        except FileNotFoundError:
            print("Error: File not found.")
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f"Error from Line {err_line.tb_lineno} -> type {error_type.__name__} -> Error msg -> {error_msg}")

    def Ride_Duration_Historical_cost(self):
        try:
            fig = px.scatter(self.data,
                           x='Expected_Ride_Duration',
                           y='Historical_Cost_of_Ride',
                           title='Expected Ride Duration vs. Historical Cost of Ride',
                           trendline='ols'
                           )
            fig.show()
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f"Error from Line {err_line.tb_lineno} -> type {error_type.__name__} -> Error msg -> {error_msg}")

    def Ride_Historical_Cost_Vehicle_Type(self):
        try:
            fig = px.box(self.data,
                       x='Vehicle_Type',
                       y='Historical_Cost_of_Ride',
                       title='Hostorical Cost of Ride By Vehicle Type')
            fig.show()
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f"Error from Line {err_line.tb_lineno} -> type {error_type.__name__} -> Error msg -> {error_msg}")

    def Correlation_matrix_data(self):
        try:
            corr_matrix = self.data1.corr()
            fig = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                            x=corr_matrix.columns,
                                            y=corr_matrix.columns,
                                            colorscale='Viridis'))
            fig.update_layout(title='Correlation Matrix')
            fig.show()
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f"Error Line is {err_line.tb_lineno} -> Type is {error_type.__name__} -> Error msg is {error_msg}")
    def Implementation_Dynamic_Pricing(self):
        try:
            # Calculate demand_multiplier based on percentile for high and low demand
            high_demand_percentile=75
            low_demand_percentile=25
            self.data['demand_multiplier']=np.where(self.data['Number_of_Riders'] > np.percentile(self.data['Number_of_Riders'], high_demand_percentile),
                                     self.data['Number_of_Riders'] / np.percentile(self.data['Number_of_Riders'], high_demand_percentile),
                                     self.data['Number_of_Riders'] / np.percentile(self.data['Number_of_Riders'], low_demand_percentile))
            # Calculate supply_multiplier based on percentile for high and low supply
            high_supply_percentile = 75
            low_supply_percentile = 25

            self.data['supply_multiplier'] = np.where(
                self.data['Number_of_Drivers'] > np.percentile(self.data['Number_of_Drivers'], low_supply_percentile),
                np.percentile(self.data['Number_of_Drivers'], high_supply_percentile) / self.data['Number_of_Drivers'],
                np.percentile(self.data['Number_of_Drivers'], low_supply_percentile) / self.data['Number_of_Drivers'])
            # Define price adjustment factors for high and low demand/supply
            demand_threshold_high=1.2
            demand_threshold_low=0.8
            supply_threshold_high=0.8
            supply_threshold_low=1.2
            # Calculate adjusted_ride_cost for dynamic pricing
            self.data['adjusted_ride_cost']=self.data['Historical_Cost_of_Ride'] *(
                np.maximum(self.data['demand_multiplier'],demand_threshold_low)*np.maximum(self.data['supply_multiplier'],supply_threshold_high)
            )
        except Exception as e:
            error_type,error_msg,err_line=sys.exc_info()
            print(f"Error Line {err_line.tb_lineno} -> Error type{error_type} -> msg {error_msg}")
    def Profit_of_Riders(self):
        try:
            # Calculate the profit percentage for each ride
            self.data['profit_percentage']=((self.data['adjusted_ride_cost']-self.data['Historical_Cost_of_Ride'])/self.data['Historical_Cost_of_Ride'])*100
            # Identify profitable rides where profit percentage is positive
            profitable_rides = self.data[self.data['profit_percentage'] > 0]
            # Identify loss rides where profit percentage is negative
            loss_rides = self.data[self.data['profit_percentage'] < 0]
            #calculate the count of profitable and loss rides
            profitable_count=len(profitable_rides)
            loss_count=len(loss_rides)
            # Create a donut chart to show the distribution of profitable and l
            labels=['Profitable Rides','Loss Rides']
            values=[profitable_count,loss_count]
            fig=go.Figure(data=[go.Pie(labels=labels,values=values,hole=0.5)])
            fig.update_layout(title='Profitability of Rides (Dynamic Pricing vs Hostorical Pricing )')
            fig.show()
        except Exception as e:
            error_type,error_msg,err_line=sys.exc_info()
            print(f"Error type {error_type}-> Error line is {err_line.tb_lineno}  -> msg {error_msg}")
    def Expected_Ride_Cost_Ride(self):
        try:
            fig=px.scatter(self.data,
                           x='Expected_Ride_Duration',
                           y='adjusted_ride_cost',
                           title='Expected Ride Duration vs Cost of Ride',
                           trendline='ols')
            fig.show()
        except Exception as e:
            error_type,error_msg,err_line=sys.exc_info()
            print(f"Error Type :{error_type.__name__}-> Error Line {err_line.tb_lineno} -> msg {error_msg}")
    def Training_Predicted_Model(self):
        try:
            def data_preprocessing_pipeline(data):
                #Identify numeric and categorical features
                numeric_features=self.data.select_dtypes(include=['float','int']).columns
                categorical_features=self.data.select_dtypes(include=['object']).columns
                #handling numeric values in numeric features
                self.data[numeric_features]=self.data[numeric_features].fillna(self.data[numeric_features].mean())
                # Detect and handle outliers in numeric features using IQR
                for feature in numeric_features:
                    Q1=self.data[feature].quantile(0.25)
                    Q3=self.data[feature].quantile(0.75)
                    IQR=Q3-Q1
                    lower_bound=Q1-(1.5*IQR)
                    upper_bound=Q3+(1.5*IQR)
                    self.data[feature]=np.where((self.data[feature]<lower_bound)| (self.data[feature]>upper_bound),
                                           self.data[feature].mean(),self.data[feature])
                # Handle missing values in categorical features
                self.data[categorical_features]=self.data[categorical_features].fillna(self.data[categorical_features].mode().iloc[0])
                return  self.data
            data=data_preprocessing_pipeline(self.data)
            self.data["Vehicle_Type"]=self.data["Vehicle_Type"].map({"Premium":1,"Economy":0})
            # splitting data
            x=np.array(self.data[["Number_of_Riders","Number_of_Drivers","Vehicle_Type","Expected_Ride_Duration"]])
            y=np.array(self.data[["adjusted_ride_cost"]])
            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(x,y,test_size=0.2,random_state=42)
            #reshape y to 1D array
            self.y_train=self.y_train.ravel()
            self.y_test=self.y_test.ravel()
            # Training a random forest regression model
            self.model=RandomForestRegressor()
            self.model.fit(self.X_train,self.y_train)
            #train predicted values
            self.y_train_pred=self.model.predict(self.X_train)
            #test predicted values
            self.y_test_pred=self.model.predict(self.X_test)
            return self.model
        except Exception as e:
            error_type,error_msg,err_line=sys.exc_info()
            print(f"Error type {error_type.__name__}-> Error msg {error_msg} -> error line {err_line.tb_lineno}")
    def model_predictions(self):
        try:
            def get_vehicle_type_numeric(vehicle_type):
                vehicle_type_mapping={
                    "Premium":1,
                    "Economy":0
                }
                vehicle_type_numeric=vehicle_type_mapping.get(vehicle_type)
                return  vehicle_type_numeric
            # Predicting using user input values
            def predicted_price(number_of_riders,number_of_drivers,vehicle_type,Expected_Ride_Duration):
                vehicle_type_numeric=get_vehicle_type_numeric(vehicle_type)
                if vehicle_type_numeric is None:
                    raise ValueError("Invalid vehicle type")
                input_data=np.array([[number_of_riders,number_of_drivers,vehicle_type_numeric,Expected_Ride_Duration]])
                predicted_price=self.model.predict(input_data)
                return predicted_price
            def input_values():
                # Example prediction using user input values
                self.user_number_of_riders = 50
                self.user_number_of_drivers = 25
                self.user_vehicle_type = "Economy"
                self.Expected_Ride_Duration = 30
                self.predicted_values = predicted_price(self.user_number_of_riders, self.user_number_of_drivers, self.user_vehicle_type,
                                                self.Expected_Ride_Duration)
                print(f"We are giving few Columns values to finding the Predicted Values of Model")
                print("Predicted price:", self.predicted_values)
            get_vehicle_type_numeric("Economy")
            predicted_price(50,25,'Economy',25)
            input_values()
        except Exception as e:
            error_type,error_msg,err_line=sys.exc_info()
            print(f"Error type {error_type.__name__} -> error line {err_line.tb_lineno}-> msg {error_msg}")
    def comparision_Actual_Predict_values(self):
        try:
            # Predict on test set
            #Create a scatter plot Actual vs Predict values
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=self.y_test.flatten(),
                                     y=self.y_test_pred,
                                     mode='markers',
                                     name="Actual vs Predicted Values"))
            #add a line ideal line representing
            fig.add_trace(go.Scatter(x=[min(self.y_test.flatten()),max(self.y_test.flatten())],
                                     y=[min(self.y_test.flatten()),max(self.y_test.flatten())],
                                     mode='lines',
                                     name='Ideal',
                                     line=dict(color='red',dash='dash')
                                     ))
            fig.update_layout(
                title="Actual vs Predicted values",
                xaxis_title="Actual Values",
                yaxis_title="Predicted Values",
                showlegend=True
            )
            fig.show()
        except Exception as e:
            error_type,error_msg,err_line=sys.exc_info()
            print(f"Error type :{error_type.__name__}-> Error line {err_line.tb_lineno}-> msg {error_msg}")


if __name__ == "__main__":
    try:
        object = Dynamic_Pricing(data=pd.read_csv("C:\\Users\\Saiku\\Downloads\\Dynamic Pricing Project ML\\dynamic_pricing.csv"))
        object.Ride_Duration_Historical_cost()
        object.Ride_Historical_Cost_Vehicle_Type()
        object.Correlation_matrix_data()
        object.Implementation_Dynamic_Pricing()
        object.Profit_of_Riders()
        object.Expected_Ride_Cost_Ride()
        object.Training_Predicted_Model()
        object.model_predictions()
        object.comparision_Actual_Predict_values()
    except Exception as e:
        error_type, error_msg, err_line = sys.exc_info()
        print(f"Error from Line {err_line.tb_lineno} -> type {error_type.__name__} -> Error msg -> {error_msg}")