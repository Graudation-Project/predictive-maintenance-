class Preprocessing:
    def __init__(self,data):
        self.data = data


    def handle_dirty_values(self,data):
        """
        Clean the dataset from Null values, Duplicates, Outliers
        ------
        Parameters:
        data: pandas.DataFrame
        ------
        return:
        cleaned_data:pandas.DataFrame
        """
        cleaned_data = None
        return cleaned_data
    

    def norm_data(self,data):
        """
        Normlize the data using standard scaler
        
        """
        cleaned_data = None
        return cleaned_data
    
    def feature_engineering(self,data):
        result_data = data
        return result_data
    
    def clean_data(self):
        data_cleaned_process1 = self.handle_dirty_values(self.data)
        data_cleaned_process2 = self.norm_data(self.data)
        cleaned_data = self.feature_engineering(self.data)
        return cleaned_data