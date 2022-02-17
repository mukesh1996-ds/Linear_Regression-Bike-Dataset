from load_csv import load_csv

# checking the shape of the data.
def check_shape(data):
    return data.shape

# Checking the top 5 records in the data.
def check_top_records(data):
    return data.head()

# Check the basic information of the records
def check_info(data):
    return data.info()


# check describe

def check_describe(data):
    return data.describe()
