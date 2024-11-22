import pandas as pd
from io import StringIO
from google.cloud import storage, bigquery

def etl_pipeline(event, context=None):
    """Triggered by a change to a Cloud Storage bucket."""
    try:
        # Extract file details
        bucket_name = event['bucket']
        file_name = event['name']

        # Initialize Cloud Storage client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        # Read the data into a Pandas DataFrame
        data = blob.download_as_text()
        df = pd.read_csv(StringIO(data))

        print('Starting feature engineering...')
        # Feature Engineering (Transform)
        customer_sales = df.groupby('Customer ID')['Sales'].sum().reset_index()
        customer_sales.rename(columns={'Sales': 'customers.SUM(sales.Sales)'}, inplace=True)

        customer_max_quantity = df.groupby('Customer ID')['Quantity'].max().reset_index()
        customer_max_quantity.rename(columns={'Quantity': 'customers.MAX(sales.Quantity)'}, inplace=True)

        product_sales = df.groupby('Product ID')['Sales'].sum().reset_index()
        product_sales.rename(columns={'Sales': 'products.SUM(sales.Sales)'}, inplace=True)

        product_avg_sales = df.groupby('Product ID')['Sales'].mean().reset_index()
        product_avg_sales.rename(columns={'Sales': 'products.MEAN(sales.Sales)'}, inplace=True)

        print('Feature engineering process completed')
        print('Creating star schema')

        # Create star schema
        print('Creating customer dimension table...')
        customer_dim = df[[
            "Customer ID", "Customer Name", "Segment", "Country", 
            "City", "State", "Postal Code"
        ]].drop_duplicates(subset=['Customer ID']).reset_index(drop=True)

        print('Creating product dimension table...')
        product_dim = df[[
            "Product ID", "Product Name", "Category", "Sub-Category"
        ]].drop_duplicates(subset=['Product ID']).reset_index(drop=True)

        print('Creating order dimension table...')
        order_dim = df[[
            "Order ID", "Order Date", "Ship Mode"
        ]].drop_duplicates(subset=['Order ID']).reset_index(drop=True)

        print('Creating time dimension table...')
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        time_dim = df[['Order Date']].drop_duplicates().reset_index(drop=True)
        time_dim['Day'] = time_dim['Order Date'].dt.day
        time_dim['Month'] = time_dim['Order Date'].dt.month
        time_dim['Quarter'] = time_dim['Order Date'].dt.quarter
        time_dim['Year'] = time_dim['Order Date'].dt.year
        time_dim.rename(columns={'Order Date': 'Date'}, inplace=True)

        print('Creating sales fact table...')
        # Create sales_fact table
        sales_fact = df[[
            "Order ID", "Product ID", "Customer ID", "Order Date", "Region", 
            "Sales", "Quantity", "Discount", "Profit"
        ]].copy()

        # Create a unique Sale ID as a sequential ID
        sales_fact['Sale ID'] = range(1, len(sales_fact) + 1)  # Creates a sequential Sale ID starting from 1
        sales_fact.rename(columns={'Order Date': 'Date'}, inplace=True)

        # Merge the feature-engineered tables to bring in the aggregation columns
        sales_fact = sales_fact.merge(customer_sales, on='Customer ID', how='left')
        sales_fact = sales_fact.merge(customer_max_quantity, on='Customer ID', how='left')
        sales_fact = sales_fact.merge(product_sales, on='Product ID', how='left')
        sales_fact = sales_fact.merge(product_avg_sales, on='Product ID', how='left')

        # Final selection of columns for the fact table
        sales_fact = sales_fact[[
            'Sale ID', 'Order ID', 'Product ID', 'Customer ID', 'Date', 
            'Sales', 'Quantity', 'Discount', 'Profit', 
            'customers.SUM(sales.Sales)', 'customers.MAX(sales.Quantity)', 
            'products.SUM(sales.Sales)', 'products.MEAN(sales.Sales)'
        ]]

        print("Sales fact table created successfully")
        # Load data to BigQuery
        print('Loading data into BigQuery...')
        bq_client = bigquery.Client()
        dataset_id = "data-mining-assignment-442318.etl_output"

        tables = {
            "customer_dim": customer_dim,
            "product_dim": product_dim,
            "order_dim": order_dim,
            "time_dim": time_dim,
            "sales_fact": sales_fact
        }

        for table_name, table_data in tables.items():
            table_id = f"{dataset_id}.{table_name}"
            print(f"Loading {table_name} to {table_id}...")
            job = bq_client.load_table_from_dataframe(table_data, table_id)
            job.result()  # Wait for the load job to complete
            print(f"Table {table_name} loaded successfully.")

        print("ETL process complete.")
    
    except Exception as e:
        print(f"An error occurred during the ETL process: {e}")
