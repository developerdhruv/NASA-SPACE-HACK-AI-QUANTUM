import obspy
import pandas as pd

def convert_mseed_to_csv(mseed_file, csv_file):
    # Read the MiniSEED file
    stream = obspy.read(mseed_file)

    # Initialize an empty list to collect data
    all_data = []

    # Loop through each trace in the stream
    for trace in stream:
        # Extract time and data values
        time = trace.times("timestamp")
        values = trace.data

        # Append the data to the list
        all_data.extend(zip(time, values))

    # Create a DataFrame
    df = pd.DataFrame(all_data, columns=['time', 'value'])

    # Save to CSV
    df.to_csv(csv_file, index=False)
    print(f"Data successfully saved to {csv_file}")

# Example usage
if __name__ == "__main__":
    mseed_file = 'data_test.mseed'  # Replace with your MiniSEED file path
    csv_file = 'output_test.csv'        # Replace with your desired output CSV file name
    convert_mseed_to_csv(mseed_file, csv_file)
