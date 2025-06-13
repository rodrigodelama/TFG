import requests
from datetime import datetime, timedelta

# --- Configuration ---
FILE_PATH = 'year.txt'  # File to store generated filenames
START_DATE = datetime(2025, 2, 14)
END_DATE = datetime(2025, 3, 18)
BASE_URL = "https://www.omie.es/es/file-download?parents%5B0%5D=marginalpdbc&filename="

# --- Functions ---

def generate_and_download_files(start_date: datetime, end_date: datetime, base_url: str, file_path: str):
    """
    Generates filenames for daily data, saves them to a file, and then downloads them.
    """
    current_date = start_date
    filenames = []

    # Generate filenames and store them
    while current_date <= end_date:
        filename = f'marginalpdbc_{current_date.strftime("%Y%m%d")}.1'
        filenames.append(filename)
        current_date += timedelta(days=1)

    with open(file_path, 'w') as f:
        for name in filenames:
            f.write(name + '\n')
    print(f"Generated {len(filenames)} filenames and saved to {file_path}")

    # Download files
    for filename in filenames:
        url = base_url + filename
        try:
            # Added a timeout for robustness
            response = requests.get(url, verify=False, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {filename}: {e}")

# --- Execution ---
if __name__ == "__main__":
    generate_and_download_files(START_DATE, END_DATE, BASE_URL, FILE_PATH)


import requests
from datetime import datetime, timedelta

# Usage instructions:
    # Set file name to save the filenames to download
    # Set start and end dates

file_path = 'year.txt'
start_date = datetime(2025, 2, 14)
end_date = datetime(2025, 3, 18)

def download_files(urls):
    for url in urls:
        # Extract the filename from the URL
        filename = url.split('=')[-1]
        
        # Send a GET request to download the file
        # response = requests.get(url)
        response = requests.get(url, verify=False) # I hate Zscaler and EY tech
        
        # Check if the request was successful
        if response.status_code == 200:
            # Write the content to a file with the extracted filename
            with open(filename, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded {filename}")
        else:
            print(f"Failed to download {filename}")

# Example address to build
# https://www.omie.es/es/file-download?parents%5B0%5D=marginalpdbc&filename=marginalpdbc_20230102.1
def read_filenames_and_compose_urls(file_path):
    base_url = "https://www.omie.es/es/file-download?parents%5B0%5D=marginalpdbc&filename="
    
    # Read the filenames from the .txt file
    with open(file_path, 'r') as file:
        filenames = [line.strip() for line in file if line.strip()]
    
    # Compose the URLs
    urls = [base_url + filename for filename in filenames]
    
    return urls

# Compose filenames to download
# def compose_filenames() -> str: # Si quiero un tipo especifico poner eso
def compose_filenames():
    # Set start and end dates
    start_date = start_date
    end_date = end_date
    
    # Generate the entries for the number of days comprehended
    to_generate = (end_date - start_date).days + 1
    entries = []
    for i in range(0, to_generate):  # x days from start to end
        date_entry = start_date + timedelta(days=i)
        entries.append(f'marginalpdbc_{date_entry.strftime("%Y%m%d")}.1')
    
    # save to a .txt file
    with open(file_path, 'w') as file:
        for entry in entries:
            file.write(entry + '\n')

# Example usage
compose_filenames()

urls = read_filenames_and_compose_urls(file_path)

download_files(urls)
