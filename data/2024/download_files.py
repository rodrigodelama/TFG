# OLD style
# https://www.omie.es/es/file-download?parents%5B0%5D=marginalpdbc&filename=marginalpdbc_20230101.1

# NEW style
# https://www.omie.es/es/file-download?parents=marginalpdbc&filename=marginalpdbc_20241002.1
import requests
import os

def download_files(urls):
    for url in urls:
        # Extract the filename from the URL
        filename = url.split('=')[-1]
        
        # Send a GET request to download the file
        # response = requests.get(url)
        # Send a GET request to download the file, ignoring SSL verification
        response = requests.get(url, verify=False)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Write the content to a file with the extracted filename
            with open(filename, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded {filename}")
        else:
            print(f"Failed to download {filename}")

def read_filenames_and_compose_urls(file_path):
    base_url = "https://www.omie.es/es/file-download?parents=marginalpdbc&filename="
    
    # Read the filenames from the .txt file
    with open(file_path, 'r') as file:
        filenames = [line.strip() for line in file if line.strip()]
    
    # Compose the URLs
    urls = [base_url + filename for filename in filenames]
    
    return urls

# Example usage
# file_path = '2024.txt'
file_path = 'data/2024/2024-2025.txt'
urls = read_filenames_and_compose_urls(file_path)

download_files(urls)
