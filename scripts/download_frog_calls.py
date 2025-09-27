import csv
import os
import requests

MULTIMEDIA_FILE = r'C:/Users/dhruv/OneDrive/Desktop/Frog Sound Project/scripts/multimedia.txt'
RAW_DATA_DIR = r"C:\\Users\\dhruv\\OneDrive\\Desktop\\Frog Sound Project\\FrogSoundProject\\data\\raw"

os.makedirs(RAW_DATA_DIR, exist_ok=True)

with open(MULTIMEDIA_FILE, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    
    for row in reader:
        if row['type'].lower() == 'sound':
            url = row['accessURI']
            filename = url.split('/')[-1]
            filepath = os.path.join(RAW_DATA_DIR, filename)

            if os.path.exists(filepath):
                print(f"Already downloaded: {filename}")
                continue

            print(f"Downloading: {filename}")
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Saved to {filepath}")
            except Exception as e:
                print(f"Failed to download {url}: {e}")