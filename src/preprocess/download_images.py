import os
import argparse
import time
import urllib.request
from urllib.error import HTTPError, URLError

def download_images(output_dir):
    """
    Downloads images following the pattern: 
    https://ft.amtb.de/doc/MB/{folder}/MB-{folder}-{file}.jpg
    where {folder} is 01-10 and {file} is 0001-9999.
    
    If an image URL returns 404 Not Found, it assumes the files in that folder 
    have ended and breaks out to the next folder.
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for folder_num in range(1, 11):
        folder_str = f"{folder_num:02d}"
        print(f"===== Starting downloads for folder {folder_str} =====")
        
        folder_path = os.path.join(output_dir, folder_str)
        os.makedirs(folder_path, exist_ok=True)
        
        for file_num in range(1, 10000):
            file_str = f"{file_num:04d}"
            url = f"https://ft.amtb.de/doc/MB/{folder_str}/MB-{folder_str}-{file_str}.jpg"
            output_file = os.path.join(folder_path, f"MB-{folder_str}-{file_str}.jpg")
            
            # Skip if we already downloaded it in a past run
            if os.path.exists(output_file):
                print(f"Already exists, skipping: {output_file}")
                continue
                
            try:
                # It's good practice to provide a user agent just in case the server blocks default python clients
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=15) as response:
                    with open(output_file, 'wb') as f:
                        f.write(response.read())
                    print(f"Successfully downloaded: {url}")
            
            except HTTPError as e:
                # 404 indicates we reached the end of the items in this folder
                if e.code == 404:
                    print(f"End of folder {folder_str} reached (404 Not Found at {url}).")
                    break
                else:
                    print(f"Warning: Unexpected HTTP Error {e.code} for {url}. Stopping folder {folder_str}.")
                    break
            except URLError as e:
                print(f"Network error downloading {url}: {e.reason}")
                break
            except Exception as e:
                print(f"Unexpected error when downloading {url}: {e}")
                break
                
            # Sleep slightly to be polite to the image server
            time.sleep(0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download training dataset images from amtb.de")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Output directory to save the images.")
    args = parser.parse_args()
    
    download_images(args.output_dir)
