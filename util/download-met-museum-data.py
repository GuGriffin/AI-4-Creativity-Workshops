import os
import random
import argparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def download_image(url, save_path, session):
    """Downloads an image from a URL and saves it."""
    try:
        response = session.get(url, stream=True, timeout=15)
        response.raise_for_status()
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(8192):
                file.write(chunk)
        print(f"✅ Downloaded: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Failed to download {url}: {e}")

def fetch_met_image_api(object_id, save_directory, session):
    """Fetches artwork data from The Met API and downloads its image if available."""
    api_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
    try:
        response = session.get(api_url, timeout=10)
        if response.status_code == 404:
            print(f"❌ Object {object_id} not found.")
            return False
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ API access failed for {object_id}: {e}")
        return False

    data = response.json()
    image_url = data.get("primaryImage")

    if not image_url:
        print(f"🚫 No image for object {object_id}: {data.get('title', 'Unknown')}")
        return False

    title = data.get("title", "Unknown Title")
    artist = data.get("artistDisplayName", "Unknown Artist")
    date = data.get("objectDate", "Unknown Date")

    print(f"🎨 {object_id}: {title} — {artist} ({date})")

    image_name = f"{object_id}.jpg"
    save_path = os.path.join(save_directory, image_name)
    download_image(image_url, save_path, session)
    return True

def main():
    parser = argparse.ArgumentParser(description="Download random public domain images from The Met Museum API.")
    parser.add_argument('--num_samples', type=int, required=True, help="Number of random artworks to download.")
    parser.add_argument('--save_directory', type=str, default="met_images", help="Directory to save images.")
    args = parser.parse_args()

    os.makedirs(args.save_directory, exist_ok=True)

    # Create a single requests session with retry strategy
    session = requests.Session()
    retry_strategy = Retry(total=3, status_forcelist=[429, 500, 502, 503, 504], backoff_factor=1)
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # ✅ Fetch valid object IDs directly from The Met's API
    print("📦 Fetching valid object IDs from The Met API...")
    ids_response = session.get("https://collectionapi.metmuseum.org/public/collection/v1/objects")
    ids_response.raise_for_status()
    all_object_ids = ids_response.json()["objectIDs"]

    print(f"📊 Total valid objects: {len(all_object_ids):,}")

    # Randomly sample N IDs from the valid ones
    random_ids = random.sample(all_object_ids, args.num_samples)

    success_count = 0
    for object_id in random_ids:
        print(f"\n🔍 Checking object {object_id}...")
        if fetch_met_image_api(object_id, args.save_directory, session):
            success_count += 1

    print(f"\n✅ Finished! {success_count}/{args.num_samples} images downloaded successfully.")

if __name__ == "__main__":
    main()
