import json
import urllib.request
from pathlib import Path
import sys

from data.downloader import GWTC_GPS, HELD_OUT_EVENTS

def download_file(url, out_path):
    print(f"Downloading {url} to {out_path}...")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        with open(out_path, 'wb') as f:
            f.write(response.read())

def fetch_gwosc_event(event_name, detector, data_dir):
    url = f"https://gwosc.org/api/v2/event-versions/{event_name}-v1"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read())
        
        # Try to find a 4096Hz HDF5 file for the detector
        strain_info = data['data']['strain']
        
        file_url = None
        for item in strain_info:
            if item['detector'] == detector and item['format'] == 'hdf5' and item['sampling_rate'] == 4096:
                file_url = item['url']
                break
        
        if file_url is None:
            # Fallback to older event versions or different formats if needed
            print(f"No 4096Hz HDF5 found for {event_name} {detector} in -v1. Trying without -v1...")
            return False
            
        out_path = data_dir / f"{event_name}_{detector}.hdf5"
        download_file(file_url, out_path)
        return True
    except Exception as e:
        print(f"Error fetching {event_name}: {e}")
        return False

def main():
    data_dir = Path("data/strain")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    detector = "L1"
    
    # 1. Fetch held out events
    print("Fetching held-out test events...")
    for ev in HELD_OUT_EVENTS:
        fetch_gwosc_event(ev, detector, data_dir)
        
    # 2. Fetch some training events (since noise segments fallback might be hard, we just fetch real events for training)
    print("Fetching training events...")
    count = 0
    for ev in GWTC_GPS:
        if ev in HELD_OUT_EVENTS: continue
        out_path = data_dir / f"{ev}_{detector}.hdf5"
        if out_path.exists():
            count += 1
            continue
            
        success = fetch_gwosc_event(ev, detector, data_dir)
        if success:
            count += 1
        if count >= 60:
            break

if __name__ == '__main__':
    main()
