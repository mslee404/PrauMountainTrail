import gpxpy
import requests
import pandas as pd
import time

GPX_FILE = "Gunung Prau.gpx"
TRACK_NAMES = ["via wates 001", "via dieng 001", "via patakbanteng 001"]

def load_points():
    with open(GPX_FILE, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    points = []

    print("\n=== LIST TRACKS DALAM GPX ===")
    for tr in gpx.tracks:
        print(repr(tr.name))

    for tr in gpx.tracks:
        if tr.name.strip().lower() in TRACK_NAMES:
            print(f"Ambil: {tr.name}")
            for seg in tr.segments:
                for p in seg.points:
                    points.append((p.latitude, p.longitude, tr.name))

    print(f"Total points loaded = {len(points)}")
    return points


def fetch_elevation(lat, lon, i, total):
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    print(f"Request {i}/{total} → {lat},{lon}")

    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        return data["results"][0]["elevation"]
    except Exception as e:
        print("ERROR:", e)
        return None


def main():
    pts = load_points()
    results = []

    for i, (lat, lon, track_name) in enumerate(pts, start=1):
        elev = fetch_elevation(lat, lon, i, len(pts))
        results.append([track_name, lat, lon, elev])
        time.sleep(1)  # hindari rate-limit

    df = pd.DataFrame(results, columns=["track", "lat", "lon", "elevation"])
    df.to_csv("track_with_elevation.csv", index=False)

    print("\n=== DONE, CSV TERSIMPAN ===")


if __name__ == "__main__":
    main()
