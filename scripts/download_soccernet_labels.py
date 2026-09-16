from SoccerNet.Downloader import SoccerNetDownloader
import os

download_path = r"E:\SprotsHighlights\datasets\SoccerNet"

os.makedirs(download_path, exist_ok=True)

print("Starting SoccerNet label download...")
print("Download location:", download_path)

downloader = SoccerNetDownloader(
    LocalDirectory=download_path
)

downloader.downloadGames(
    files=["Labels-v2.json"],
    split=["train", "valid", "test"]
)

print("SoccerNet labels downloaded successfully.")