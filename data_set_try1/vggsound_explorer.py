import pandas as pd
import random
from IPython.display import display, HTML
import os
import requests
from pytube import YouTube
import subprocess

class VGGSoundExplorer:
    """
    A class to explore and sample from the VGGSound dataset.
    """
    
    def __init__(self, dataset_path='dataset/vggsound.csv'):
        """
        Initialize the explorer with the dataset path.
        
        Args:
            dataset_path (str): Path to the VGGSound CSV file
        """
        self.df = pd.read_csv(dataset_path, header=None, 
                             names=['video_id', 'start_sec', 'label', 'split'])
        
        # Create label counts dataframe
        self.label_counts = (
            self.df['label']
            .value_counts()
            .reset_index()
            .rename(columns={'index': 'label', 'label': 'count'})
        )
        
        # Create a dictionary to quickly get videos by label
        self._create_label_index()
        
    def _create_label_index(self):
        """Create an index mapping labels to their video entries"""
        self.label_index = {}
        for label in self.df['label'].unique():
            self.label_index[label] = self.df[self.df['label'] == label]
            
    def get_available_labels(self):
        """Return a sorted list of all available labels"""
        return sorted(self.label_index.keys())
    
    def get_label_count(self, label):
        """Return the count of videos for a given label"""
        if label not in self.label_index:
            return 0
        return len(self.label_index[label])
    
    def get_random_videos(self, label, n=5, split=None):
        """
        Get n random videos with the specified label.
        
        Args:
            label (str): The label to filter by
            n (int): Number of videos to return
            split (str, optional): Filter by split ('train', 'test', or None for all)
            
        Returns:
            pd.DataFrame: DataFrame containing n random video entries
        """
        if label not in self.label_index:
            print(f"Label '{label}' not found in dataset.")
            return pd.DataFrame()
            
        videos = self.label_index[label]
        
        if split:
            videos = videos[videos['split'] == split]
            
        if len(videos) < n:
            print(f"Warning: Requested {n} videos but only {len(videos)} available for label '{label}'")
            return videos
            
        return videos.sample(n).reset_index(drop=True)
    
    def display_video_info(self, videos_df):
        """Display information about the videos in a nice format"""
        for i, row in videos_df.iterrows():
            print(f"Video {i+1}:")
            print(f"  YouTube ID: {row['video_id']}")
            print(f"  Start time: {row['start_sec']} seconds")
            print(f"  Label: {row['label']}")
            print(f"  Split: {row['split']}")
            print(f"  YouTube URL: https://www.youtube.com/watch?v={row['video_id']}")
            print()
    
    def generate_video_preview_html(self, videos_df):
        """Generate HTML to display YouTube video previews"""
        html = "<div style='display: flex; flex-wrap: wrap;'>"
        
        for i, row in videos_df.iterrows():
            video_id = row['video_id']
            start_time = row['start_sec']
            label = row['label']
            
            html += f"""
            <div style='margin: 10px; width: 320px;'>
                <div style='font-weight: bold;'>{label}</div>
                <iframe width="320" height="180" 
                    src="https://www.youtube.com/embed/{video_id}?start={start_time}" 
                    frameborder="0" allowfullscreen>
                </iframe>
                <div>Start: {start_time}s, Split: {row['split']}</div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def show_videos(self, videos_df):
        """Display YouTube video previews"""
        html = self.generate_video_preview_html(videos_df)
        display(HTML(html))
    
    def download_audio_segment(self, video_id, start_sec, output_dir="audio_samples", 
                              duration=10, format="wav"):
        """
        Download a segment of audio from a YouTube video.
        
        Args:
            video_id (str): YouTube video ID
            start_sec (int): Starting second
            output_dir (str): Directory to save the audio
            duration (int): Duration in seconds to extract
            format (str): Audio format (wav, mp3, etc.)
            
        Returns:
            str: Path to the downloaded audio file or None if failed
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{video_id}_{start_sec}.{format}")
        
        # If file already exists, return its path
        if os.path.exists(output_file):
            return output_file
            
        try:
            # Get the video URL
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            # Download the audio using pytube
            yt = YouTube(url)
            audio_stream = yt.streams.filter(only_audio=True).first()
            temp_file = audio_stream.download(output_path=output_dir, filename=f"temp_{video_id}")
            
            # Use ffmpeg to extract the segment
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_file,
                "-ss", str(start_sec),
                "-t", str(duration),
                "-acodec", "pcm_s16le" if format == "wav" else "libmp3lame",
                "-ar", "44100",
                "-ac", "2",
                output_file
            ]
            
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Remove the temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            return output_file
            
        except Exception as e:
            print(f"Error downloading audio: {e}")
            return None
    
    def batch_download_audio(self, videos_df, output_dir="audio_samples", 
                           duration=10, format="wav"):
        """
        Download audio segments for multiple videos.
        
        Args:
            videos_df (pd.DataFrame): DataFrame containing video information
            output_dir (str): Directory to save the audio
            duration (int): Duration in seconds to extract
            format (str): Audio format (wav, mp3, etc.)
            
        Returns:
            list: Paths to the downloaded audio files
        """
        audio_files = []
        
        for i, row in videos_df.iterrows():
            video_id = row['video_id']
            start_sec = row['start_sec']
            label = row['label']
            
            # Create label-specific subdirectory
            label_dir = os.path.join(output_dir, label.replace(" ", "_").replace(",", ""))
            os.makedirs(label_dir, exist_ok=True)
            
            print(f"Downloading audio for {video_id} at {start_sec}s...")
            audio_file = self.download_audio_segment(
                video_id, start_sec, output_dir=label_dir, 
                duration=duration, format=format)
            
            if audio_file:
                audio_files.append(audio_file)
                print(f"Downloaded: {audio_file}")
            else:
                print(f"Failed to download audio for {video_id}")
        
        return audio_files


# Usage example (see 01_notebook.ipynb)
if __name__ == "__main__":
    # Initialize the explorer
    explorer = VGGSoundExplorer(dataset_path='dataset/vggsound.csv')
    
    # Get a list of all available labels
    labels = explorer.get_available_labels()
    print(f"Total number of labels: {len(labels)}")
    
    # Example: Get 5 random videos with the label "playing piano"
    label = "playing piano"
    videos = explorer.get_random_videos(label, n=5)
    
    # Display information about the videos
    explorer.display_video_info(videos)
    
    # Show video previews (works in Jupyter notebook)
    # explorer.show_videos(videos)
    
    # Example: Download audio segments for these videos
    # audio_files = explorer.batch_download_audio(videos, duration=5)