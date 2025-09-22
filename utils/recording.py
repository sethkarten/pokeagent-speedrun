#!/usr/bin/env python3
"""
Video recording utilities for capturing gameplay footage.
"""

import cv2
import datetime
import numpy as np
from PIL import Image


class VideoRecorder:
    """Handles video recording of gameplay"""
    
    def __init__(self, fps=120, output_fps=30, enabled=False):
        """
        Initialize video recorder.
        
        Args:
            fps: Source FPS of the emulator
            output_fps: Target FPS for the video file
            enabled: Whether recording is enabled
        """
        self.enabled = enabled
        self.recording = False
        self.video_writer = None
        self.video_filename = None
        
        # Frame rate settings
        self.source_fps = fps
        self.output_fps = output_fps
        self.frame_skip = max(1, fps // output_fps)  # How many frames to skip
        self.frame_counter = 0
        
        print(f"📹 Video recorder initialized (enabled={enabled})")
        if enabled:
            print(f"   Recording settings: {output_fps} FPS (every {self.frame_skip} frames)")
    
    def start_recording(self):
        """Start a new recording session"""
        if not self.enabled:
            return False
        
        try:
            # Create filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_filename = f"pokegent_recording_{timestamp}.mp4"
            
            # Video settings (GBA resolution is 240x160)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.video_filename, 
                fourcc, 
                float(self.output_fps), 
                (240, 160)
            )
            
            if self.video_writer.isOpened():
                self.recording = True
                print(f"🎬 Started recording to {self.video_filename}")
                return True
            else:
                print(f"❌ Failed to open video writer")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            return False
    
    def record_frame(self, screenshot):
        """
        Record a single frame if recording is active.
        
        Args:
            screenshot: PIL Image or numpy array of the frame
        """
        if not self.recording or not self.video_writer:
            return
        
        try:
            # Skip frames based on frame_skip setting
            self.frame_counter += 1
            if self.frame_counter % self.frame_skip != 0:
                return
            
            # Convert PIL Image to numpy array if needed
            if isinstance(screenshot, Image.Image):
                frame = np.array(screenshot)
            else:
                frame = screenshot
            
            # Ensure frame is in BGR format for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert RGB to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 4:
                # Convert RGBA to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Write frame
            self.video_writer.write(frame)
            
        except Exception as e:
            print(f"⚠️ Failed to record frame: {e}")
    
    def stop_recording(self):
        """Stop the current recording session"""
        if not self.recording:
            return
        
        try:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            self.recording = False
            self.frame_counter = 0
            
            if self.video_filename:
                print(f"💾 Recording saved to {self.video_filename}")
                self.video_filename = None
                
        except Exception as e:
            print(f"⚠️ Failed to stop recording properly: {e}")
    
    def toggle_recording(self):
        """Toggle recording on/off"""
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def __del__(self):
        """Cleanup on deletion"""
        if self.recording:
            self.stop_recording()


# Global recorder instance
_video_recorder = None


def init_video_recording(enabled=False, fps=120, output_fps=30):
    """
    Initialize global video recording.
    
    Args:
        enabled: Whether to enable recording
        fps: Source FPS
        output_fps: Target output FPS
    
    Returns:
        VideoRecorder: The initialized recorder
    """
    global _video_recorder
    _video_recorder = VideoRecorder(fps=fps, output_fps=output_fps, enabled=enabled)
    
    if enabled:
        _video_recorder.start_recording()
    
    return _video_recorder


def get_video_recorder():
    """Get the global video recorder instance"""
    global _video_recorder
    if _video_recorder is None:
        _video_recorder = VideoRecorder(enabled=False)
    return _video_recorder


def record_frame(screenshot):
    """
    Record a frame using the global recorder.
    
    Args:
        screenshot: The frame to record
    """
    recorder = get_video_recorder()
    if recorder:
        recorder.record_frame(screenshot)


def stop_recording():
    """Stop the global video recording"""
    recorder = get_video_recorder()
    if recorder:
        recorder.stop_recording()