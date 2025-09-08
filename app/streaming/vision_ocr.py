#!/usr/bin/env python3
"""
Vision.framework OCR integration for text detection and extraction.
Uses native macOS Vision API for high-performance text recognition.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import time

import Vision
import Quartz
from Quartz import CoreGraphics as CG
from Foundation import NSData, NSMutableArray

logger = logging.getLogger(__name__)

@dataclass
class TextRegion:
    """Detected text region with metadata"""
    text: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x, y, width, height normalized
    words: List[str]
    
@dataclass
class OCRResult:
    """Complete OCR result for a frame"""
    regions: List[TextRegion]
    processing_time: float
    frame_timestamp: float
    total_words: int
    average_confidence: float

class VisionOCR:
    """High-performance OCR using Vision.framework"""
    
    def __init__(self, recognition_level: str = 'accurate'):
        """
        Initialize Vision OCR.
        
        Args:
            recognition_level: 'accurate' or 'fast'
        """
        self.recognition_level = recognition_level
        self.last_result: Optional[OCRResult] = None
        self.processing_times = []
        
    def process_frame(self, frame: np.ndarray) -> OCRResult:
        """
        Process a frame for text detection.
        
        Args:
            frame: RGB numpy array
            
        Returns:
            OCRResult with detected text regions
        """
        start_time = time.time()
        
        # Convert numpy array to CGImage
        cg_image = self._numpy_to_cgimage(frame)
        if not cg_image:
            logger.error("Failed to convert frame to CGImage")
            return OCRResult([], 0, time.time(), 0, 0)
        
        # Create Vision request
        request = Vision.VNRecognizeTextRequest.alloc().init()
        
        # Set recognition level
        if self.recognition_level == 'fast':
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast)
        else:
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        
        # Set language
        request.setRecognitionLanguages_(["en-US"])
        request.setUsesLanguageCorrection_(True)
        
        # Create request handler
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, None
        )
        
        # Perform OCR
        error = None
        success = handler.performRequests_error_([request], error)
        
        if not success:
            logger.error(f"Vision OCR failed: {error}")
            return OCRResult([], 0, time.time(), 0, 0)
        
        # Extract results
        regions = []
        total_confidence = 0
        
        observations = request.results()
        if observations:
            for observation in observations:
                # Get bounding box (normalized coordinates)
                bbox = observation.boundingBox()
                x, y, w, h = bbox.origin.x, bbox.origin.y, bbox.size.width, bbox.size.height
                
                # Get recognized text
                text = observation.text()
                confidence = observation.confidence()
                
                # Get word-level candidates if available
                words = []
                candidates = observation.topCandidates_(1)
                if candidates and len(candidates) > 0:
                    candidate = candidates[0]
                    if hasattr(candidate, 'string'):
                        # Extract individual words
                        words = candidate.string().split()
                
                regions.append(TextRegion(
                    text=text,
                    confidence=confidence,
                    bbox=(x, y, w, h),
                    words=words
                ))
                
                total_confidence += confidence
        
        # Calculate metrics
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        total_words = sum(len(r.words) for r in regions)
        avg_confidence = total_confidence / len(regions) if regions else 0
        
        result = OCRResult(
            regions=regions,
            processing_time=processing_time,
            frame_timestamp=time.time(),
            total_words=total_words,
            average_confidence=avg_confidence
        )
        
        self.last_result = result
        logger.debug(f"OCR processed {len(regions)} regions in {processing_time*1000:.1f}ms")
        
        return result
    
    def _numpy_to_cgimage(self, frame: np.ndarray):
        """Convert numpy array to CGImage"""
        height, width = frame.shape[:2]
        
        # Ensure frame is RGB (not RGBA)
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        
        # Convert to BGRA for Core Graphics
        bgra = np.zeros((height, width, 4), dtype=np.uint8)
        bgra[:, :, 2] = frame[:, :, 0]  # R -> B
        bgra[:, :, 1] = frame[:, :, 1]  # G -> G
        bgra[:, :, 0] = frame[:, :, 2]  # B -> R
        bgra[:, :, 3] = 255  # Alpha
        
        # Create CGImage
        bytes_per_row = width * 4
        color_space = CG.CGColorSpaceCreateDeviceRGB()
        
        # Create bitmap context
        context = CG.CGBitmapContextCreate(
            bgra.data,
            width,
            height,
            8,  # bits per component
            bytes_per_row,
            color_space,
            CG.kCGImageAlphaPremultipliedLast | CG.kCGBitmapByteOrder32Big
        )
        
        if context:
            return CG.CGBitmapContextCreateImage(context)
        
        return None
    
    def get_text_at_point(self, x: float, y: float) -> Optional[TextRegion]:
        """
        Get text region at specific point (normalized coordinates).
        
        Args:
            x: Normalized x coordinate (0-1)
            y: Normalized y coordinate (0-1)
            
        Returns:
            TextRegion if found at point
        """
        if not self.last_result:
            return None
        
        for region in self.last_result.regions:
            rx, ry, rw, rh = region.bbox
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return region
        
        return None
    
    def get_all_text(self) -> str:
        """Get all detected text as a single string"""
        if not self.last_result:
            return ""
        
        return "\n".join(r.text for r in self.last_result.regions)
    
    def get_high_confidence_text(self, threshold: float = 0.8) -> List[TextRegion]:
        """Get only high-confidence text regions"""
        if not self.last_result:
            return []
        
        return [r for r in self.last_result.regions if r.confidence >= threshold]
    
    def get_metrics(self) -> Dict:
        """Get OCR performance metrics"""
        metrics = {
            "recognition_level": self.recognition_level,
            "last_processing_time_ms": self.processing_times[-1] * 1000 if self.processing_times else 0,
            "avg_processing_time_ms": np.mean(self.processing_times) * 1000 if self.processing_times else 0,
            "total_processed": len(self.processing_times)
        }
        
        if self.last_result:
            metrics.update({
                "last_regions_count": len(self.last_result.regions),
                "last_words_count": self.last_result.total_words,
                "last_avg_confidence": self.last_result.average_confidence
            })
        
        return metrics


class TextTracker:
    """Track text regions across frames for stability"""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.tracked_regions: Dict[str, List[TextRegion]] = {}
        self.region_ids: Dict[str, str] = {}
        self.next_id = 0
        
    def update(self, regions: List[TextRegion]) -> List[Tuple[str, TextRegion]]:
        """
        Update tracking with new regions.
        
        Returns:
            List of (tracking_id, region) tuples
        """
        tracked = []
        
        for region in regions:
            # Find matching tracked region
            best_match_id = None
            best_similarity = 0
            
            for track_id, history in self.tracked_regions.items():
                if history:
                    last_region = history[-1]
                    similarity = self._calculate_similarity(region, last_region)
                    
                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_match_id = track_id
            
            # Assign ID
            if best_match_id:
                # Update existing track
                self.tracked_regions[best_match_id].append(region)
                tracked.append((best_match_id, region))
            else:
                # Create new track
                new_id = f"text_{self.next_id}"
                self.next_id += 1
                self.tracked_regions[new_id] = [region]
                tracked.append((new_id, region))
        
        # Clean old tracks
        self._clean_old_tracks()
        
        return tracked
    
    def _calculate_similarity(self, r1: TextRegion, r2: TextRegion) -> float:
        """Calculate similarity between two text regions"""
        # Text similarity
        text_sim = self._text_similarity(r1.text, r2.text)
        
        # Position similarity
        x1, y1, w1, h1 = r1.bbox
        x2, y2, w2, h2 = r2.bbox
        
        # Calculate IoU (Intersection over Union)
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        
        intersection = x_overlap * y_overlap
        union = w1 * h1 + w2 * h2 - intersection
        
        iou = intersection / union if union > 0 else 0
        
        # Combined similarity
        return 0.7 * text_sim + 0.3 * iou
    
    def _text_similarity(self, t1: str, t2: str) -> float:
        """Calculate text similarity using Levenshtein ratio"""
        if not t1 or not t2:
            return 0.0
        
        # Simple character overlap ratio
        common = sum(1 for c1, c2 in zip(t1, t2) if c1 == c2)
        return common / max(len(t1), len(t2))
    
    def _clean_old_tracks(self, max_age: int = 30):
        """Remove tracks that haven't been updated recently"""
        # Keep only recent history (last 30 frames)
        for track_id in list(self.tracked_regions.keys()):
            history = self.tracked_regions[track_id]
            if len(history) > max_age:
                self.tracked_regions[track_id] = history[-max_age:]


# Demo usage
if __name__ == "__main__":
    import cv2
    
    def demo():
        """Demonstrate Vision OCR on a screenshot"""
        print("🔤 Vision.framework OCR Demo")
        print("=" * 40)
        
        # Create OCR instance
        ocr = VisionOCR(recognition_level='accurate')
        tracker = TextTracker()
        
        # Capture a frame using mss for testing
        import mss
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            screenshot = sct.grab(monitor)
            
            # Convert to numpy
            frame = np.array(screenshot)[:, :, :3]  # Remove alpha
            
            print(f"Frame shape: {frame.shape}")
            print("\nProcessing OCR...")
            
            # Process frame
            result = ocr.process_frame(frame)
            
            print(f"\n📊 OCR Results:")
            print(f"  • Regions detected: {len(result.regions)}")
            print(f"  • Total words: {result.total_words}")
            print(f"  • Average confidence: {result.average_confidence:.2%}")
            print(f"  • Processing time: {result.processing_time*1000:.1f}ms")
            
            # Track regions
            tracked = tracker.update(result.regions)
            
            print(f"\n📝 Detected Text (top 5):")
            for i, (track_id, region) in enumerate(tracked[:5]):
                print(f"\n  [{track_id}] Confidence: {region.confidence:.2%}")
                print(f"  Position: ({region.bbox[0]:.2f}, {region.bbox[1]:.2f})")
                print(f"  Text: '{region.text[:100]}...' " if len(region.text) > 100 else f"  Text: '{region.text}'")
            
            # Get metrics
            metrics = ocr.get_metrics()
            print(f"\n⚡ Performance Metrics:")
            for key, value in metrics.items():
                print(f"  • {key}: {value}")
    
    demo()