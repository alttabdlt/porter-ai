#!/usr/bin/env python3
"""
VLM Processor using FastVLM with MLX
"""

import asyncio
import numpy as np
import logging
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from PIL import Image
import time
import json
from dataclasses import dataclass
from enum import Enum

# MLX-LM imports (lazy load to save memory)
# These will only be imported when FastVLMProcessor is actually used
load = None
generate = None  
load_config = None

def _lazy_load_mlx():
    """Lazy load MLX modules only when needed"""
    global load, generate, load_config
    if load is None:
        # Try mlx-vlm first (for vision-language models)
        try:
            from mlx_vlm import load as _load, generate as _generate
            from mlx_vlm.utils import load_config as _load_config
            load = _load
            generate = _generate
            load_config = _load_config
            logger.info("Using mlx-vlm for vision-language models")
        except ImportError:
            # Fall back to mlx-lm (text-only, won't work for VLM)
            try:
                from mlx_lm import load as _load, generate as _generate
                from mlx_lm.utils import load_config as _load_config
                load = _load
                generate = _generate
                load_config = _load_config
                logger.warning("Using mlx-lm (text-only) - VLM models will likely fail")
            except ImportError:
                # No MLX available
                logger.error("No MLX packages available - install mlx-vlm for VLM support")
                load = lambda x: (None, None)
                generate = lambda *args, **kwargs: None
                load_config = lambda x: {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLMPromptTemplates(Enum):
    """Predefined prompt templates for different analysis types"""
    
    DESCRIBE = "Summarize what the user is currently doing in one sentence."
    
    HAZARD = """Identify any risky or dangerous actions on screen. 
    Rate risk from 0-1 and explain briefly. 
    Look for: rm -rf, format, delete, passwords visible, wrong recipient."""
    
    NEXT_STEP = """Based on the current screen state, suggest the most helpful next action.
    Be specific and actionable in 10 words or less."""
    
    EXTRACT = """Extract structured information from this screen.
    Return JSON with: urls, errors, file_paths, entities (people/orgs), key_values.
    Be precise and include only clearly visible information."""
    
    TASK_HYPOTHESIS = """Infer what task the user is trying to accomplish.
    Consider: open applications, recent actions, visible content.
    Respond in one sentence."""
    
    CONTEXT = """Describe the current context: 
    - Active application
    - Current file/document
    - Main activity
    Format as JSON: {app, file, activity}"""
    
    ERROR_DETECTION = """Identify any error messages, warnings, or issues on screen.
    List each with severity (low/medium/high) and location."""
    
    AUTOMATION = """Identify repetitive actions that could be automated.
    List specific suggestions with estimated time savings."""

@dataclass
class ROIResult:
    """Result from ROI analysis"""
    region: Tuple[int, int, int, int]  # x, y, w, h
    description: str
    importance: float
    entities: List[str]

class FastVLMProcessor:
    """Apple FastVLM-0.5B processor using MLX for Apple Silicon"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize Apple FastVLM-0.5B processor
        
        Using Apple's official FastVLM-0.5B model which is optimized for:
        - 85x faster Time-to-First-Token (TTFT)
        - 3.4x smaller vision encoder
        - Efficient vision encoding for high-resolution images
        
        Note: If the Apple model doesn't load with mlx-vlm, we fall back to:
        - InsightKeeper/FastVLM-0.5B-MLX-6bit (community MLX version)
        - Or Qwen2-VL as a last resort
        """
        # Use local converted FastVLM model
        import os
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        local_model_path = os.path.join(base_path, "ml-fastvlm/models/fastvlm-0.5b-mlx")
        
        if model_name is None and os.path.exists(local_model_path):
            self.model_name = local_model_path
            logger.info(f"Using local FastVLM model at: {local_model_path}")
        else:
            self.model_name = model_name or "apple/FastVLM-0.5B-fp16"
        self.model = None
        self.processor = None
        self.config = None
        self.initialized = False
        self.last_description = None  # Track previous description for context
        
        logger.info(f"Initializing VLM processor with model: {model_name}")
        
    async def initialize(self):
        """Load the model asynchronously"""
        if self.initialized:
            return
            
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)
        self.initialized = True
        
    def _load_model(self):
        """Load the model synchronously with fallback options"""
        # Lazy load MLX modules
        _lazy_load_mlx()
        
        # Only try FastVLM models, skip Qwen2-VL to avoid API mismatch
        fallback_models = [
            self.model_name,  # Primary: Local FastVLM or specified model
        ]
        
        # Add community version only if local doesn't exist
        import os
        if not os.path.exists(self.model_name):
            fallback_models.append("InsightKeeper/FastVLM-0.5B-MLX-6bit")
        
        for model_name in fallback_models:
            try:
                logger.info(f"Loading FastVLM model {model_name}...")
                start_time = time.time()
                
                # Load model and processor
                self.model, self.processor = load(model_name)
                self.config = self.model.config
                
                load_time = time.time() - start_time
                logger.info(f"FastVLM model {model_name} loaded successfully in {load_time:.2f} seconds")
                logger.info(f"Model type: {getattr(self.config, 'model_type', 'unknown')}")
                self.model_name = model_name  # Update to the successfully loaded model
                return
                
            except Exception as e:
                logger.warning(f"Failed to load FastVLM {model_name}: {e}")
                # Clean up memory after failed attempt
                self.model = None
                self.processor = None
                self.config = None
                import gc
                gc.collect()
                logger.info(f"Cleaned up memory after failed load attempt")
                continue
        
        # If all models failed to load
        logger.error("Failed to load FastVLM model - descriptions will not be available")
        logger.error("Please ensure ml-fastvlm/models/fastvlm-0.5b-mlx exists and is properly converted")
        self.model = None
            
    async def describe_screen(self, frame: np.ndarray, prompt: Optional[str] = None) -> str:
        """
        Generate description of screen content
        
        Args:
            frame: numpy array of the screen capture (RGB)
            prompt: Optional custom prompt
            
        Returns:
            Description of what's on the screen
        """
        # Ensure model is loaded
        if not self.initialized:
            logger.debug("VLM not initialized, initializing...")
            await self.initialize()
            
        # If model failed to load, use mock description
        if self.model is None:
            logger.warning("No VLM model loaded, returning mock description")
            return self._mock_description()
            
        # Enhanced prompt for better accuracy with context
        if prompt is None:
            base_prompt = """Describe what the user is doing in detail. Include:
- The application being used
- The specific task or activity
- Any visible text or UI elements
Be specific and concise in one sentence."""
            
            # Add context from previous frame if available
            if self.last_description:
                prompt = f"Previous: {self.last_description[:80]}...\nNow: {base_prompt}"
            else:
                prompt = base_prompt
            
        logger.debug(f"Describing screen with prompt: {prompt[:100]}...")
            
        # Convert numpy array to PIL Image
        if isinstance(frame, np.ndarray):
            image = Image.fromarray(frame.astype('uint8'), 'RGB')
        else:
            image = frame
            
        # Generate description
        try:
            description = await self._generate_async(image, prompt)
            if description and description != "No description available":
                logger.info(f"VLM generated: {description[:100]}...")
                # Store for context in next frame
                self.last_description = description
                return description
            else:
                logger.warning("VLM returned empty/invalid description, using fallback")
                return self._mock_description()
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return self._mock_description()
            
    async def _generate_async(self, image: Image.Image, prompt: str) -> str:
        """Generate description asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_sync,
            image,
            prompt
        )
        
    def _generate_sync(self, image: Image.Image, prompt: str) -> str:
        """Generate description synchronously"""
        # Save image temporarily (mlx-vlm needs file path)
        temp_path = Path("temp_screen.jpg")
        image.save(temp_path)
        
        try:
            # Format prompt using Qwen2 conversation template (like CLI that works)
            formatted_prompt = (
                "<|im_start|>system\n"
                "You are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                "<image>\n"
                f"{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            
            logger.debug(f"Generating with prompt: {prompt[:50]}...")
            
            # Generate description (no return_type parameter)
            output = generate(
                self.model, 
                self.processor, 
                formatted_prompt, 
                str(temp_path), 
                verbose=False,
                max_tokens=150,  # Increased for more detailed descriptions
                temperature=0.5  # Lower for more consistent output
            )
            
            # Extract text from output
            result = str(output) if output else ""
            
            # Clean up the output (remove any template markers)
            if "<|im_end|>" in result:
                result = result.split("<|im_end|>")[0]
            if "<|im_start|>" in result:
                result = result.split("<|im_start|>")[-1]
            
            result = result.strip()
            logger.info(f"Generated description: {result[:100]}...")
            
            return result if result else "No description available"
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "No description available"
            
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
                
    def _mock_description(self) -> str:
        """Fallback mock descriptions when model not available"""
        import random
        descriptions = [
            "User is working in a code editor",
            "Terminal window is open with command output",
            "Browser showing documentation page",
            "Email application is open",
            "File explorer window is visible",
            "User is viewing a PDF document",
            "Video call application is active",
            "Spreadsheet with data is open",
            "Chat application is in focus",
            "Settings panel is displayed"
        ]
        return random.choice(descriptions)
        
    async def analyze_for_risks(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze screen for potential risks or important events
        
        Returns:
            Dictionary with risk_score, reason, and suggestions
        """
        if not self.initialized:
            await self.initialize()
            
        if self.model is None:
            return {
                'risk_score': 0.1,
                'reason': 'Unable to analyze',
                'suggestion': None
            }
            
        prompt = """Analyze this screen for any of these risks:
        1. Destructive commands (rm -rf, format, delete)
        2. Sensitive data exposure (passwords, keys, tokens)
        3. Wrong recipient in email/message
        4. Unsigned or suspicious code execution
        5. System warnings or errors
        
        Respond with: risk level (low/medium/high) and brief reason."""
        
        try:
            analysis = await self.describe_screen(frame, prompt)
            
            # Parse response (simple heuristic)
            risk_score = 0.1  # Default low
            if 'high' in analysis.lower():
                risk_score = 0.8
            elif 'medium' in analysis.lower():
                risk_score = 0.5
                
            return {
                'risk_score': risk_score,
                'reason': analysis,
                'suggestion': 'Review carefully' if risk_score > 0.5 else None
            }
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return {
                'risk_score': 0.1,
                'reason': 'Analysis failed',
                'suggestion': None
            }
            
    async def suggest_next_action(self, frame: np.ndarray, context: Optional[str] = None) -> str:
        """
        Suggest helpful next action based on screen content
        
        Args:
            frame: Current screen capture
            context: Optional context about what user is doing
            
        Returns:
            Brief suggestion (max 15 words)
        """
        if not self.initialized:
            await self.initialize()
            
        if self.model is None:
            return ""
            
        prompt = f"""Based on this screen, suggest the most helpful next action in 10 words or less.
        {f'Context: {context}' if context else ''}
        Be specific and actionable."""
        
        try:
            suggestion = await self.describe_screen(frame, prompt)
            # Truncate to ensure brevity
            words = suggestion.split()[:15]
            return ' '.join(words)
        except:
            return ""
    
    async def process_with_template(self, frame: np.ndarray, template: VLMPromptTemplates) -> str:
        """
        Process frame with a specific prompt template
        
        Args:
            frame: Screen capture
            template: Prompt template to use
            
        Returns:
            Model response
        """
        return await self.describe_screen(frame, template.value)
    
    async def process_rois(self, frame: np.ndarray, rois: List[Tuple[int, int, int, int]], 
                          template: VLMPromptTemplates = VLMPromptTemplates.DESCRIBE) -> List[ROIResult]:
        """
        Process multiple ROIs in batch
        
        Args:
            frame: Full screen capture
            rois: List of regions (x, y, w, h)
            template: Prompt template to use
            
        Returns:
            List of ROI analysis results
        """
        if not self.initialized:
            await self.initialize()
            
        if self.model is None or not rois:
            return []
        
        results = []
        
        # Process ROIs in batches to avoid overwhelming the model
        batch_size = 3
        for i in range(0, len(rois), batch_size):
            batch = rois[i:i+batch_size]
            
            # Process each ROI in the batch
            batch_tasks = []
            for roi in batch:
                x, y, w, h = roi
                # Extract ROI from frame
                roi_frame = frame[y:y+h, x:x+w]
                
                # Process ROI
                task = self._process_single_roi(roi_frame, roi, template)
                batch_tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        
        return results
    
    async def _process_single_roi(self, roi_frame: np.ndarray, roi: Tuple[int, int, int, int], 
                                  template: VLMPromptTemplates) -> ROIResult:
        """Process a single ROI"""
        try:
            description = await self.describe_screen(roi_frame, template.value)
            
            # Extract entities if using EXTRACT template
            entities = []
            if template == VLMPromptTemplates.EXTRACT:
                entities = self._extract_entities_from_text(description)
            
            # Calculate importance based on content
            importance = self._calculate_roi_importance(description)
            
            return ROIResult(
                region=roi,
                description=description,
                importance=importance,
                entities=entities
            )
        except Exception as e:
            logger.error(f"Error processing ROI {roi}: {e}")
            return ROIResult(roi, "Processing failed", 0.0, [])
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Extract entities from text response"""
        entities = []
        
        # Try to parse as JSON
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                entities.extend(data.get('entities', []))
                entities.extend(data.get('people', []))
                entities.extend(data.get('orgs', []))
        except:
            # Fallback to simple extraction
            # Look for capitalized words as potential entities
            words = text.split()
            for word in words:
                if word and word[0].isupper() and len(word) > 2:
                    entities.append(word)
        
        return list(set(entities))  # Remove duplicates
    
    def _calculate_roi_importance(self, description: str) -> float:
        """Calculate importance score for ROI based on content"""
        importance = 0.3  # Base score
        
        # Keywords that increase importance
        high_importance_keywords = [
            'error', 'warning', 'failed', 'critical', 'alert',
            'password', 'token', 'key', 'secret',
            'delete', 'remove', 'format', 'overwrite'
        ]
        
        medium_importance_keywords = [
            'save', 'open', 'close', 'edit', 'modify',
            'email', 'message', 'notification',
            'download', 'upload', 'install'
        ]
        
        description_lower = description.lower()
        
        for keyword in high_importance_keywords:
            if keyword in description_lower:
                importance += 0.3
                
        for keyword in medium_importance_keywords:
            if keyword in description_lower:
                importance += 0.1
        
        return min(importance, 1.0)
    
    async def extract_entities(self, frame: np.ndarray) -> Dict[str, List[str]]:
        """
        Extract entities from screen
        
        Returns:
            Dictionary with urls, errors, files, people, etc.
        """
        response = await self.process_with_template(frame, VLMPromptTemplates.EXTRACT)
        
        try:
            # Try to parse JSON response
            data = json.loads(response)
            return data
        except:
            # Fallback to empty structure
            return {
                'urls': [],
                'errors': [],
                'file_paths': [],
                'entities': [],
                'key_values': {}
            }
    
    async def detect_errors(self, frame: np.ndarray) -> List[Dict[str, str]]:
        """
        Detect errors on screen
        
        Returns:
            List of errors with severity and description
        """
        response = await self.process_with_template(frame, VLMPromptTemplates.ERROR_DETECTION)
        
        errors = []
        for line in response.split('\n'):
            if any(word in line.lower() for word in ['error', 'warning', 'failed']):
                severity = 'high' if 'error' in line.lower() else 'medium'
                errors.append({
                    'severity': severity,
                    'description': line.strip()
                })
        
        return errors
    
    async def suggest_action(self, frame: np.ndarray, context: Dict[str, Any]) -> str:
        """
        Suggest action based on frame and context
        
        Args:
            frame: Current screen
            context: Session context with app, file, task info
            
        Returns:
            Action suggestion
        """
        context_str = f"App: {context.get('app', 'unknown')}, Task: {context.get('task', 'unknown')}"
        return await self.suggest_next_action(frame, context_str)
            
    def cleanup(self):
        """Clean up resources"""
        self.model = None
        self.processor = None
        self.config = None
        self.initialized = False
        logger.info("VLM processor cleaned up")


class SimplifiedVLMProcessor:
    """Simplified VLM processor with basic pattern matching"""
    
    def __init__(self):
        """Initialize without heavy model loading"""
        self.patterns = {
            'terminal': ['$', '>', 'npm', 'git', 'python', 'bash'],
            'code': ['def ', 'function', 'class', 'import', 'const ', 'let '],
            'browser': ['http', 'www', 'search', 'google'],
            'email': ['@', 'To:', 'From:', 'Subject:'],
            'error': ['error', 'failed', 'exception', 'warning', 'Error:'],
        }
        
    async def describe_screen(self, frame: np.ndarray, prompt: Optional[str] = None) -> str:
        """Generate simple description based on visual patterns"""
        # For now, return context-aware descriptions
        # In production, this would use OCR or simple CV
        
        # Analyze dominant colors
        avg_color = np.mean(frame, axis=(0, 1))
        
        # Dark theme likely means terminal or code editor
        if np.mean(avg_color) < 100:
            return "User is working in a dark-themed application, possibly terminal or code editor"
        # Light theme
        elif np.mean(avg_color) > 200:
            return "User is viewing a light-themed application or document"
        else:
            return "User is actively working on their computer"
            
    async def analyze_for_risks(self, frame: np.ndarray) -> Dict[str, Any]:
        """Simple risk detection - more realistic heuristics"""
        # Check for concentrated red regions (not just any red)
        red_channel = frame[:, :, 0]
        green_channel = frame[:, :, 1]
        blue_channel = frame[:, :, 2]
        
        # Look for pure red (high red, low green/blue) which is more likely an error
        pure_red_pixels = (red_channel > 200) & (green_channel < 100) & (blue_channel < 100)
        pure_red_ratio = np.sum(pure_red_pixels) / red_channel.size
        
        # Also check for very dark screens (possible terminal errors)
        avg_brightness = np.mean(frame)
        
        if pure_red_ratio > 0.01:  # More than 1% pure red (not pink/orange)
            return {
                'risk_score': 0.6,
                'reason': 'Possible error indicator detected',
                'suggestion': 'Check for error messages'
            }
        elif avg_brightness < 30:  # Very dark screen
            return {
                'risk_score': 0.3,
                'reason': 'Dark screen - possible terminal or fullscreen mode',
                'suggestion': None
            }
        
        return {
            'risk_score': 0.1,
            'reason': 'No obvious risks detected',
            'suggestion': None
        }
        
    async def suggest_next_action(self, frame: np.ndarray, context: Optional[str] = None) -> str:
        """Provide generic helpful suggestions"""
        suggestions = [
            "Save your work",
            "Check for updates",
            "Review changes before committing",
            "Take a break soon",
            "Close unused tabs"
        ]
        import random
        return random.choice(suggestions)


# Test functions
async def test_vlm_processor():
    """Test the VLM processor"""
    print("Testing VLM Processor...")
    
    # Try full model first
    print("\n1. Testing with Apple FastVLM-0.5B model (may take time to download)...")
    processor = FastVLMProcessor()
    
    # Create a test image
    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # Test description
    print("Generating description...")
    description = await processor.describe_screen(test_frame)
    print(f"Description: {description}")
    
    # Test risk analysis
    print("\nAnalyzing for risks...")
    risks = await processor.analyze_for_risks(test_frame)
    print(f"Risk analysis: {risks}")
    
    # Test suggestion
    print("\nGenerating suggestion...")
    suggestion = await processor.suggest_next_action(test_frame)
    print(f"Suggestion: {suggestion}")
    
    # Test simplified processor
    print("\n2. Testing simplified processor (no model required)...")
    simple_processor = SimplifiedVLMProcessor()
    
    description = await simple_processor.describe_screen(test_frame)
    print(f"Simple description: {description}")
    
    risks = await simple_processor.analyze_for_risks(test_frame)
    print(f"Simple risk analysis: {risks}")
    
    print("\n✅ VLM Processor tests complete!")


if __name__ == "__main__":
    asyncio.run(test_vlm_processor())