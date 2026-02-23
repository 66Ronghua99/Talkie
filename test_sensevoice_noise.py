import asyncio
import numpy as np
import time
import os
import sys

# Ensure we can import from src
sys.path.insert(0, os.path.abspath("src"))

from talkie.perception.asr import SherpaOnnxASR

async def main():
    print("Initializing SherpaOnnxASR (SenseVoice)...")
    asr = SherpaOnnxASR(
        num_threads=4,
        sample_rate=16000,
        use_itn=True
    )
    
    # Load the model
    await asr.load()
    print("Model loaded successfully.\n")
    
    # Test 1: Absolute silence
    print("--- Test 1: Absolute Silence (1 second) ---")
    silence = np.zeros(16000, dtype=np.int16).tobytes()
    start_time = time.time()
    result = await asr.recognize(silence)
    latency = time.time() - start_time
    print(f"Result: '{result.text}' (Confidence: {result.confidence})")
    print(f"Latency: {latency:.4f}s\n")
    
    # Test 2: Low-level white noise (simulating quiet room noise)
    print("--- Test 2: Low-level White Noise (1 second) ---")
    # Low amplitude noise, max amplitude 300 out of 32767
    low_noise = np.random.uniform(-300, 300, 16000).astype(np.int16).tobytes()
    start_time = time.time()
    result = await asr.recognize(low_noise)
    latency = time.time() - start_time
    print(f"Result: '{result.text}' (Confidence: {result.confidence})")
    print(f"Latency: {latency:.4f}s\n")
    
    # Test 3: Moderate white noise (simulating static or louder room noise)
    print("--- Test 3: Moderate White Noise (1 second) ---")
    # Moderate amplitude noise, max amplitude 2000 out of 32767
    med_noise = np.random.uniform(-2000, 2000, 16000).astype(np.int16).tobytes()
    start_time = time.time()
    result = await asr.recognize(med_noise)
    latency = time.time() - start_time
    print(f"Result: '{result.text}' (Confidence: {result.confidence})")
    print(f"Latency: {latency:.4f}s\n")

    # Expected behavior if SenseVoice has perfect internal VAD/Noise rejection:
    # All results should be empty strings ('').
    
    # Clean up
    await asr.close()
    print("Test complete.")

if __name__ == "__main__":
    asyncio.run(main())
