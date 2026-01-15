import torch
import time
import os
import numpy as np

class Benchmarker:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'

    def measure_latency(self, model, input_shape=(1, 3, 640, 640), warmups=10, runs=50):
        print(f"Measuring Latency (Runs: {runs})...")
        model.to(self.device)
        model.eval()
        dummy_input = torch.randn(input_shape).to(self.device).float()

        # Warmup
        with torch.no_grad():
            for _ in range(warmups):
                _ = model(dummy_input)

        # Measure
        latencies = []
        with torch.no_grad():
            if self.device == 'cuda':
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                for _ in range(runs):
                    starter.record()
                    _ = model(dummy_input)
                    ender.record()
                    torch.cuda.synchronize()
                    latencies.append(starter.elapsed_time(ender)) # ms
            else:
                for _ in range(runs):
                    start = time.perf_counter()
                    _ = model(dummy_input)
                    end = time.perf_counter()
                    latencies.append((end - start) * 1000) # ms

        avg_latency = np.mean(latencies)
        print(f"Average Latency: {avg_latency:.4f} ms")
        return avg_latency

    def measure_throughput(self, model, batch_size=16, duration=5.0):
        print(f"Measuring Throughput (Duration: {duration}s, Batch: {batch_size})...")
        model.to(self.device)
        model.eval()
        dummy_input = torch.randn(batch_size, 3, 640, 640).to(self.device).float()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if self.device == 'cuda' else None
        
        start_time = time.perf_counter()
        frames = 0
        while time.perf_counter() - start_time < duration:
            with torch.no_grad():
                _ = model(dummy_input)
            frames += batch_size
            
        torch.cuda.synchronize() if self.device == 'cuda' else None
        end_time = time.perf_counter()
        
        fps = frames / (end_time - start_time)
        print(f"Throughput: {fps:.2f} FPS")
        return fps

    def measure_vram(self, model, input_shape=(1, 3, 640, 640)):
        if self.device != 'cuda':
            return 0.0
            
        print("Measuring Peak VRAM...")
        torch.cuda.reset_peak_memory_stats()
        model.to(self.device)
        model.eval()
        dummy_input = torch.randn(input_shape).to(self.device).float()
        
        with torch.no_grad():
            _ = model(dummy_input)
            
        max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
        print(f"Peak VRAM: {max_mem:.2f} MB")
        return max_mem

    def measure_size(self, state_dict=None, path=None):
        if path and os.path.exists(path):
            size = os.path.getsize(path) / 1024 / 1024 # MB
        elif state_dict:
            # Estimate
            torch.save(state_dict, "temp.pt")
            size = os.path.getsize("temp.pt") / 1024 / 1024
            os.remove("temp.pt")
        else:
            return 0.0
        print(f"Model Size: {size:.2f} MB")
        return size
