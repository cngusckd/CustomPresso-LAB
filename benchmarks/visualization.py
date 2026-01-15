import matplotlib.pyplot as plt
import numpy as np
import os

def plot_metrics(results, save_dir='benchmarks/plots'):
    """
    results: dict { 'ModelName': {'latency': ms, 'fps': val, 'vram': mb, 'size': mb, 'map': val} }
    """
    os.makedirs(save_dir, exist_ok=True)
    
    models = list(results.keys())
    latencies = [results[m].get('latency', 0) for m in models]
    fps = [results[m].get('fps', 0) for m in models]
    vram = [results[m].get('vram', 0) for m in models]
    
    # Latency Plot
    plt.figure(figsize=(10, 6))
    plt.bar(models, latencies, color='skyblue')
    plt.title('Inference Latency (ms) - Lower is Better')
    plt.ylabel('Time (ms)')
    plt.savefig(f'{save_dir}/latency_comparison.png')
    plt.close()
    
    # FPS Plot
    plt.figure(figsize=(10, 6))
    plt.bar(models, fps, color='lightgreen')
    plt.title('Throughput (FPS) - Higher is Better')
    plt.ylabel('FPS')
    plt.savefig(f'{save_dir}/throughput_comparison.png')
    plt.close()

    # VRAM Plot
    plt.figure(figsize=(10, 6))
    plt.bar(models, vram, color='salmon')
    plt.title('Peak VRAM (MB) - Lower is Better')
    plt.ylabel('Memory (MB)')
    plt.savefig(f'{save_dir}/vram_comparison.png')
    plt.close()
    
    print(f"Plots saved to {save_dir}")

def plot_pareto(results, save_dir='benchmarks/plots'):
    os.makedirs(save_dir, exist_ok=True)
    
    models = list(results.keys())
    latencies = [results[m].get('latency', 0) for m in models]
    maps = [results[m].get('map', 0) for m in models]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(latencies, maps, color='purple', s=100)
    
    for i, txt in enumerate(models):
        plt.annotate(txt, (latencies[i], maps[i]), xytext=(5, 5), textcoords='offset points')
        
    plt.title('Pareto Front: Latency vs Accuracy (mAP)')
    plt.xlabel('Latency (ms)')
    plt.ylabel('mAP@50')
    plt.grid(True)
    plt.savefig(f'{save_dir}/pareto_chart.png')
    plt.close()
