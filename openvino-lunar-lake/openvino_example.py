import openvino_genai as ov_genai
import time
from pathlib import Path
import argparse
import sys
import os
import requests
import json
import subprocess
import threading
import time

parser = argparse.ArgumentParser(description="Run OpenVINO LLM inference on selected device.")
parser.add_argument('--device', type=str, default='GPU', choices=['CPU', 'GPU', 'NPU'], help='Device to run inference on (CPU, GPU, NPU)')
parser.add_argument('--prompt', type=str, default='Deep learning is ', help='Prompt for the model')
parser.add_argument('--precision', type=str, default='int4', choices=['int4', 'int8'], help='Precision for CPU/GPU (ignored for NPU)')
parser.add_argument('--ovms', action='store_true', help='Use OVMS server with OpenAI-compatible API')
parser.add_argument('--ovms-port', type=int, default=9001, help='OVMS server port')
parser.add_argument('--start-server', action='store_true', help='Start OVMS server before inference')
args = parser.parse_args()

device = args.device.upper()
prompt = args.prompt
precision = args.precision.lower()
use_ovms = args.ovms
ovms_port = args.ovms_port
start_server = args.start_server

# Print configuration summary
print("=" * 50)
print("OpenVINO LLM Inference Configuration")
print("=" * 50)
print(f"Mode: {'OVMS Server' if use_ovms else 'Local OpenVINO'}")
print(f"Device: {device}")
print(f"Precision: {precision}")
print(f"Prompt: {prompt}")

# Determine model path for display
if device == 'NPU':
    model_path = "./models/supernova-npu-int4-channel-wise"
elif device == 'GPU':
    if precision == 'int8':
        model_path = "./models/supernova-gpu-int8"
    else:
        model_path = "./models/supernova-gpu-int4"
else:  # CPU
    if precision == 'int8':
        model_path = "./models/supernova-cpu-int8"
    else:
        model_path = "./models/supernova-cpu-int4"

print(f"Model Path: {model_path}")

if use_ovms:
    print(f"OVMS Port: {ovms_port}")
    print(f"Start Server: {start_server}")
    if device == 'NPU':
        print("NPU: Using MediaPipe configuration with supernova-npu-int4-channel-wise")
print("=" * 50)

def start_ovms_server(device, precision):
    """Start OVMS server with the specified model"""
    
    # Use model-specific configuration files
    if device == 'NPU':
        config_path = "./models/config-npu-int4.json"
        model_name = "supernova-npu-int4-channel-wise"
        print("Using NPU with MediaPipe configuration")
    elif device == 'GPU':
        if precision == 'int8':
            config_path = "./models/config-gpu-int8.json"
            model_name = "supernova-gpu-int8"
        else:
            config_path = "./models/config-gpu-int4.json"
            model_name = "supernova-gpu-int4"
        print(f"Using GPU {precision.upper()} with MediaPipe configuration")
    else:  # CPU
        if precision == 'int8':
            config_path = "./models/config-cpu-int8.json"
            model_name = "supernova-cpu-int8"
        else:
            config_path = "./models/config-cpu-int4.json"
            model_name = "supernova-cpu-int4"
        print(f"Using CPU {precision.upper()} with MediaPipe configuration")
    
    # Use device-specific config file
    cmd = [
        "./ovms/ovms",
        "--rest_port", str(ovms_port),
        "--config_path", config_path
    ]
    
    print(f"Starting OVMS server with device-specific config")
    print(f"Config path: {config_path}")
    print(f"Model: {model_name}")
    print(f"Command: {' '.join(cmd)}")
    
    # Create log file with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"ovms_server_{timestamp}.log"
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Start a thread to save server logs to file
    def save_logs_to_file(pipe, log_file, prefix):
        with open(log_file, 'a', encoding='utf-8') as f:
            for line in iter(pipe.readline, ''):
                if line:
                    log_entry = f"[{prefix}] {line.strip()}"
                    f.write(log_entry + '\n')
                    f.flush()  # Ensure logs are written immediately
    
    import threading
    stdout_thread = threading.Thread(target=save_logs_to_file, args=(process.stdout, log_file, "OVMS-OUT"))
    stderr_thread = threading.Thread(target=save_logs_to_file, args=(process.stderr, log_file, "OVMS-ERR"))
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()
    
    print(f"OVMS server logs will be saved to: {log_file}")
    
    # Wait for server to start and check if it's ready
    print("Waiting for OVMS server to start...")
    for i in range(120):  # Wait up to 120 seconds (2 minutes)
        time.sleep(1)
        try:
            # Try to connect to the server
            test_response = requests.get(f"http://localhost:{ovms_port}/v1/config", timeout=5)
            if test_response.status_code == 200:
                print(f"OVMS server is ready on port {ovms_port}")
                break
        except requests.exceptions.RequestException as e:
            if i % 10 == 0:  # Print status every 10 seconds
                print(f"Still waiting for server... ({i+1}/120 seconds) - Error: {e}")
            continue
    else:
        print("Warning: OVMS server may not be fully ready after 120 seconds")
        print(f"Check the log file for details: {log_file}")
    
    # Additional wait to ensure graph is fully loaded
    print("Waiting additional 30 seconds to ensure graph is fully loaded...")
    time.sleep(30)
    print("Server should now be fully ready for inference")
    
    return process, log_file

def check_ovms_server(port=9001):
    """Check if OVMS server is running and accessible"""
    try:
        # Try to get config (this endpoint works with MediaPipe configuration)
        response = requests.get(f"http://localhost:{port}/v1/config", timeout=5)
        print(f"Server check - Status: {response.status_code}")
        print(f"Server check - Response: {response.text}")
        
        # Also try to get the models endpoint to see what's available
        try:
            models_response = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
            print(f"Models endpoint - Status: {models_response.status_code}")
            print(f"Models endpoint - Response: {models_response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Models endpoint failed: {e}")
        
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Server check failed: {e}")
        return False

def call_ovms_openai_api(prompt, port=9001, device="GPU", precision="int4"):
    """Call OVMS server using OpenAI-compatible API"""
    url = f"http://localhost:{port}/v3/chat/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Determine model name based on device and precision
    # These must match exactly the "name" field in the mediapipe_config_list of the config file
    if device == 'NPU':
        model_name = "supernova-npu-int4-channel-wise"
    elif device == 'GPU':
        if precision == 'int8':
            model_name = "supernova-gpu-int8"
        else:
            model_name = "supernova-gpu-int4"
    else:  # CPU
        if precision == 'int8':
            model_name = "supernova-cpu-int8"
        else:
            model_name = "supernova-cpu-int4"
    
    data = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "stream": True
    }
    
    try:
        print(f"Making API call to: {url}")
        print(f"Request data: {json.dumps(data, indent=2)}")
        response = requests.post(url, headers=headers, json=data, stream=True, timeout=30)
        response.raise_for_status()
        
        response_text = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    if data_str == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data_str)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content = delta['content']
                                print(content, end="", flush=True)
                                response_text += content
                    except json.JSONDecodeError:
                        continue
        
        return response_text
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling OVMS API: {e}")
        print(f"Response status code: {getattr(e.response, 'status_code', 'N/A')}")
        print(f"Response text: {getattr(e.response, 'text', 'N/A')}")
        return None

if use_ovms:
    server_process = None
    log_file = None
    
    if start_server:
        server_process, log_file = start_ovms_server(device, precision)
    
    print(f"Using OVMS server on port {ovms_port}")
    print(f"Prompt: {prompt}")
    
    # Check if server is accessible
    print("Checking OVMS server status...")
    if not check_ovms_server(ovms_port):
        print("OVMS server is not accessible. Please ensure it's running.")
        if server_process:
            try:
                server_process.terminate()
                server_process.wait(timeout=10)
                print("OVMS server stopped gracefully")
            except subprocess.TimeoutExpired:
                print("Force killing OVMS server...")
                server_process.kill()
                server_process.wait()
                print("OVMS server force stopped")
            except Exception as e:
                print(f"Error stopping OVMS server: {e}")
        sys.exit(1)
    
    tick = time.time()
    response_text = call_ovms_openai_api(prompt, ovms_port, device, precision)
    tock = time.time()
    
    if response_text:
        words = len(response_text.split(" "))
        print(f"\nGenerated {words} words in {tock-tick:.2f} seconds")
        print(f"{words/(tock-tick):.1f} words per second")
    else:
        print("Failed to get response from OVMS server")
    
    # Clean up server if we started it
    if server_process:
        try:
            server_process.terminate()
            server_process.wait(timeout=10)
            print("OVMS server stopped gracefully")
        except subprocess.TimeoutExpired:
            print("Force killing OVMS server...")
            server_process.kill()
            server_process.wait()
            print("OVMS server force stopped")
        except Exception as e:
            print(f"Error stopping OVMS server: {e}")
    
else:
    # Local OpenVINO inference
    if device == 'NPU':
        model_path = Path("./models/supernova-npu-int4-channel-wise")
        model_name = "supernova-npu-int4-channel-wise"
        pipe = ov_genai.LLMPipeline(model_path, "NPU", CACHE_DIR=".npucache", GENERATE_HINT="BEST_PERF")
    else:
        if device == 'GPU':
            if precision == 'int8':
                model_path = Path("./models/supernova-gpu-int8")
                model_name = "supernova-gpu-int8"
            else:
                model_path = Path("./models/supernova-gpu-int4")
                model_name = "supernova-gpu-int4"
        else:  # CPU
            if precision == 'int8':
                model_path = Path("./models/supernova-cpu-int8")
                model_name = "supernova-cpu-int8"
            else:
                model_path = Path("./models/supernova-cpu-int4")
                model_name = "supernova-cpu-int4"
        pipe = ov_genai.LLMPipeline(model_path, device)
    
    print(f"Model: {model_name}")
    print(f"Model Path: {model_path}")

    streamer = lambda x: print(x, end="", flush=True)

    tick = time.time()
    response = pipe.generate(prompt, streamer=streamer, max_new_tokens=512)
    tock = time.time()

    response_text = str(response)
    words = len(response_text.split(" "))
    print(response_text)
    print(f"Generated {words} words in {tock-tick:.2f} seconds")
    print(f"{words/(tock-tick):.1f} words per second")
