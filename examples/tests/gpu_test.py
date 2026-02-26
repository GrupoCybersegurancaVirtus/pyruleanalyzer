import numpy as np
import time

# Tenta importar o CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy não encontrado! O teste de GPU será ignorado.")

def run_benchmark(size=5000):
    print(f"--- Iniciando Benchmark (Matrizes {size}x{size}) ---")
    
    # --- TESTE CPU (NumPy) ---
    print("\n[CPU] Executando com NumPy...")
    # Criando matrizes na memória RAM
    a_cpu = np.random.rand(size, size).astype(np.float32)
    b_cpu = np.random.rand(size, size).astype(np.float32)
    
    start_cpu = time.time()
    # Multiplicação de matrizes + Soma
    result_cpu = np.dot(a_cpu, b_cpu)
    sum_cpu = np.sum(result_cpu)
    end_cpu = time.time()
    
    time_cpu = end_cpu - start_cpu
    print(f"Tempo CPU: {time_cpu:.4f} segundos")

    # --- TESTE GPU (CuPy) ---
    if CUPY_AVAILABLE:
        print("\n[GPU] Executando com CuPy...")
        try:
            # Criando matrizes diretamente na memória da VRAM (GPU)
            a_gpu = cp.random.rand(size, size).astype(cp.float32)
            b_gpu = cp.random.rand(size, size).astype(cp.float32)
            
            # Primeira execução (Warm-up): CuPy compila os kernels CUDA aqui
            _ = cp.dot(a_gpu, b_gpu)
            cp.cuda.Stream.null.synchronize() # Garante que a GPU terminou
            
            start_gpu = time.time()
            # Multiplicação de matrizes + Soma na GPU
            result_gpu = cp.dot(a_gpu, b_gpu)
            sum_gpu = cp.sum(result_gpu)
            cp.cuda.Stream.null.synchronize() # Sincroniza para medição real
            end_gpu = time.time()
            
            time_gpu = end_gpu - start_gpu
            print(f"Tempo GPU: {time_gpu:.4f} segundos")
            
            # Resultados
            speedup = time_cpu / time_gpu
            print(f"\nResultado: A GPU foi {speedup:.2f}x mais rápida que a CPU!")
            
        except Exception as e:
            print(f"Erro na GPU: {e}")
    else:
        print("\nTeste de GPU não realizado pois o CuPy não está disponível.")

if __name__ == "__main__":
    run_benchmark(size=60000)