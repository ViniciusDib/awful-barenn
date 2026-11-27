// barenn_cuda_opt.cu — Optimized CUDA implementation of Barenn neural network
// Compile: nvcc -arch=sm_86 -O3 -std=c++17 -o barenn_cuda_opt barenn_cuda_opt.cu
//
// Advanced CUDA techniques:
//   1. Shared memory for input caching in forward/backward kernels
//   2. Fused forward kernels (matmul + bias + activation)
//   3. Persistent device memory (weights remain on GPU entire training)
//   4. Pinned (page-locked) host memory for async transfers
//   5. CUDA streams with double-buffered input for transfer/compute overlap
//   6. Warp-level reduction (__shfl_down_sync) in loss computation
//   7. Mini-batch processing for improved GPU utilization (optional)
//   8. CUDA event-based timing with component breakdown
//
// Usage: ./barenn_cuda_opt [train_file test_file [batch_size]]
//   batch_size=1 (default): online SGD, matches original output
//   batch_size>1: mini-batch SGD, better GPU throughput

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// ============================================================================
// CUDA Error Checking
// ============================================================================
#define CUDA_CHECK(call) do {                                             \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(err));             \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)

// ============================================================================
// Constants (match original exactly)
// ============================================================================
constexpr int ENTRADAS = 64;
constexpr int CLASSES  = 10;
constexpr int L1_IN = 64, L1_OUT = 32;
constexpr int L2_IN = 32, L2_OUT = 16;
constexpr int L3_IN = 16, L3_OUT = 10;
constexpr int EPOCAS   = 51;
constexpr float TAXA   = 0.02f;

// ============================================================================
// Data Loading (identical parsing to original)
// ============================================================================
struct Dataset {
    vector<float> images; // flattened [N * ENTRADAS]
    vector<int>   labels; // [N]
    int N = 0;
};

Dataset ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo)
        throw runtime_error("Nao foi possivel abrir " + caminho);
    Dataset ds;
    string linha;
    size_t numero_linha = 0;
    while (getline(arquivo, linha)) {
        ++numero_linha;
        istringstream campos(linha);
        vector<float> img;
        for (int i = 0; i < ENTRADAS; ++i) {
            float valor; char sep;
            if (!(campos >> valor >> sep) || sep != ',' ||
                !isfinite(valor) || valor < 0 || valor > 16) break;
            img.push_back(valor / 16.0f);
        }
        int rotulo;
        if ((int)img.size() != ENTRADAS ||
            !(campos >> rotulo) || rotulo < 0 || rotulo >= CLASSES ||
            !(campos >> ws).eof()) {
            throw runtime_error(caminho + ":" + to_string(numero_linha) +
                " — esperados 64 atributos e rotulo 0-9");
        }
        ds.images.insert(ds.images.end(), img.begin(), img.end());
        ds.labels.push_back(rotulo);
    }
    ds.N = (int)ds.labels.size();
    if (arquivo.bad() || ds.N == 0)
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    return ds;
}

// ============================================================================
// Device Helpers
// ============================================================================

/// Warp-level sum using shuffle-down. Result valid only in lane 0.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// ============================================================================
// Kernels — Single-Sample (Online SGD)
// ============================================================================

/// Fused forward: matmul + bias + ReLU/linear.
/// Shared memory caches input so all output neurons read it from fast SRAM.
__global__ void kernel_forward(
    const float* __restrict__ W,      // [out_f * in_f]
    const float* __restrict__ B,      // [out_f]
    const float* __restrict__ input,  // [in_f]
    float* __restrict__ output,       // [out_f]
    float* __restrict__ pre_act,      // [out_f] saved for backward
    int in_f, int out_f, int is_final)
{
    extern __shared__ float s_input[];
    int tid = threadIdx.x;
    for (int i = tid; i < in_f; i += blockDim.x)
        s_input[i] = input[i];
    __syncthreads();

    if (tid < out_f) {
        const float* wr = W + tid * in_f;
        float sum = B[tid];
        for (int j = 0; j < in_f; ++j)
            sum += s_input[j] * wr[j];
        pre_act[tid] = sum;
        output[tid]  = is_final ? sum : fmaxf(0.0f, sum);
    }
}

/// Backward: grad_input then SGD weight/bias update.
/// Shared memory: [delta | saved_input].
/// __syncthreads between grad_input reads and weight writes prevents races.
__global__ void kernel_backward(
    float* __restrict__ W,
    float* __restrict__ B,
    const float* __restrict__ saved_input,
    const float* __restrict__ pre_act,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    int in_f, int out_f, float lr, int is_final)
{
    extern __shared__ float smem[];
    float* s_delta = smem;
    float* s_input = smem + out_f;
    int tid = threadIdx.x;

    if (tid < out_f) {
        float g = grad_out[tid];
        s_delta[tid] = is_final ? g : g * (pre_act[tid] > 0.0f ? 1.0f : 0.0f);
    }
    for (int i = tid; i < in_f; i += blockDim.x)
        s_input[i] = saved_input[i];
    __syncthreads();

    // Phase 2: W^T @ delta
    if (tid < in_f) {
        float sum = 0.0f;
        for (int j = 0; j < out_f; ++j)
            sum += s_delta[j] * W[j * in_f + tid];
        grad_in[tid] = sum;
    }
    __syncthreads();

    // Phase 3: SGD update
    if (tid < out_f) {
        float d = s_delta[tid];
        float* wr = W + tid * in_f;
        for (int j = 0; j < in_f; ++j)
            wr[j] -= lr * d * s_input[j];
        B[tid] -= lr * d;
    }
}

/// Half-MSE loss with warp-level reduction via __shfl_down_sync.
__global__ void kernel_loss(
    const float* __restrict__ scores, int label,
    float* __restrict__ grad,
    float* __restrict__ d_loss_acc, int nc)
{
    int tid = threadIdx.x;
    float err = 0.0f;
    if (tid < nc) {
        err = scores[tid] - (tid == label ? 1.0f : 0.0f);
        grad[tid] = err;
    }
    float loss = warp_reduce_sum(0.5f * err * err);
    if (tid == 0) atomicAdd(d_loss_acc, loss);
}

// ============================================================================
// Kernels — Batched (Mini-batch SGD)
// ============================================================================

/// Batched forward: gridDim.x = batch_size, one block per sample.
__global__ void kernel_forward_batch(
    const float* __restrict__ W,
    const float* __restrict__ B,
    const float* __restrict__ inputs,
    float* __restrict__ outputs,
    float* __restrict__ pre_acts,
    int in_f, int out_f, int is_final)
{
    extern __shared__ float s_input[];
    int b = blockIdx.x, tid = threadIdx.x;
    const float* my_in  = inputs   + b * in_f;
    float*       my_out = outputs  + b * out_f;
    float*       my_z   = pre_acts + b * out_f;

    for (int i = tid; i < in_f; i += blockDim.x)
        s_input[i] = my_in[i];
    __syncthreads();

    if (tid < out_f) {
        const float* wr = W + tid * in_f;
        float sum = B[tid];
        for (int j = 0; j < in_f; ++j) sum += s_input[j] * wr[j];
        my_z[tid]   = sum;
        my_out[tid] = is_final ? sum : fmaxf(0.0f, sum);
    }
}

/// Batched loss with warp reduction + atomic epoch accumulator.
__global__ void kernel_loss_batch(
    const float* __restrict__ scores,
    const int*   __restrict__ labels,
    float*       __restrict__ grads,
    float*       __restrict__ d_loss_acc, int nc)
{
    int b = blockIdx.x, tid = threadIdx.x;
    int label = labels[b];
    const float* s = scores + b * nc;
    float*       g = grads  + b * nc;
    float err = 0.0f;
    if (tid < nc) {
        err = s[tid] - (tid == label ? 1.0f : 0.0f);
        g[tid] = err;
    }
    float loss = warp_reduce_sum(0.5f * err * err);
    if (tid == 0) atomicAdd(d_loss_acc, loss);
}

/// Batched backward: per-sample grad_input + atomic weight-gradient accumulation.
__global__ void kernel_backward_batch(
    const float* __restrict__ W,
    const float* __restrict__ saved_inputs,
    const float* __restrict__ pre_acts,
    const float* __restrict__ grad_outs,
    float*       __restrict__ grad_ins,
    float*       __restrict__ dW,
    float*       __restrict__ dB,
    int in_f, int out_f, int is_final)
{
    extern __shared__ float smem[];
    float* s_delta = smem;
    float* s_input = smem + out_f;
    int b = blockIdx.x, tid = threadIdx.x;

    const float* my_go = grad_outs    + b * out_f;
    const float* my_z  = pre_acts     + b * out_f;
    const float* my_in = saved_inputs + b * in_f;
    float*       my_gi = grad_ins     + b * in_f;

    if (tid < out_f) {
        float g = my_go[tid];
        s_delta[tid] = is_final ? g : g * (my_z[tid] > 0.0f ? 1.0f : 0.0f);
    }
    for (int i = tid; i < in_f; i += blockDim.x)
        s_input[i] = my_in[i];
    __syncthreads();

    if (tid < in_f) {
        float sum = 0.0f;
        for (int j = 0; j < out_f; ++j)
            sum += s_delta[j] * W[j * in_f + tid];
        my_gi[tid] = sum;
    }

    if (tid < out_f) {
        float d = s_delta[tid];
        for (int j = 0; j < in_f; ++j)
            atomicAdd(&dW[tid * in_f + j], d * s_input[j]);
        atomicAdd(&dB[tid], d);
    }
}

/// Apply averaged gradient and zero accumulators.
__global__ void kernel_apply_grads(
    float* __restrict__ W, float* __restrict__ B,
    float* __restrict__ dW, float* __restrict__ dB,
    int total_w, int out_f, float lr_over_bs)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total_w) { W[tid] -= lr_over_bs * dW[tid]; dW[tid] = 0.0f; }
    if (tid < out_f)   { B[tid] -= lr_over_bs * dB[tid]; dB[tid] = 0.0f; }
}

// ============================================================================
// CPU forward (for evaluation matching original output)
// ============================================================================
static void cpu_forward(const float* W, const float* B,
                        const float* in, float* out,
                        int in_f, int out_f, bool is_final) {
    for (int i = 0; i < out_f; ++i) {
        float s = B[i];
        for (int j = 0; j < in_f; ++j) s += in[j] * W[i * in_f + j];
        out[i] = is_final ? s : fmaxf(0.0f, s);
    }
}

// ============================================================================
// Evaluation (identical output format to original)
// ============================================================================
static void avaliar(const float* w1, const float* b1,
                    const float* w2, const float* b2,
                    const float* w3, const float* b3,
                    const Dataset& teste) {
    struct Exemplo { size_t linha; int real, previsto, segundo; };
    array<array<size_t,CLASSES>,CLASSES> conf{};
    vector<Exemplo> exs;
    size_t acertos = 0, ea = 0, ee = 0;
    double perda_total = 0;
    float a1[L1_OUT], a2[L2_OUT], sc[L3_OUT];

    for (int i = 0; i < teste.N; ++i) {
        const float* img = &teste.images[i * ENTRADAS];
        cpu_forward(w1, b1, img, a1, L1_IN, L1_OUT, false);
        cpu_forward(w2, b2, a1,  a2, L2_IN, L2_OUT, false);
        cpu_forward(w3, b3, a2,  sc, L3_IN, L3_OUT, true);
        int prev = (int)(max_element(sc, sc+CLASSES) - sc);
        bool ok = prev == teste.labels[i];
        acertos += ok;
        ++conf[teste.labels[i]][prev];
        for (int c = 0; c < CLASSES; ++c) {
            float e = sc[c] - (c == teste.labels[i] ? 1.0f : 0.0f);
            perda_total += 0.5 * e * e;
        }
        size_t& et = ok ? ea : ee;
        if (et < 3) {
            int seg = prev == 0 ? 1 : 0;
            for (int c = 0; c < CLASSES; ++c)
                if (c != prev && sc[c] > sc[seg]) seg = c;
            exs.push_back({(size_t)i+1, teste.labels[i], prev, seg});
            ++et;
        }
    }
    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0*acertos/teste.N
         << "% | " << acertos << " acertos em " << teste.N << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total/teste.N << "\n";
    cout << "\nALGUMAS INFERENCIAS\nEsperado  Previsto\n";
    for (auto& x : exs)
        cout << right << setw(8) << x.real << "  " << setw(8) << x.previsto << '\n';

    struct Conf { int r, p; size_t n; };
    vector<Conf> errs;
    for (int r = 0; r < CLASSES; ++r)
        for (int p = 0; p < CLASSES; ++p)
            if (r != p && conf[r][p] > 0) errs.push_back({r, p, conf[r][p]});
    stable_sort(errs.begin(), errs.end(),
                [](auto& a, auto& b){ return a.n > b.n; });
    cout << "\nCONFUSOES MAIS FREQUENTES\n";
    if (errs.empty()) { cout << "Nenhuma neste conjunto.\n"; }
    else {
        cout << "Esperado  Previsto  Ocorrencias\n";
        for (size_t i = 0; i < min<size_t>(3, errs.size()); ++i)
            cout << right << setw(8) << errs[i].r
                 << "  " << setw(8) << errs[i].p
                 << "  " << setw(11) << errs[i].n << '\n';
    }
}

// ============================================================================
// Weight init — each layer independently seeds mt19937(1337)
// ============================================================================
static void init_layer(float* w, float* b, int in_f, int out_f) {
    mt19937 gen(1337);
    uniform_real_distribution<float> d(-0.05f, 0.05f);
    for (int i = 0; i < out_f * in_f; ++i) w[i] = d(gen);
    for (int i = 0; i < out_f; ++i) b[i] = 0.0f;
}

// ============================================================================
// Device query
// ============================================================================
static void print_gpu_info() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("=== GPU Info ===\n");
    printf("Device:          %s\n", p.name);
    printf("Compute:         sm_%d%d\n", p.major, p.minor);
    printf("SMs:             %d\n", p.multiProcessorCount);
    printf("Global Memory:   %zu MB\n", p.totalGlobalMem/(1024*1024));
    printf("Shared/Block:    %zu KB\n", p.sharedMemPerBlock/1024);
    printf("Warp Size:       %d\n", p.warpSize);
    printf("================\n\n");
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    if (argc != 1 && argc != 3 && argc != 4) {
        fprintf(stderr, "Uso: %s [treino teste [batch_size]]\n", argv[0]);
        return 1;
    }
    const string arq_treino = argc >= 3 ? argv[1] : "optdigits.tra";
    const string arq_teste  = argc >= 3 ? argv[2] : "optdigits.tes";
    const int batch_size    = argc >= 4 ? atoi(argv[3]) : 1;
    if (batch_size < 1) { fprintf(stderr, "batch_size >= 1\n"); return 1; }

    try {
        Dataset treino = ler_dados(arq_treino);
        Dataset teste  = ler_dados(arq_teste);
        if (filesystem::equivalent(arq_treino, arq_teste))
            throw runtime_error("Use arquivos distintos para treino e teste.");

        print_gpu_info();
        const int N = treino.N;
        printf("Modo: %s (batch_size=%d)\n\n",
               batch_size == 1 ? "Online SGD (streams+pinned)" : "Mini-batch SGD",
               batch_size);
        printf("Tecnicas CUDA avancadas ativas:\n");
        printf("  [x] Shared memory (input caching)\n");
        printf("  [x] Fused kernels (matmul+bias+ativacao)\n");
        printf("  [x] Persistent device memory\n");
        printf("  [x] Warp-level reduction (__shfl_down_sync)\n");
        printf("  [x] CUDA event timing\n");
        if (batch_size == 1) {
            printf("  [x] Pinned host memory (cudaMallocHost)\n");
            printf("  [x] CUDA Streams x2 (double-buffered async H2D)\n");
        } else {
            printf("  [x] Mini-batch processing\n");
            printf("  [x] Atomic gradient accumulation\n");
        }
        printf("\n");

        // ---- Host weight buffers ----
        vector<float> h_w1(L1_OUT*L1_IN), h_b1(L1_OUT);
        vector<float> h_w2(L2_OUT*L2_IN), h_b2(L2_OUT);
        vector<float> h_w3(L3_OUT*L3_IN), h_b3(L3_OUT);
        init_layer(h_w1.data(), h_b1.data(), L1_IN, L1_OUT);
        init_layer(h_w2.data(), h_b2.data(), L2_IN, L2_OUT);
        init_layer(h_w3.data(), h_b3.data(), L3_IN, L3_OUT);

        // ---- Device: persistent weights ----
        float *d_w1,*d_b1,*d_w2,*d_b2,*d_w3,*d_b3;
        CUDA_CHECK(cudaMalloc(&d_w1, L1_OUT*L1_IN*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b1, L1_OUT*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w2, L2_OUT*L2_IN*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b2, L2_OUT*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w3, L3_OUT*L3_IN*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b3, L3_OUT*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_w1,h_w1.data(),L1_OUT*L1_IN*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b1,h_b1.data(),L1_OUT*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w2,h_w2.data(),L2_OUT*L2_IN*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2,h_b2.data(),L2_OUT*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w3,h_w3.data(),L3_OUT*L3_IN*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b3,h_b3.data(),L3_OUT*4,cudaMemcpyHostToDevice));

        // ---- Device: intermediate buffers ----
        const int B = max(1, batch_size);
        float *d_z1,*d_a1,*d_z2,*d_a2,*d_z3,*d_a3;
        float *d_grad,*d_g3,*d_g2,*d_g1;
        CUDA_CHECK(cudaMalloc(&d_z1,  B*L1_OUT*4));
        CUDA_CHECK(cudaMalloc(&d_a1,  B*L1_OUT*4));
        CUDA_CHECK(cudaMalloc(&d_z2,  B*L2_OUT*4));
        CUDA_CHECK(cudaMalloc(&d_a2,  B*L2_OUT*4));
        CUDA_CHECK(cudaMalloc(&d_z3,  B*L3_OUT*4));
        CUDA_CHECK(cudaMalloc(&d_a3,  B*L3_OUT*4));
        CUDA_CHECK(cudaMalloc(&d_grad,B*CLASSES*4));
        CUDA_CHECK(cudaMalloc(&d_g3,  B*L2_OUT*4));
        CUDA_CHECK(cudaMalloc(&d_g2,  B*L1_OUT*4));
        CUDA_CHECK(cudaMalloc(&d_g1,  B*L1_IN*4));

        // ---- Loss accumulator ----
        float *d_loss_acc, *h_loss;
        CUDA_CHECK(cudaMalloc(&d_loss_acc, 4));
        CUDA_CHECK(cudaMallocHost(&h_loss, 4));

        // ---- Streams & events ----
        cudaStream_t stream[2]; cudaEvent_t ev[2], ev_start, ev_stop;
        CUDA_CHECK(cudaStreamCreate(&stream[0]));
        CUDA_CHECK(cudaStreamCreate(&stream[1]));
        for (int i = 0; i < 2; ++i) CUDA_CHECK(cudaEventCreate(&ev[i]));
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));

        // ---- Mode-specific resources ----
        float *d_in[2]={}, *h_pin[2]={};
        float *d_images=nullptr; int *d_labels=nullptr;
        float *d_dw1=nullptr,*d_db1=nullptr;
        float *d_dw2=nullptr,*d_db2=nullptr;
        float *d_dw3=nullptr,*d_db3=nullptr;

        if (batch_size == 1) {
            for (int i = 0; i < 2; ++i) {
                CUDA_CHECK(cudaMallocHost(&h_pin[i], ENTRADAS*4));
                CUDA_CHECK(cudaMalloc(&d_in[i], ENTRADAS*4));
            }
        } else {
            CUDA_CHECK(cudaMalloc(&d_images, N*ENTRADAS*4));
            CUDA_CHECK(cudaMalloc(&d_labels, N*sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_images,treino.images.data(),N*ENTRADAS*4,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_labels,treino.labels.data(),N*sizeof(int),cudaMemcpyHostToDevice));
            auto alloc_grad = [](float** dw,float** db,int nw,int nb){
                CUDA_CHECK(cudaMalloc(dw,nw*4)); CUDA_CHECK(cudaMemset(*dw,0,nw*4));
                CUDA_CHECK(cudaMalloc(db,nb*4)); CUDA_CHECK(cudaMemset(*db,0,nb*4));
            };
            alloc_grad(&d_dw1,&d_db1,L1_OUT*L1_IN,L1_OUT);
            alloc_grad(&d_dw2,&d_db2,L2_OUT*L2_IN,L2_OUT);
            alloc_grad(&d_dw3,&d_db3,L3_OUT*L3_IN,L3_OUT);
        }

        // Block sizes (warp-aligned)
        const int BK1=64, BK2=32, BK3=32;
        // Shared memory sizes (bytes)
        const int SF1=L1_IN*4, SF2=L2_IN*4, SF3=L3_IN*4;
        const int SB1=(L1_OUT+L1_IN)*4, SB2=(L2_OUT+L2_IN)*4, SB3=(L3_OUT+L3_IN)*4;

        array<double,EPOCAS> perdas{};

        // =============================================================
        // TRAINING
        // =============================================================
        cout << fixed;
        CUDA_CHECK(cudaEventRecord(ev_start));

        if (batch_size == 1) {
            // ---- ONLINE SGD with 2-stream double-buffered pipelining ----
            for (int ep = 0; ep < EPOCAS; ++ep) {
                CUDA_CHECK(cudaMemset(d_loss_acc, 0, 4));
                for (int i = 0; i < N; ++i) {
                    int s = i & 1;
                    // Protect pinned buffer from prior async read
                    if (i >= 2) CUDA_CHECK(cudaEventSynchronize(ev[s]));
                    // CPU -> pinned host
                    memcpy(h_pin[s], &treino.images[i*ENTRADAS], ENTRADAS*4);
                    // Async H2D (overlaps with other stream's backward)
                    CUDA_CHECK(cudaMemcpyAsync(d_in[s],h_pin[s],ENTRADAS*4,
                                               cudaMemcpyHostToDevice,stream[s]));
                    // Weight dependency: wait for previous sample's backward
                    if (i > 0) CUDA_CHECK(cudaStreamWaitEvent(stream[s],ev[1-s],0));

                    // Forward (fused matmul + bias + activation)
                    kernel_forward<<<1,BK1,SF1,stream[s]>>>(d_w1,d_b1,d_in[s],d_a1,d_z1,L1_IN,L1_OUT,0);
                    kernel_forward<<<1,BK2,SF2,stream[s]>>>(d_w2,d_b2,d_a1,d_a2,d_z2,L2_IN,L2_OUT,0);
                    kernel_forward<<<1,BK3,SF3,stream[s]>>>(d_w3,d_b3,d_a2,d_a3,d_z3,L3_IN,L3_OUT,1);
                    // Loss (warp shuffle reduction)
                    kernel_loss<<<1,32,0,stream[s]>>>(d_a3,treino.labels[i],d_grad,d_loss_acc,CLASSES);
                    // Backward (shared-memory delta+input, then weight update)
                    kernel_backward<<<1,BK3,SB3,stream[s]>>>(d_w3,d_b3,d_a2,d_z3,d_grad,d_g3,L3_IN,L3_OUT,TAXA,1);
                    kernel_backward<<<1,BK2,SB2,stream[s]>>>(d_w2,d_b2,d_a1,d_z2,d_g3,d_g2,L2_IN,L2_OUT,TAXA,0);
                    kernel_backward<<<1,BK1,SB1,stream[s]>>>(d_w1,d_b1,d_in[s],d_z1,d_g2,d_g1,L1_IN,L1_OUT,TAXA,0);

                    CUDA_CHECK(cudaEventRecord(ev[s], stream[s]));
                }
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(h_loss,d_loss_acc,4,cudaMemcpyDeviceToHost));
                perdas[ep] = (double)*h_loss / N;
            }
        } else {
            // ---- MINI-BATCH SGD ----
            for (int ep = 0; ep < EPOCAS; ++ep) {
                CUDA_CHECK(cudaMemset(d_loss_acc, 0, 4));
                for (int st = 0; st < N; st += batch_size) {
                    int bs = min(batch_size, N - st);
                    float lr_bs = TAXA / bs;
                    // Forward
                    kernel_forward_batch<<<bs,BK1,SF1>>>(d_w1,d_b1,d_images+st*ENTRADAS,d_a1,d_z1,L1_IN,L1_OUT,0);
                    kernel_forward_batch<<<bs,BK2,SF2>>>(d_w2,d_b2,d_a1,d_a2,d_z2,L2_IN,L2_OUT,0);
                    kernel_forward_batch<<<bs,BK3,SF3>>>(d_w3,d_b3,d_a2,d_a3,d_z3,L3_IN,L3_OUT,1);
                    // Loss
                    kernel_loss_batch<<<bs,32>>>(d_a3,d_labels+st,d_grad,d_loss_acc,CLASSES);
                    // Backward (atomic gradient accumulation)
                    kernel_backward_batch<<<bs,BK3,SB3>>>(d_w3,d_a2,d_z3,d_grad,d_g3,d_dw3,d_db3,L3_IN,L3_OUT,1);
                    kernel_backward_batch<<<bs,BK2,SB2>>>(d_w2,d_a1,d_z2,d_g3,d_g2,d_dw2,d_db2,L2_IN,L2_OUT,0);
                    kernel_backward_batch<<<bs,BK1,SB1>>>(d_w1,d_images+st*ENTRADAS,d_z1,d_g2,d_g1,d_dw1,d_db1,L1_IN,L1_OUT,0);
                    // Apply averaged gradients + zero accumulators
                    kernel_apply_grads<<<(L1_OUT*L1_IN+255)/256,256>>>(d_w1,d_b1,d_dw1,d_db1,L1_OUT*L1_IN,L1_OUT,lr_bs);
                    kernel_apply_grads<<<(L2_OUT*L2_IN+255)/256,256>>>(d_w2,d_b2,d_dw2,d_db2,L2_OUT*L2_IN,L2_OUT,lr_bs);
                    kernel_apply_grads<<<(L3_OUT*L3_IN+255)/256,256>>>(d_w3,d_b3,d_dw3,d_db3,L3_OUT*L3_IN,L3_OUT,lr_bs);
                }
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(h_loss,d_loss_acc,4,cudaMemcpyDeviceToHost));
                perdas[ep] = (double)*h_loss / N;
            }
        }

        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float total_ms;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_stop));

        // ---- Print training metrics ----
        cout << setprecision(4) << "TREINO | " << N << " imagens\n";
        for (int e = 0; e < EPOCAS; e += 5)
            cout << "Epoca " << setw(2) << e+1 << "/" << EPOCAS
                 << " | perda media " << perdas[e] << "\n";
        cout << "Tempo de treino " << total_ms/1000.0f << " s\n";

        // ---- Timing Breakdown (single-sample benchmark, lr=0) ----
        {
            printf("\n--- Detalhamento de Tempo (1 amostra, sem update) ---\n");
            cudaEvent_t t[5];
            for (auto& e : t) CUDA_CHECK(cudaEventCreate(&e));
            float* fwd_in;
            if (batch_size == 1) {
                memcpy(h_pin[0], treino.images.data(), ENTRADAS*4);
                fwd_in = d_in[0];
            } else {
                fwd_in = d_images;
            }
            CUDA_CHECK(cudaEventRecord(t[0], stream[0]));
            if (batch_size == 1)
                CUDA_CHECK(cudaMemcpyAsync(d_in[0],h_pin[0],ENTRADAS*4,cudaMemcpyHostToDevice,stream[0]));
            CUDA_CHECK(cudaEventRecord(t[1], stream[0]));
            kernel_forward<<<1,BK1,SF1,stream[0]>>>(d_w1,d_b1,fwd_in,d_a1,d_z1,L1_IN,L1_OUT,0);
            kernel_forward<<<1,BK2,SF2,stream[0]>>>(d_w2,d_b2,d_a1,d_a2,d_z2,L2_IN,L2_OUT,0);
            kernel_forward<<<1,BK3,SF3,stream[0]>>>(d_w3,d_b3,d_a2,d_a3,d_z3,L3_IN,L3_OUT,1);
            CUDA_CHECK(cudaEventRecord(t[2], stream[0]));
            kernel_loss<<<1,32,0,stream[0]>>>(d_a3,treino.labels[0],d_grad,d_loss_acc,CLASSES);
            CUDA_CHECK(cudaEventRecord(t[3], stream[0]));
            kernel_backward<<<1,BK3,SB3,stream[0]>>>(d_w3,d_b3,d_a2,d_z3,d_grad,d_g3,L3_IN,L3_OUT,0.0f,1);
            kernel_backward<<<1,BK2,SB2,stream[0]>>>(d_w2,d_b2,d_a1,d_z2,d_g3,d_g2,L2_IN,L2_OUT,0.0f,0);
            kernel_backward<<<1,BK1,SB1,stream[0]>>>(d_w1,d_b1,fwd_in,d_z1,d_g2,d_g1,L1_IN,L1_OUT,0.0f,0);
            CUDA_CHECK(cudaEventRecord(t[4], stream[0]));
            CUDA_CHECK(cudaEventSynchronize(t[4]));
            float ms[4];
            for (int i = 0; i < 4; ++i) CUDA_CHECK(cudaEventElapsedTime(&ms[i],t[i],t[i+1]));
            printf("H2D Transfer: %.4f ms\n", ms[0]);
            printf("Forward:      %.4f ms\n", ms[1]);
            printf("Loss:         %.4f ms\n", ms[2]);
            printf("Backward:     %.4f ms\n", ms[3]);
            printf("Total:        %.4f ms\n", ms[0]+ms[1]+ms[2]+ms[3]);
            for (auto& e : t) CUDA_CHECK(cudaEventDestroy(e));
        }

        // ---- Copy weights back for evaluation ----
        CUDA_CHECK(cudaMemcpy(h_w1.data(),d_w1,L1_OUT*L1_IN*4,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b1.data(),d_b1,L1_OUT*4,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_w2.data(),d_w2,L2_OUT*L2_IN*4,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b2.data(),d_b2,L2_OUT*4,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_w3.data(),d_w3,L3_OUT*L3_IN*4,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b3.data(),d_b3,L3_OUT*4,cudaMemcpyDeviceToHost));

        avaliar(h_w1.data(),h_b1.data(), h_w2.data(),h_b2.data(),
                h_w3.data(),h_b3.data(), teste);

        // ---- Cleanup ----
        CUDA_CHECK(cudaFree(d_w1)); CUDA_CHECK(cudaFree(d_b1));
        CUDA_CHECK(cudaFree(d_w2)); CUDA_CHECK(cudaFree(d_b2));
        CUDA_CHECK(cudaFree(d_w3)); CUDA_CHECK(cudaFree(d_b3));
        CUDA_CHECK(cudaFree(d_z1)); CUDA_CHECK(cudaFree(d_a1));
        CUDA_CHECK(cudaFree(d_z2)); CUDA_CHECK(cudaFree(d_a2));
        CUDA_CHECK(cudaFree(d_z3)); CUDA_CHECK(cudaFree(d_a3));
        CUDA_CHECK(cudaFree(d_grad));CUDA_CHECK(cudaFree(d_g3));
        CUDA_CHECK(cudaFree(d_g2)); CUDA_CHECK(cudaFree(d_g1));
        CUDA_CHECK(cudaFree(d_loss_acc)); CUDA_CHECK(cudaFreeHost(h_loss));
        for (int i = 0; i < 2; ++i) {
            if (d_in[i])  CUDA_CHECK(cudaFree(d_in[i]));
            if (h_pin[i]) CUDA_CHECK(cudaFreeHost(h_pin[i]));
        }
        if (d_images) CUDA_CHECK(cudaFree(d_images));
        if (d_labels) CUDA_CHECK(cudaFree(d_labels));
        if (d_dw1){ CUDA_CHECK(cudaFree(d_dw1)); CUDA_CHECK(cudaFree(d_db1)); }
        if (d_dw2){ CUDA_CHECK(cudaFree(d_dw2)); CUDA_CHECK(cudaFree(d_db2)); }
        if (d_dw3){ CUDA_CHECK(cudaFree(d_dw3)); CUDA_CHECK(cudaFree(d_db3)); }
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaStreamDestroy(stream[i]));
            CUDA_CHECK(cudaEventDestroy(ev[i]));
        }
        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));

    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
    return 0;
}
