// nvcc -arch=sm_86 -O2 -std=c++17 -o barenn_cuda barenn.cu
//
// Migração CUDA do barenn.cpp — porting funcional inicial.
// Rede: 64 → 32 → 16 → 10 (ReLU nas camadas ocultas, saída linear)
// Treino: SGD online (amostra a amostra), 51 épocas, lr = 0.02, perda MSE
// Dataset: optdigits (3823 treino, 1797 teste)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

// ---------------------------------------------------------------------------
// Macro de verificação de erros CUDA
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                                 \
    cudaError_t e = (call);                                                   \
    if (e != cudaSuccess) {                                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                cudaGetErrorString(e));                                        \
        exit(1);                                                              \
    }                                                                         \
} while (0)

// ---------------------------------------------------------------------------
// Constantes da rede
// ---------------------------------------------------------------------------
constexpr int ENTRADAS = 64;
constexpr int CLASSES  = 10;

constexpr int L1_IN = 64,  L1_OUT = 32;
constexpr int L2_IN = 32,  L2_OUT = 16;
constexpr int L3_IN = 16,  L3_OUT = 10;

// ---------------------------------------------------------------------------
// Estrutura de dados (idêntica ao original)
// ---------------------------------------------------------------------------
struct Amostra {
    vector<float> imagem;
    int rotulo;
};

vector<Amostra> ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo)
        throw runtime_error("Não foi possível abrir " + caminho);

    vector<Amostra> dados;
    string linha;
    size_t numero_linha = 0;
    while (getline(arquivo, linha)) {
        ++numero_linha;
        istringstream campos(linha);
        Amostra amostra{};
        for (int i = 0; i < ENTRADAS; ++i) {
            float valor;
            char separador;
            if (!(campos >> valor >> separador) || separador != ',' ||
                !isfinite(valor) || valor < 0 || valor > 16)
                break;
            amostra.imagem.push_back(valor / 16.0f);
        }
        if (amostra.imagem.size() != ENTRADAS ||
            !(campos >> amostra.rotulo) ||
            amostra.rotulo < 0 || amostra.rotulo >= CLASSES ||
            !(campos >> ws).eof()) {
            throw runtime_error(caminho + ":" + to_string(numero_linha) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
        dados.push_back(move(amostra));
    }
    if (arquivo.bad() || dados.empty())
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    return dados;
}

// ---------------------------------------------------------------------------
// Inicialização de pesos — reproduz o comportamento exato do original:
// cada camada usa mt19937(1337) + uniform_real_distribution(-0.05, 0.05).
// Os pesos são armazenados em row-major: weights[i * in + j] = peso do
// neurônio de saída i recebendo da entrada j.
// ---------------------------------------------------------------------------
void init_weights(float* w, float* b, int in_features, int out_features) {
    mt19937 gen(1337);
    uniform_real_distribution<float> dist(-0.05f, 0.05f);
    for (int i = 0; i < out_features; ++i)
        for (int j = 0; j < in_features; ++j)
            w[i * in_features + j] = dist(gen);
    for (int i = 0; i < out_features; ++i)
        b[i] = 0.0f;
}

// ===========================================================================
//                           KERNELS CUDA
// ===========================================================================

// Forward de uma camada densa: cada thread calcula um neurônio de saída.
//   z[i]   = bias[i] + Σⱼ input[j] * weights[i * in + j]
//   out[i] = is_final ? z[i] : max(0, z[i])
__global__ void forward_kernel(const float* __restrict__ weights,
                               const float* __restrict__ bias,
                               const float* __restrict__ input,
                               float* __restrict__ z_out,
                               float* __restrict__ a_out,
                               int in_features, int out_features,
                               int is_final) {
    int i = threadIdx.x;
    if (i >= out_features) return;

    float sum = bias[i];
    for (int j = 0; j < in_features; ++j)
        sum += input[j] * weights[i * in_features + j];

    z_out[i] = sum;
    a_out[i] = is_final ? sum : fmaxf(0.0f, sum);
}

// Perda MSE: gradient[i] = score[i] − target[i], loss = ½ Σ gradient[i]²
__global__ void loss_kernel(const float* __restrict__ scores,
                            int label,
                            float* __restrict__ gradient,
                            float* __restrict__ loss_out,
                            int num_classes) {
    int i = threadIdx.x;
    if (i >= num_classes) return;

    float target = (i == label) ? 1.0f : 0.0f;
    float err = scores[i] - target;
    gradient[i] = err;

    extern __shared__ float s_loss[];
    s_loss[i] = 0.5f * err * err;
    __syncthreads();

    // Thread 0 reduz as parcelas (são apenas 10).
    if (i == 0) {
        float total = 0.0f;
        for (int k = 0; k < num_classes; ++k)
            total += s_loss[k];
        *loss_out = total;
    }
}

// Delta: aplica a derivada da ativação sobre o gradiente que chega.
//   delta[i] = is_final ? grad_out[i] : grad_out[i] * (z[i] > 0 ? 1 : 0)
__global__ void compute_delta_kernel(const float* __restrict__ grad_out,
                                     const float* __restrict__ z,
                                     float* __restrict__ delta,
                                     int n, int is_final) {
    int i = threadIdx.x;
    if (i >= n) return;
    delta[i] = is_final ? grad_out[i]
                        : grad_out[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
}

// Gradiente de entrada: grad_in[j] = Σᵢ delta[i] * weights[i * in + j]
// Deve ser lançado ANTES de update_weights_kernel (usa os pesos antigos).
__global__ void compute_grad_input_kernel(const float* __restrict__ delta,
                                          const float* __restrict__ weights,
                                          float* __restrict__ grad_in,
                                          int in_features, int out_features) {
    int j = threadIdx.x;
    if (j >= in_features) return;

    float sum = 0.0f;
    for (int i = 0; i < out_features; ++i)
        sum += delta[i] * weights[i * in_features + j];
    grad_in[j] = sum;
}

// SGD por amostra: cada thread i atualiza toda a linha i de pesos e o viés i.
//   weights[i*in + j] -= lr * delta[i] * input[j]
//   bias[i]           -= lr * delta[i]
__global__ void update_weights_kernel(float* __restrict__ weights,
                                      float* __restrict__ bias,
                                      const float* __restrict__ delta,
                                      const float* __restrict__ input,
                                      int in_features, int out_features,
                                      float lr) {
    int i = threadIdx.x;
    if (i >= out_features) return;

    float d = delta[i];
    for (int j = 0; j < in_features; ++j)
        weights[i * in_features + j] -= lr * d * input[j];
    bias[i] -= lr * d;
}

// ===========================================================================
//                     AVALIAÇÃO NO HOST (mesmo formato)
// ===========================================================================

static void cpu_forward(const float* w, const float* b, const float* in,
                        float* out, int in_f, int out_f, bool is_final) {
    for (int i = 0; i < out_f; ++i) {
        float sum = b[i];
        for (int j = 0; j < in_f; ++j)
            sum += in[j] * w[i * in_f + j];
        out[i] = is_final ? sum : max(0.0f, sum);
    }
}

static float cpu_perda(const float* escores, int rotulo) {
    float total = 0;
    for (int i = 0; i < CLASSES; ++i) {
        float e = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * e * e;
    }
    return total;
}

void avaliar(const float* w1, const float* b1,
             const float* w2, const float* b2,
             const float* w3, const float* b3,
             const vector<Amostra>& teste) {
    struct Exemplo {
        size_t linha; int real, previsto, segundo;
        float maior_escore, segundo_escore;
    };

    array<array<size_t, CLASSES>, CLASSES> confusoes{};
    vector<Exemplo> exemplos;
    size_t acertos = 0, exemplos_acerto = 0, exemplos_erro = 0;
    double perda_total = 0;
    float a1[L1_OUT], a2[L2_OUT], escores[L3_OUT];

    for (size_t i = 0; i < teste.size(); ++i) {
        const auto& am = teste[i];
        cpu_forward(w1, b1, am.imagem.data(), a1, L1_IN, L1_OUT, false);
        cpu_forward(w2, b2, a1,               a2, L2_IN, L2_OUT, false);
        cpu_forward(w3, b3, a2,         escores,  L3_IN, L3_OUT, true);

        int previsto = static_cast<int>(
            max_element(escores, escores + CLASSES) - escores);
        bool acertou = previsto == am.rotulo;
        acertos += acertou;
        ++confusoes[am.rotulo][previsto];
        perda_total += cpu_perda(escores, am.rotulo);

        size_t& et = acertou ? exemplos_acerto : exemplos_erro;
        if (et < 3) {
            int seg = previsto == 0 ? 1 : 0;
            for (int c = 0; c < CLASSES; ++c)
                if (c != previsto && escores[c] > escores[seg]) seg = c;
            exemplos.push_back({i + 1, am.rotulo, previsto, seg,
                                escores[previsto], escores[seg]});
            ++et;
        }
    }

    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / teste.size()
         << "% | " << acertos << " acertos em " << teste.size() << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / teste.size()
         << "\n";

    cout << "\nALGUMAS INFERENCIAS\n"
         << "Esperado  Previsto\n";
    for (const auto& ex : exemplos)
        cout << right << setw(8) << ex.real
             << "  " << setw(8) << ex.previsto << '\n';

    struct Confusao { int real, previsto; size_t ocorrencias; };
    vector<Confusao> erros;
    for (int r = 0; r < CLASSES; ++r)
        for (int p = 0; p < CLASSES; ++p)
            if (r != p && confusoes[r][p] > 0)
                erros.push_back({r, p, confusoes[r][p]});
    stable_sort(erros.begin(), erros.end(),
                [](const auto& a, const auto& b) {
                    return a.ocorrencias > b.ocorrencias;
                });

    cout << "\nCONFUSOES MAIS FREQUENTES\n";
    if (erros.empty()) {
        cout << "Nenhuma neste conjunto.\n";
    } else {
        cout << "Esperado  Previsto  Ocorrencias\n";
        for (size_t i = 0; i < min<size_t>(3, erros.size()); ++i)
            cout << right << setw(8) << erros[i].real
                 << "  " << setw(8) << erros[i].previsto
                 << "  " << setw(11) << erros[i].ocorrencias << '\n';
    }
}

// ===========================================================================
//                                MAIN
// ===========================================================================
int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste  = argc == 3 ? argv[2] : "optdigits.tes";

    try {
        // ----- Leitura dos dados -----
        const auto treino = ler_dados(arquivo_treino);
        const auto teste  = ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste))
            throw runtime_error("Use arquivos distintos para treino e teste.");

        // ----- Informações da GPU -----
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        cout << "GPU: " << prop.name
             << " | " << prop.multiProcessorCount << " SMs"
             << " | Compute " << prop.major << "." << prop.minor << "\n\n";

        constexpr int epocas = 51;
        constexpr float taxa = 0.02f;

        // ----- Inicialização dos pesos no host (mesma semente por camada) -----
        vector<float> h_w1(L1_OUT * L1_IN), h_b1(L1_OUT, 0);
        vector<float> h_w2(L2_OUT * L2_IN), h_b2(L2_OUT, 0);
        vector<float> h_w3(L3_OUT * L3_IN), h_b3(L3_OUT, 0);

        init_weights(h_w1.data(), h_b1.data(), L1_IN, L1_OUT);
        init_weights(h_w2.data(), h_b2.data(), L2_IN, L2_OUT);
        init_weights(h_w3.data(), h_b3.data(), L3_IN, L3_OUT);

        // ----- Alocação de memória no Device -----

        // Pesos e vieses
        float *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3;
        CUDA_CHECK(cudaMalloc(&d_w1, L1_OUT * L1_IN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b1, L1_OUT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w2, L2_OUT * L2_IN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b2, L2_OUT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w3, L3_OUT * L3_IN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b3, L3_OUT * sizeof(float)));

        // Ativações e pré-ativações (z)
        float *d_input, *d_z1, *d_a1, *d_z2, *d_a2, *d_z3, *d_a3;
        CUDA_CHECK(cudaMalloc(&d_input, L1_IN  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_z1,    L1_OUT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_a1,    L1_OUT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_z2,    L2_OUT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_a2,    L2_OUT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_z3,    L3_OUT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_a3,    L3_OUT * sizeof(float)));

        // Gradiente da perda e valor escalar
        float *d_loss_grad, *d_loss_val;
        CUDA_CHECK(cudaMalloc(&d_loss_grad, CLASSES * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_loss_val,  sizeof(float)));

        // Buffers do backward (reutilizados entre camadas)
        //   d_delta  : max(L1_OUT, L2_OUT, L3_OUT) = 32
        //   d_grad_a : usado para propagar gradiente de L3→L2 (16 floats)
        //   d_grad_b : usado para propagar gradiente de L2→L1 (32 floats)
        float *d_delta, *d_grad_a, *d_grad_b;
        CUDA_CHECK(cudaMalloc(&d_delta,  L1_OUT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_a, L1_IN  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_b, L1_IN  * sizeof(float)));

        // ----- Transferência Host → Device dos pesos iniciais -----
        CUDA_CHECK(cudaMemcpy(d_w1, h_w1.data(), L1_OUT * L1_IN * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b1, h_b1.data(), L1_OUT * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w2, h_w2.data(), L2_OUT * L2_IN * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2, h_b2.data(), L2_OUT * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w3, h_w3.data(), L3_OUT * L3_IN * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b3, h_b3.data(), L3_OUT * sizeof(float),
                              cudaMemcpyHostToDevice));

        // ----- Treino -----
        array<double, epocas> perdas{};
        float h_loss;

        // Cronometragem via CUDA Events
        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
        CUDA_CHECK(cudaEventRecord(ev_start));

        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;

            for (const auto& amostra : treino) {
                // --- Entrada Host → Device ---
                CUDA_CHECK(cudaMemcpy(d_input, amostra.imagem.data(),
                                      L1_IN * sizeof(float),
                                      cudaMemcpyHostToDevice));

                // --- Forward ---
                forward_kernel<<<1, L1_OUT>>>(d_w1, d_b1, d_input,
                                              d_z1, d_a1,
                                              L1_IN, L1_OUT, 0);
                forward_kernel<<<1, L2_OUT>>>(d_w2, d_b2, d_a1,
                                              d_z2, d_a2,
                                              L2_IN, L2_OUT, 0);
                forward_kernel<<<1, L3_OUT>>>(d_w3, d_b3, d_a2,
                                              d_z3, d_a3,
                                              L3_IN, L3_OUT, 1);

                // --- Perda ---
                loss_kernel<<<1, CLASSES, CLASSES * sizeof(float)>>>(
                    d_a3, amostra.rotulo, d_loss_grad, d_loss_val, CLASSES);

                // Copia o escalar de perda para o host (sync implícito)
                CUDA_CHECK(cudaMemcpy(&h_loss, d_loss_val, sizeof(float),
                                      cudaMemcpyDeviceToHost));
                perda_total += h_loss;

                // --- Backward camada 3 (final) ---
                // delta = grad_out (is_final)
                compute_delta_kernel<<<1, L3_OUT>>>(
                    d_loss_grad, d_z3, d_delta, L3_OUT, 1);
                // grad_input → d_grad_a (16 floats, lê w3 antes do update)
                compute_grad_input_kernel<<<1, L3_IN>>>(
                    d_delta, d_w3, d_grad_a, L3_IN, L3_OUT);
                // atualiza w3 e b3
                update_weights_kernel<<<1, L3_OUT>>>(
                    d_w3, d_b3, d_delta, d_a2, L3_IN, L3_OUT, taxa);

                // --- Backward camada 2 ---
                compute_delta_kernel<<<1, L2_OUT>>>(
                    d_grad_a, d_z2, d_delta, L2_OUT, 0);
                compute_grad_input_kernel<<<1, L2_IN>>>(
                    d_delta, d_w2, d_grad_b, L2_IN, L2_OUT);
                update_weights_kernel<<<1, L2_OUT>>>(
                    d_w2, d_b2, d_delta, d_a1, L2_IN, L2_OUT, taxa);

                // --- Backward camada 1 ---
                compute_delta_kernel<<<1, L1_OUT>>>(
                    d_grad_b, d_z1, d_delta, L1_OUT, 0);
                // grad_input da camada 1 não é necessário (não há camada anterior)
                update_weights_kernel<<<1, L1_OUT>>>(
                    d_w1, d_b1, d_delta, d_input, L1_IN, L1_OUT, taxa);

                CUDA_CHECK(cudaDeviceSynchronize());
            }
            perdas[epoca] = perda_total / treino.size();
        }

        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float gpu_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));

        // ----- Transferência Device → Host dos pesos treinados -----
        CUDA_CHECK(cudaMemcpy(h_w1.data(), d_w1, L1_OUT * L1_IN * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b1.data(), d_b1, L1_OUT * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_w2.data(), d_w2, L2_OUT * L2_IN * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b2.data(), d_b2, L2_OUT * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_w3.data(), d_w3, L3_OUT * L3_IN * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b3.data(), d_b3, L3_OUT * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // ----- Impressão dos resultados de treino -----
        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens | GPU\n";
        for (int e = 0; e < epocas; e += 5)
            cout << "Epoca " << setw(2) << e + 1 << "/" << epocas
                 << " | perda media " << perdas[e] << "\n";
        cout << "Tempo de treino " << gpu_ms / 1000.0f << " s (GPU)\n";

        // ----- Avaliação no host (mesmo formato da versão CPU) -----
        avaliar(h_w1.data(), h_b1.data(),
                h_w2.data(), h_b2.data(),
                h_w3.data(), h_b3.data(), teste);

        // ----- Limpeza -----
        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));

        CUDA_CHECK(cudaFree(d_w1));  CUDA_CHECK(cudaFree(d_b1));
        CUDA_CHECK(cudaFree(d_w2));  CUDA_CHECK(cudaFree(d_b2));
        CUDA_CHECK(cudaFree(d_w3));  CUDA_CHECK(cudaFree(d_b3));
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_z1));  CUDA_CHECK(cudaFree(d_a1));
        CUDA_CHECK(cudaFree(d_z2));  CUDA_CHECK(cudaFree(d_a2));
        CUDA_CHECK(cudaFree(d_z3));  CUDA_CHECK(cudaFree(d_a3));
        CUDA_CHECK(cudaFree(d_loss_grad));
        CUDA_CHECK(cudaFree(d_loss_val));
        CUDA_CHECK(cudaFree(d_delta));
        CUDA_CHECK(cudaFree(d_grad_a));
        CUDA_CHECK(cudaFree(d_grad_b));

    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
    return 0;
}
