// Compile: g++ -O3 -fopenmp -march=native -std=c++17 -o barenn_cpu barenn_cpu.cpp
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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
#include <omp.h>

using namespace std;

namespace Barenn {

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;

class DenseLayer {
    int in_features;
    int out_features;
    bool final_layer;
    // Weights flattened into contiguous row-major storage: weights[i * in_features + j]
    vector<float> weights;
    vector<float> bias;
    // Dados do último forward usados para calcular os gradientes.
    vector<float> last_input;
    vector<float> last_z;
    // Pre-allocated output buffer for Forward
    vector<float> out_buf;

public:
    DenseLayer(int input_size, int output_size, bool is_final = false)
        : in_features(input_size), out_features(output_size), final_layer(is_final),
          weights(output_size * input_size), bias(output_size, 0),
          last_z(output_size), out_buf(output_size) {
        // A semente fixa torna a inicialização de cada camada reproduzível.
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                weights[i * in_features + j] = distribution(gen);
            }
        }
    }

    // Pass by const reference to avoid copying the input vector.
    const vector<float>& Forward(const vector<float>& input) {
        last_input = input;
        // Cada neurônio soma o viés aos produtos entre entradas e pesos.
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < out_features; ++i) {
            float sum = bias[i];
            const float* w_row = &weights[i * in_features];
            for (int j = 0; j < in_features; ++j) {
                sum += input[j] * w_row[j];
            }
            last_z[i] = sum;
            // As camadas ocultas usam ReLU. A última devolve escores lineares.
            out_buf[i] = final_layer ? sum : max(0.0f, sum);
        }
        return out_buf;
    }

    // Backward writes grad_in into a caller-provided buffer to avoid allocation.
    void Backward(const vector<float>& grad_out, float learning_rate,
                  vector<float>& grad_in) {
        // Delta é o gradiente da perda em relação à soma antes da ativação.
        // Na ReLU, o gradiente passa apenas quando essa soma é positiva.
        // Reuse last_z temporarily as delta storage (we don't need last_z after this).
        // Actually, keep a local delta to avoid aliasing issues with last_z.
        // We'll compute delta inline below.

        // O gradiente da entrada usa os pesos anteriores à atualização.
        // Loop order: iterate j (output neuron) in outer loop for sequential weight access
        // when reading weights[j * in_features + i]. But we need to accumulate into grad_in[i].
        // Better approach: zero grad_in, then for each output neuron j, scatter its contribution.
        fill(grad_in.begin(), grad_in.end(), 0.0f);

        for (int j = 0; j < out_features; ++j) {
            float delta_j = final_layer ? grad_out[j]
                                        : grad_out[j] * (last_z[j] > 0.0f ? 1.0f : 0.0f);
            const float* w_row = &weights[j * in_features];
            for (int i = 0; i < in_features; ++i) {
                grad_in[i] += delta_j * w_row[i];
            }
        }

        // SGD por amostra. O gradiente do peso é delta vezes a entrada.
        // O gradiente do viés é o próprio delta.
        for (int i = 0; i < out_features; ++i) {
            float delta_i = final_layer ? grad_out[i]
                                        : grad_out[i] * (last_z[i] > 0.0f ? 1.0f : 0.0f);
            float* w_row = &weights[i * in_features];
            float lr_delta = learning_rate * delta_i;
            for (int j = 0; j < in_features; ++j) {
                w_row[j] -= lr_delta * last_input[j];
            }
            bias[i] -= lr_delta;
        }
    }
};

struct Amostra {
    vector<float> imagem;
    int rotulo;
};

vector<Amostra> ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }

    vector<Amostra> dados;
    dados.reserve(4096); // Reserve space for expected number of samples
    string linha;
    size_t numero_linha = 0;
    while (getline(arquivo, linha)) {
        ++numero_linha;
        istringstream campos(linha);
        Amostra amostra{};
        amostra.imagem.reserve(entradas);
        // Cada atributo deve ser seguido por vírgula, inclusive o último.
        for (int i = 0; i < entradas; ++i) {
            float valor;
            char separador;
            if (!(campos >> valor >> separador) || separador != ',' ||
                !isfinite(valor) || valor < 0 || valor > 16) {
                break;
            }
            // Converte os valores de [0, 16] para [0, 1].
            amostra.imagem.push_back(valor / 16.0f);
        }
        // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
        if (amostra.imagem.size() != entradas ||
            !(campos >> amostra.rotulo) ||
            amostra.rotulo < 0 || amostra.rotulo >= classes ||
            !(campos >> ws).eof()) {
            throw runtime_error(caminho + ":" + to_string(numero_linha) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
        dados.push_back(move(amostra));
    }
    if (arquivo.bad() || dados.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }
    return dados;
}

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
float perda(const vector<float>& escores, int rotulo,
            vector<float>& gradiente) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        gradiente[i] = erro;
    }
    return total;
}

// Overload without gradient for evaluation
float perda(const vector<float>& escores, int rotulo) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
    }
    return total;
}

void avaliar(DenseLayer& l1, DenseLayer& l2, DenseLayer& l3,
             const vector<Amostra>& teste) {
    struct Exemplo {
        size_t linha;
        int real;
        int previsto;
        int segundo;
        float maior_escore;
        float segundo_escore;
    };

    // Linhas representam a classe esperada e colunas, a classe prevista.
    array<array<size_t, classes>, classes> confusoes{};
    vector<Exemplo> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;

    // Só fazemos forward aqui. Os pesos ficam fixos.
    for (size_t i = 0; i < teste.size(); ++i) {
        const auto& amostra = teste[i];
        const auto& escores = l3.Forward(l2.Forward(l1.Forward(amostra.imagem)));
        // O índice do maior escore indica a classe prevista.
        int previsto = static_cast<int>(
            max_element(escores.begin(), escores.end()) - escores.begin());
        bool acertou = previsto == amostra.rotulo;
        acertos += acertou;
        ++confusoes[amostra.rotulo][previsto];
        perda_total += perda(escores, amostra.rotulo);

        // Guarda até três acertos e três erros para ilustrar a saída.
        // A acurácia usa todas as imagens, inclusive as não exibidas.
        size_t& exemplos_tipo = acertou ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            int segundo = previsto == 0 ? 1 : 0;
            for (int classe = 0; classe < classes; ++classe) {
                if (classe != previsto && escores[classe] > escores[segundo]) {
                    segundo = classe;
                }
            }
            exemplos.push_back({i + 1, amostra.rotulo, previsto, segundo,
                                escores[previsto], escores[segundo]});
            ++exemplos_tipo;
        }
    }

    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / teste.size()
         << "% | " << acertos << " acertos em " << teste.size() << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / teste.size() << "\n";

    cout << "\nALGUMAS INFERENCIAS\n";
    cout << "Esperado  Previsto\n";

    for (const auto& exemplo : exemplos) {
        cout << right << setw(8) << exemplo.real
             << "  " << setw(8) << exemplo.previsto << '\n';
    }

    struct Confusao {
        int real;
        int previsto;
        size_t ocorrencias;
    };
    vector<Confusao> erros;
    for (int real = 0; real < classes; ++real) {
        for (int previsto = 0; previsto < classes; ++previsto) {
            if (real != previsto && confusoes[real][previsto] > 0) {
                erros.push_back({real, previsto, confusoes[real][previsto]});
            }
        }
    }
    // Ordena os pares incorretos do mais frequente para o menos frequente.
    stable_sort(erros.begin(), erros.end(), [](const auto& a, const auto& b) {
        return a.ocorrencias > b.ocorrencias;
    });
    cout << "\nCONFUSOES MAIS FREQUENTES\n";

    if (erros.empty()) {
        cout << "Nenhuma neste conjunto.\n";
    } else {
        cout << "Esperado  Previsto  Ocorrencias\n";

        for (size_t i = 0; i < min<size_t>(3, erros.size()); ++i) {
            const auto& erro = erros[i];
            cout << right << setw(8) << erro.real
                << "  " << setw(8) << erro.previsto
                << "  " << setw(11) << erro.ocorrencias << '\n';
        }
    }
}

} // namespace Barenn

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";

    try {
        cout << "Threads: " << omp_get_max_threads() << endl;

        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        constexpr int epocas = 51;
        constexpr float taxa = 0.02f;
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        Barenn::DenseLayer l1(Barenn::entradas, 32);
        Barenn::DenseLayer l2(32, 16);
        Barenn::DenseLayer l3(16, Barenn::classes, true);

        // Pre-allocate gradient vectors to avoid per-sample allocation.
        vector<float> gradiente(Barenn::classes);
        vector<float> grad3(32);  // grad from l3 backward -> size = l3.in_features = 16
        vector<float> grad2(64);  // grad from l2 backward -> size = l2.in_features = 32
        vector<float> grad1(64);  // grad from l1 backward -> size = l1.in_features = 64
        // Correct sizes:
        grad3.resize(16);
        grad2.resize(32);
        grad1.resize(Barenn::entradas);

        array<double, epocas> perdas{};

        // A medição inclui forward, perda e backward de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;
            // Uma época percorre todo o conjunto na ordem do arquivo.
            for (const auto& amostra : treino) {
                const auto& escores = l3.Forward(l2.Forward(l1.Forward(amostra.imagem)));
                perda_total += Barenn::perda(escores, amostra.rotulo, gradiente);

                // Propaga o gradiente da última camada até a primeira.
                // Cada Backward também atualiza os parâmetros da sua camada.
                l3.Backward(gradiente, taxa, grad3);
                l2.Backward(grad3, taxa, grad2);
                l1.Backward(grad2, taxa, grad1);
            }
            perdas[epoca] = perda_total / treino.size();
        }
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";
        Barenn::avaliar(l1, l2, l3, teste);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
