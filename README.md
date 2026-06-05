# Trabalho Final — Inteligência Artificial
## Do NumPy ao Estado da Arte em Visão Computacional
### Classificação de Tecidos Histopatológicos com PathMNIST

Centro Universitário Católica de Quixadá — UniCatólica
Curso de Sistemas de Informação · Disciplina de Inteligência Artificial

---

## Equipe
- Rawenne da Silva Leite
- Gustavo Costa Silva
- Pedro Paulo Alves dos Santos Nunes
- Eduardo Nobre Nogueira de Oliveira

## Sobre o projeto
Classificação multiclasse (9 classes) de tecidos histopatológicos de câncer
colorretal usando o dataset PathMNIST 224×224 (coleção MedMNIST v2). O trabalho
percorre uma jornada progressiva: da implementação manual de uma rede neural em
NumPy até arquiteturas estado da arte (CNNs profundas e Vision Transformers),
com foco em rigor experimental e interpretabilidade (Explainable AI).

## Resultados principais
| Etapa | Modelo | Resultado |
|-------|--------|-----------|
| 1 | MLP NumPy (do zero) | ~50% val · gradient checking < 1e-8 |
| 2 | MLP PyTorch | Equivalência com NumPy (dif. 1.97 p.p.) |
| 3 | EfficientNet-B0 (fine-tuning) | 99.3% validação (melhor modelo) |
| 5 | EfficientNet-B0 — **TESTE** | **93.6% (generalização real)** |

## Ambiente de execução
- Plataforma: Kaggle Notebooks
- GPU: NVIDIA Tesla T4 (Etapas 2-5) · CPU para Etapa 1 (NumPy puro)
- RAM: ~33 GB
- Python 3.12
- Seed fixada: 42 (numpy, torch, random)

## Estrutura do repositório

trabalho-final-ia/

├── README.md

├── requirements.txt

├── notebooks/

│   ├── 01_numpy_mlp.ipynb          # MLP do zero + gradient checking

│   ├── 02_pytorch_validation.ipynb # Equivalência PyTorch + pipeline 224x224

│   ├── 03_cnns_and_vit.ipynb       # CNN própria, 3 CNNs pré-treinadas, ViT

│   ├── 04_xai.ipynb                # Feature Maps, Grad-CAM, Integrated Gradients

│   └── 05_final_model.ipynb        # Avaliação final no teste + matriz de confusão

├── experiments/

│   ├── results.csv                 # Resultados de todos os modelos

│   └── results_grid.csv            # Busca de hiperparâmetros

├── figures/                        # Visualizações do relatório

└── report/

└── relatorio.pdf

## Como rodar
1. Abrir os notebooks no Kaggle (com GPU e Internet habilitados).
2. Executar as células na ordem (Run All).
3. O dataset PathMNIST 224×224 é baixado automaticamente via `medmnist`.
4. Para a Etapa 1 (NumPy), usa-se a versão 28×28; das Etapas 2-5, a versão 224×224.

## Notas metodológicas
- **Subconjunto de treino:** por restrição de memória e tempo de GPU, os modelos
  das Etapas 3-5 foram treinados com um subconjunto de 20.000 imagens. Os conjuntos
  de validação (10.004) e teste (7.180) foram mantidos completos e oficiais.
- **Isolamento do teste:** o conjunto de teste foi utilizado uma única vez, sobre
  o modelo final, para reportar a generalização. Todas as decisões foram baseadas
  na validação.
- **Carregamento de dados:** o PathMNIST 224×224 (12.6 GB) é lido via memória
  mapeada (`np.load(..., mmap_mode='r')`) para evitar estouro de RAM.

## Checkpoints dos modelos
Os pesos treinados (.pth) estão disponíveis em: https://drive.google.com/drive/folders/1RFSViTbs6GTBEjaueRp9cNQKYQNPHVg6?usp=drive_link

## Declaração de uso de IA generativa
Conforme a política da disciplina, declaramos o uso do assistente Claude
(Anthropic) com as seguintes finalidades: explicação de conceitos (backpropagation,
momentum, gradient checking, transfer learning, XAI), apoio na estruturação e
depuração do código, e organização dos experimentos. Todo o código foi escrito,
executado e compreendido pela equipe. O relatório foi redigido integralmente pela
equipe, sem uso de IA para a escrita.

## Referências principais
- Yang et al. (2023). MedMNIST v2. *Scientific Data*, 10(1).
- He et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Dosovitskiy et al. (2021). An Image is Worth 16×16 Words. *ICLR*.
- Selvaraju et al. (2017). Grad-CAM. *ICCV*.  
