# Trabalho Final — Inteligência Artificial
## Do NumPy ao Estado da Arte em Visão Computacional
### Classificação de Tecidos Histopatológicos com PathMNIST

Centro Universitário Católica de Quixadá — UniCatólica
Curso de Sistemas de Informação · Disciplina de Inteligência Artificial

## Equipe
- Rawenne da Silva Leite
- 
- 

## Sobre o projeto
Classificação multiclasse (9 classes) de tecidos histopatológicos de câncer
colorretal usando o dataset PathMNIST (coleção MedMNIST v2). O trabalho percorre
da implementação manual de uma rede neural em NumPy até arquiteturas estado da
arte (CNNs e Vision Transformers), com foco em rigor experimental e
interpretabilidade (XAI).

## Ambiente de execução
- Plataforma: Kaggle Notebooks
- Etapa 1: CPU (NumPy puro, sem frameworks)
- Python 3.x
- Bibliotecas: numpy, medmnist, matplotlib
- Seed fixada: 42

## Progresso das etapas
- [x] Etapa 1 — MLP do zero em NumPy (forward, backprop, SGD com Momentum, gradient checking)
- [ ] Etapa 2 — Migração e validação em PyTorch
- [ ] Etapa 3 — CNN própria, 3 CNNs pré-treinadas e ViT (Transfer Learning)
- [ ] Etapa 4 — Explicabilidade (Feature Maps e Grad-CAM)
- [ ] Etapa 5 — Modelo final e avaliação única no teste

## Como rodar
1. Abrir o notebook `notebooks/01_numpy_mlp.ipynb` no Kaggle (Internet ligada).
2. Executar as células na ordem (Run All).
3. O dataset PathMNIST é baixado automaticamente via pacote `medmnist`.

## Etapa 1 — Resultados
- Gradient checking aprovado: diferença relativa na ordem de 1e-8 a 1e-9 (< 1e-5 exigido).
- Acurácia de validação: ~50%, coerente com uma MLP densa sobre imagens 28×28
  achatadas (a estrutura espacial é descartada — limitação superada pelas CNNs nas etapas seguintes).
- Treino e validação com curvas sobrepostas (sem overfitting).

## Estrutura do repositório
