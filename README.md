# CIFAR-10 AlexNet Experiments

This repository contains PyTorch implementations of AlexNet trained on CIFAR-10, exploring various performance optimization techniques: ONNX export, PyTorch JIT Trace mode, Automatic Mixed Precision (AMP), and Knowlege Distillation. These experiments demonstrate trade-offs between inference speed, training time, and model accuracy.

## Folder Structure

cifar10-experiments/  
├── pytorch/  
│   ├── baseline/                 # Baseline AlexNet implementation  
│   ├── amp/                      # AlexNet with Automatic Mixed Precision  
│   ├── knowledge distillation/   # Knowledge Distillation with Teacher and Student Models  
│   └── onnx-jit/                 # ONNX export and JIT trace mode experiments  
├── tensorflow/                   # TensorFlow AlexNet implementation (not covered here)  
├── README.md  
└── LICENSE  

## Experiments and Results

### 1. ONNX Runtime Optimization

- **Inference Speed:**  
  Original: 5.4511 seconds  
  ONNX Exported: 4.7638 seconds  
  **Improvement:** 0.6873 seconds (~12.6%)

- **Insights:**  
  ONNX optimizations such as static graph analysis and operator fusion enable faster inference.

---

### 2. PyTorch JIT Trace Mode

- **Inference Speed:**  
  Original: 5.4511 seconds  
  JIT Trace: 4.9518 seconds  
  **Improvement:** 0.4993 seconds (~9.15%)

- **Insights:**  
  Trace mode optimizes static graph execution by removing Python overhead. The model’s static control flow enables this optimization.

---

### 3. Automatic Mixed Precision (AMP)

| Metric                  | Default FP32     | AMP Applied FP16  |
|-------------------------|------------------|-------------------|
| Best Training Accuracy   | 63.49% (Epoch 32)| 63.28% (Epoch 35) |
| Best Validation Accuracy | 67.74%           | 68.18%            |
| Best Training Loss       | 1.0428           | 1.0447            |
| Best Validation Loss     | 0.8921           | 0.8922            |
| Training Time (seconds)  | 2103.02          | 1601.25           |
| Test Accuracy            | 64.89%           | 61.37%            |
| Test Time (seconds)      | 5.51             | 4.55              |

- **Insights:**  
  AMP reduces training and inference times significantly (~23.9% faster training, ~17.4% faster inference) by leveraging FP16 and Tensor Cores.  
  However, AMP shows a slight accuracy drop (~3.5%), possibly due to reduced numerical precision.

---

### 4. Knowledge Distillation

| Model Performance              | Accuracy         | Inference Speed      |
|--------------------------------|------------------|----------------------|
| Teacher Model (AlexNet):       | 71.51%           | 23.8994 seconds      |
| Student Model (from scratch)   | 60.37%           | 23.3891 seconds      |
| Student Model (with KD)        | 63.07%           | 21.9342 second       |

- **Insights:**  
The teacher model achieved the highest accuracy at 71.51% but had the slowest inference time (23.90s). The distilled student model was faster (21.93s) but less accurate (63.07%), though still 2.70% more accurate than a student model trained without distillation. This highlights a trade-off: the teacher model is more accurate but slower, while the distilled model offers a good balance by achieving about 90% of the teacher’s accuracy with faster inference and lower cost.

---

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/your-username/cifar10-experiments.git
cd cifar10-experiments
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```
