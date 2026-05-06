# PRD — Fase 0: Preparación sin GPU para CodePause

> **UPDATE (Phase 1C Pivot)**: The official execution environment has been changed from AMD MI300X to Google Colab T4. All references to "AMD Developer Cloud" in this document are now considered historical/optional.

## 1. Nombre del producto

**CodePause: Think-Anywhere Code Generation on AMD**

## 2. Fase cubierta por este PRD

Este PRD cubre **solo la fase previa a la aprobación del crédito AMD/DigitalOcean**.

No incluye entrenamiento real en GPU, despliegue en AMD Developer Cloud, GRPO completo ni demo pública final.

La idea central de esta fase es dejar listo todo lo que se puede preparar localmente:

```text
repo + dataset + evaluador + scripts + README + demo mock
```

Cuando aprueben el crédito, la GPU debe usarse para ejecutar, no para improvisar.

---

## 3. Contexto técnico

El proyecto se basa en el paper **Think Anywhere in Code Generation**.

El mecanismo central del paper permite que el modelo invoque razonamiento durante la generación del código. Fuente: [arXiv — Think Anywhere in Code Generation](https://arxiv.org/html/2603.29957v3), cita textual: “enables LLMs to invoke thinking on-demand at any token position during code generation”.

El paper propone una canalización de dos etapas. Fuente: [arXiv — Think Anywhere in Code Generation](https://arxiv.org/html/2603.29957v3), cita textual: “First, through cold-start training” y “Second, we employ RLVR”.

Para esta fase 0, solo se preparará la base para la primera etapa: dataset, formato, validación y script SFT/LoRA. La parte RLVR queda fuera de alcance, aunque se dejará una estructura mínima para rewards.

---

## 4. Problema

Actualmente el proyecto tiene una dependencia bloqueante: **no hay acceso activo a GPU AMD todavía**.

Eso no impide avanzar, porque las piezas más importantes del MVP no dependen de GPU:

```text
- definición del formato <thinkanywhere>
- dataset de entrenamiento
- dataset de evaluación
- limpieza de bloques de pensamiento
- ejecución de tests
- métricas
- README
- demo mock
- scripts listos para correr en AMD Cloud
```

El riesgo principal sería esperar pasivamente la aprobación del crédito y luego gastar horas GPU resolviendo detalles de producto, estructura de repo o bugs triviales.

---

## 5. Objetivo de la fase

Construir una base reproducible para que, al recibir acceso a AMD Developer Cloud, se pueda ejecutar inmediatamente:

```text
1. baseline del modelo base
2. SFT/LoRA con dataset Think-Anywhere
3. evaluación contra tests
4. reporte baseline vs fine-tuned
```

---

## 6. No objetivos

Esta fase **no** debe intentar:

```text
- entrenar en GPU
- correr GRPO completo
- optimizar vLLM
- publicar modelo final
- crear una UI perfecta
- hacer benchmark grande
- reproducir el paper completo
```

La frontera mental es simple: esta fase es el **molde de la fábrica**, no la producción.

---

## 7. Usuario objetivo

El usuario principal de esta fase eres tú como builder del hackathon.

El usuario secundario es cualquier juez o reviewer técnico que abra el repo y quiera entender:

```text
- qué problema resuelve
- cómo se genera el dataset
- cómo se validan outputs
- cómo se remueven los bloques <thinkanywhere>
- cómo se miden resultados
- cómo se correrá luego en AMD Cloud
```

---

## 8. Propuesta de solución

Crear un repositorio mínimo pero completo llamado:

```text
codepause-amd
```

El repositorio debe contener:

```text
- problemas de programación con tests
- ejemplos SFT con <think> y <thinkanywhere>
- limpiador de bloques de pensamiento
- validador de formato
- runner de tests
- cálculo de métricas
- script de SFT/LoRA preparado
- script de evaluación baseline preparado
- README técnico
- demo mock
```

El paper exige que el código resultante sea ejecutable luego de remover bloques de pensamiento. Fuente: [arXiv — Think Anywhere in Code Generation](https://arxiv.org/html/2603.29957v3), cita textual: “The final executable code is obtained by removing all thinking blocks”.

También especifica una restricción directa sobre los bloques `<thinkanywhere>`. Fuente: [arXiv — Think Anywhere in Code Generation](https://arxiv.org/html/2603.29957v3), cita textual: “The code must remain valid and executable after removing all `<thinkanywhere>...</thinkanywhere>` segments”.

---

## 9. Requisitos funcionales

### RF-1 — Crear estructura del repo

El repo debe tener esta estructura:

```text
codepause-amd/
├── README.md
├── requirements.txt
├── data/
│   ├── problems.jsonl
│   ├── thinkanywhere_sft.jsonl
│   └── eval_cases.jsonl
├── src/
│   ├── strip_thinking.py
│   ├── sandbox_runner.py
│   ├── metrics.py
│   └── prompts.py
├── training/
│   ├── sft_lora.py
│   └── rewards.py
├── eval/
│   ├── evaluate_baseline.py
│   └── make_report.py
├── app/
│   └── demo_mock.py
└── results/
    └── .gitkeep
```

### RF-2 — Crear dataset de problemas

Crear `data/problems.jsonl` con al menos **30 problemas iniciales**.

Cada problema debe incluir:

```json
{
  "id": "binary_search_001",
  "prompt": "Write a Python function search(nums, target)...",
  "entry_point": "search",
  "tests": [
    "assert search([1,2,3], 2) == 1",
    "assert search([1,2,3], 4) == -1"
  ]
}
```

Los problemas deben priorizar casos donde los modelos suelen fallar:

```text
- índices
- límites
- listas vacías
- loops
- recursión
- parsing
- ramas if/else
- casos borde
```

### RF-3 — Crear dataset SFT Think-Anywhere

Crear `data/thinkanywhere_sft.jsonl` con al menos **30 ejemplos alineados a los problemas del dataset**.

Cada ejemplo debe usar esta estructura:

```text
<think>plan general</think>
código con uno o más bloques <thinkanywhere>
```

TRL soporta datasets de tipo language modeling y prompt-completion. Fuente: [Hugging Face TRL — SFT Trainer](https://huggingface.co/docs/trl/sft_trainer), cita textual: “SFT supports both language modeling and prompt-completion datasets”.

Para esta fase, se usará formato `text` por simplicidad:

```json
{
  "text": "### Instruction\n...\n\n### Response\n<think>...</think>\n```python\n...\n<thinkanywhere>...</thinkanywhere>\n...\n```"
}
```

### RF-4 — Implementar limpieza de pensamiento

Crear `src/strip_thinking.py`.

Debe remover:

```text
<think>...</think>
<thinkanywhere>...</thinkanywhere>
```

Debe exponer estas funciones:

```python
strip_thinking_blocks(text: str) -> str
has_balanced_tags(text: str) -> bool
count_thinkanywhere_blocks(text: str) -> int
```

Criterio de aceptación:

```text
input con tags → output sin tags
output sin tags → código Python ejecutable
tags desbalanceados → métrica de error
```

### RF-5 — Implementar runner de tests

Crear `src/sandbox_runner.py`.

Debe ejecutar código Python generado contra tests del dataset.

pytest es una opción válida para tests pequeños y legibles. Fuente: [pytest documentation](https://docs.pytest.org/), cita textual: “The pytest framework makes it easy to write small, readable tests”.

Para esta fase, se permite un runner simple basado en `subprocess`, con timeout.

Debe retornar:

```json
{
  "passed": true,
  "stdout": "",
  "stderr": "",
  "returncode": 0,
  "timeout": false
}
```

### RF-6 — Implementar métricas

Crear `src/metrics.py`.

Debe calcular:

```text
- tests_passed
- balanced_tags
- has_thinkanywhere
- thinkanywhere_blocks
- executable_after_strip
- clean_code_lines
```

### RF-7 — Preparar script SFT/LoRA

Crear `training/sft_lora.py`.

No necesita correr todavía en GPU, pero debe quedar listo para ejecutarse en AMD Cloud.

AMD documenta el entorno de fine-tuning single accelerator con estas librerías. Fuente: [ROCm Docs — Single GPU fine-tuning and inference](https://rocm.docs.amd.com/en/docs-6.4.1/how-to/rocm-for-ai/fine-tuning/single-gpu-fine-tuning-and-inference.html), cita textual: “transformers datasets huggingface-hub peft trl scipy”.

La estrategia LoRA tiene sentido para esta fase porque PEFT permite entrenar pocos parámetros adicionales sobre un modelo preentrenado. Fuente: [Hugging Face Transformers — PEFT](https://huggingface.co/docs/transformers/en/peft), cita textual: “only fine-tune a small number of extra model parameters (adapters) on top of a pretrained model”.

El script debe permitir configurar:

```text
- model_name
- dataset_path
- output_dir
- epochs
- max_seq_length
- learning_rate
- lora_rank
- gradient_accumulation_steps
```

### RF-8 — Preparar script baseline

Crear `eval/evaluate_baseline.py`.

Debe aceptar:

```text
--model_name
--problems_path
--output_path
```

Debe producir:

```text
results/baseline.jsonl
```

Cada línea debe incluir:

```json
{
  "id": "binary_search_001",
  "prompt": "...",
  "raw_output": "...",
  "clean_code": "...",
  "passed": true,
  "metrics": {}
}
```

### RF-9 — Crear README técnico

El README debe explicar:

```text
- problema
- solución
- relación con Think-Anywhere
- pipeline
- estructura del repo
- cómo correr tests locales
- cómo correr baseline cuando haya GPU
- cómo correr SFT cuando haya GPU
- métricas esperadas
```

Debe incluir una sección explícita:

```text
GPU execution plan
```

AMD describe AMD Developer Cloud como una plataforma con software ROCm y Docker preparados para cargas de IA. Fuente: [AMD — How to Get Started on the AMD Developer Cloud](https://www.amd.com/en/developer/resources/technical-articles/2025/how-to-get-started-on-the-amd-developer-cloud-.html), cita textual: “preconfigured ROCm™ software, and Docker setups”.

La misma guía indica que las imágenes Quick Start incluyen varios paquetes. Fuente: [AMD — How to Get Started on the AMD Developer Cloud](https://www.amd.com/en/developer/resources/technical-articles/2025/how-to-get-started-on-the-amd-developer-cloud-.html), cita textual: “vLLM”, “SGLang”, “PyTorch”, “Megatron” y “JAX”.

### RF-10 — Crear demo mock

Crear `app/demo_mock.py`.

No debe depender de modelo real.

Debe mostrar:

```text
- prompt
- output con <thinkanywhere>
- output limpio
- resultado de tests simulado o local
- métricas básicas
```

Puede hacerse con Streamlit o Gradio.

---

## 10. Requisitos no funcionales

### RNF-1 — Reproducibilidad

Todo debe poder ejecutarse desde cero con:

```bash
pip install -r requirements.txt
python -m pytest
```

### RNF-2 — Bajo costo

La fase no debe requerir GPU.

### RNF-3 — Simplicidad

No introducir frameworks innecesarios.

La demo puede ser fea pero funcional.

### RNF-4 — Separación clara

El repo debe separar:

```text
src/        lógica reusable
training/   scripts de entrenamiento
eval/       evaluación
app/        demo
data/       datasets
results/    outputs
```

### RNF-5 — Seguridad local

El runner de código generado debe usar timeout.

No debe permitir ejecución indefinida.

---

## 11. Criterios de aceptación

La fase se considera completa cuando exista:

```text
1. Repo creado y subido a GitHub.
2. requirements.txt funcional.
3. 30 problemas en data/problems.jsonl.
4. 30 ejemplos en data/thinkanywhere_sft.jsonl.
5. strip_thinking.py probado.
6. sandbox_runner.py probado.
7. metrics.py probado.
8. sft_lora.py preparado.
9. evaluate_baseline.py preparado.
10. README técnico escrito.
11. demo_mock.py ejecutable localmente.
```

Criterio mínimo de calidad:

```text
- 100% de ejemplos SFT deben tener tags balanceados.
- 100% de ejemplos SFT deben producir código limpio después de strip.
- Al menos 80% de ejemplos manuales deben pasar tests después de strip.
- El repo debe poder clonarse y validarse sin GPU.
```

---

## 12. Métricas de la fase

Esta fase no mide mejora del modelo todavía.

Mide preparación:

```text
dataset_problem_count >= 30
sft_example_count >= 30
tag_balance_rate = 100%
strip_success_rate = 100%
local_test_pass_rate >= 80%
repo_bootstrap_time <= 10 minutos
```

---

## 13. Riesgos

### Riesgo 1 — Dataset artificial de baja calidad

Si los ejemplos `<thinkanywhere>` son demasiado obvios o repetitivos, el fine-tuning aprenderá formato, pero no buen posicionamiento.

Mitigación:

```text
poner <thinkanywhere> cerca de decisiones reales:
- if
- while
- índices
- casos borde
- recursión
```

### Riesgo 2 — Pensamiento que rompe el código

El paper exige que el código siga siendo válido después de remover bloques. Fuente: [arXiv — Think Anywhere in Code Generation](https://arxiv.org/html/2603.29957v3), cita textual: “valid and executable after removing all `<thinkanywhere>...</thinkanywhere>` segments”.

Mitigación:

```text
validar cada ejemplo con strip_thinking.py + sandbox_runner.py
```

### Riesgo 3 — Gastar GPU en debugging trivial

AMD indica que en Quick Start se entra al contenedor con un comando específico. Fuente: [AMD — How to Get Started on the AMD Developer Cloud](https://www.amd.com/en/developer/resources/technical-articles/2025/how-to-get-started-on-the-amd-developer-cloud-.html), cita textual: “docker exec -it rocm bash”.

Mitigación:

```text
tener scripts listos antes de crear la VM
```

### Riesgo 4 — Scope creep hacia GRPO

El paper dice que el método completo incluye RLVR. Fuente: [arXiv — Think Anywhere in Code Generation](https://arxiv.org/html/2603.29957v3), cita textual: “we employ RLVR to further reinforce this capability”.

Mitigación:

```text
dejar GRPO como Fase 2
esta fase solo prepara rewards.py como stub
```

---

## 14. Plan de trabajo sugerido

### Bloque 1 — Repo y estructura

```text
Crear repo
Crear carpetas
Crear requirements.txt
Crear README base
```

### Bloque 2 — Dataset

```text
Crear 30 problemas
Crear 30 ejemplos SFT
Validar JSONL
```

### Bloque 3 — Core utilities

```text
strip_thinking.py
sandbox_runner.py
metrics.py
prompts.py
```

### Bloque 4 — Scripts futuros

```text
evaluate_baseline.py
sft_lora.py
rewards.py stub
make_report.py stub
```

### Bloque 5 — Demo mock

```text
demo_mock.py
ejemplo hardcodeado
visualización de tags
código limpio
resultado de tests
```

---

## 15. Definición de “done”

Esta fase termina cuando puedas ejecutar localmente:

```bash
python src/strip_thinking.py
python src/sandbox_runner.py
python eval/evaluate_baseline.py --mock
python app/demo_mock.py
```

Y cuando el README tenga una sección clara que diga:

```text
When AMD Developer Cloud access is approved:
1. Create MI300X GPU Droplet
2. Select Quick Start → PyTorch
3. SSH into the VM
4. Run docker exec -it rocm bash
5. Clone this repo
6. Install requirements
7. Run baseline
8. Run SFT/LoRA
9. Run evaluation
```

AMD documenta ese flujo general como elección de plan, imagen, SSH key, creación de VM y acceso por SSH. Fuente: [AMD — How to Get Started on the AMD Developer Cloud](https://www.amd.com/en/developer/resources/technical-articles/2025/how-to-get-started-on-the-amd-developer-cloud-.html), cita textual: “choosing a GPU plan and software image, adding an SSH key, creating the virtual machine with a single click, and accessing from their preferred SSH client”.

---

## 16. Entregable final de esta fase

El entregable no es un modelo entrenado.

El entregable es un repo listo para GPU:

```text
GitHub repo funcional
dataset inicial
evaluador local
scripts de entrenamiento preparados
demo mock
README técnico
```

La medida de éxito es que, cuando aprueben el crédito, puedas pasar de cero a primer entrenamiento sin rediseñar nada.
