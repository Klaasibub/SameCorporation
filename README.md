# Find Similar Companies
## About
Find Similar Companies is a project that will allow you to find the most simmilar company names for your given input.
For the development of this project, both general techniques from NLP and machine learning are used.  
This project are published on [hugging face spaces](https://huggingface.co/spaces/Vsevolod/find-similar-companies)! Our production model are also present on [hugging face models](https://huggingface.co/Vsevolod/company-names-similarity-sentence-transformer).

![Score](./media/same.png)  

---

### Installation
In order to run inference you need to install the necessary dependencies:
```
pip install -r requirements.txt
```

---

## Inference
To launch inference run the server script:
```bash
python src/server.py
```

---

## Test stand
| Type            | Model                       |
|-----------------|-----------------------------|
| CPU             | Intel Core i5-3470          |
| GPU (optional)  | NVIDIA GeForce GTX 1060 6gb |
| RAM             | Crucial DDR3 1600MHz 8GB x2 |

---

## Comparison
| Method                       | F1 - score | Accuracy   | Precision  | Recall     | Performance   |
|------------------------------|------------|------------|------------|------------|---------------|
| word-by-word comparison      | 0.3540     | 0.9931     | 0.5398     | 0.2633     | **4.9571**    |
| Levenshtein distance         | 0.3499     | 0.9931     | 0.546      | 0.2574     | 6.4292        |
| TF-IDF                       | 0.5204     | 0.9918     | 0.457      | 0.6042     | -             |
| TF-IDF + Logistic regression | 0.5009     | 0.9914     | 0.4336     | 0.593      | -             |
| fastText cosine similarity   | 0.409      | 0.9916     | 0.4629     | 0.3664     | 15.0971       |
| sentence-bert (pretrained)   | 0.4459     | 0.9925     | 0.4223     | 0.4724     | 14.9001 (GPU) |
| sentence-bert (fine-tuned)   | **0.8815** | **0.9982** | **0.8642** | **0.8996** | 15.2045 (GPU) |

Performance is a value (in seconds) for which the entire dataset (500k rows) is processed by method.
For fastText and sentence-bert methods sentences embeddings are cached.
Also, for sentence-bert, caching done by passing all unique names (17k samples) in one batch to GPU.

---

## License
[MIT](https://choosealicense.com/licenses/mit/)
