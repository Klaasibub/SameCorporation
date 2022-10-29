# Check Same Corporation
## About
Check Same Corporation is a project that will allow you to determine whether two names belong to the same company.  
For the development of this project, both general techniques from NLP and machine learning are used.  
Also, for ease of use, a web interface has been developed!

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

## Performance
Recommended PC specifications:
| Type | Model                           |
|------|---------------------------------|
| CPU  | Intel Core i5-3470              |
| GPU (optional)  | NVIDIA GeForce GTX 1050 6gb     |
| RAM  | Crucial DDR3 1600MHz 8GB |


## Comparsion

| Method | F1 | Accuracy | Precision | Recall|
|--------|----|----------|-----------|-------|
| tf-idf + log reg | 0.5009 | 0.9914 | 0.4336 | 0.593 |
| idf | 0.5204 | 0.9918 | 0.457 | 0.6042 |

---

## License
[MIT](https://choosealicense.com/licenses/mit/)
