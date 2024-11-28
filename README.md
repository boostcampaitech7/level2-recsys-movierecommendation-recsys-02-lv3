<div align='center'>
<p align='center'>
    <img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=250&section=header&text=It's%20me,%20RecSys%20🤓&fontSize=60&animation=fadeIn&fontAlignY=38&desc=Lv2%20Project&descAlignY=51&descAlign=80"/>
</p>
    <img src="https://github.com/user-attachments/assets/d3ac8d42-2738-4ddb-8b09-fa346494bcd6" width=500/>

</div>

# 🍿 LV.3 RecSys 프로젝트 : Movie Recommendation



## 🏆 대회 소개
| 특징 | 설명 |
|:---:|---|
| 대회 주제 | 네이버 부스트캠프 AI-Tech 7기 RecSys level3 - Movie Recommendation|
| 데이터 구성 | `train_ratings.tsv, genres.tsv, titles.tsv` 총 3개의 CSV 파일 |
| 평가 지표 | Recall@10로 전체 선호 아이템 중 추천된 아이템이 얼마나 존재하는지를 측정한다. |

---
## 💻 팀 구성 및 역할
| 박재욱 | 서재은 | 임태우 | 조유솔 | 최태순 | 허진경 |
|:---:|:---:|:---:|:---:|:---:|:---:|
|[<img src="https://github.com/user-attachments/assets/0c4ff6eb-95b0-4ee4-883c-b10c1a42be14" width=130>](https://github.com/park-jaeuk)|[<img src="https://github.com/user-attachments/assets/b6cff4bf-79c8-4946-896a-666dd54c63c7" width=130>](https://github.com/JaeEunSeo)|[<img src="https://github.com/user-attachments/assets/f6572f19-901b-4aea-b1c4-16a62a111e8d" width=130>](https://github.com/Cyberger)|[<img src="https://avatars.githubusercontent.com/u/112920170?v=4" width=130>](https://github.com/YusolCho)|[<img src="https://github.com/user-attachments/assets/a10088ec-29b4-47aa-bf6a-53520b6106ce" width=130>](https://github.com/choitaesoon)|[<img src="https://github.com/user-attachments/assets/7ab5112f-ca4b-4e54-a005-406756262384" width=130>](https://github.com/jinnk0)|
|Data EDA, BERT4Rec with Side-information, Multi DAE|Data EDA, RecBole Setting, RecVAE|Baseline Setting, Modularization, SASRec, RecVAE|Data EDA, BERT4Rec, EASER, Hard Ensemble module|Baseline Setting, Modularization, EASE, Soft Ensemble module|Data EDA, DeepFM, MF with TF-IDF|
---
## 🎬 프로젝트 개요
|    개요    | 설명 |
|:---:| --- |
| 주제 | 사용자의 영화 시청 이력 데이터를 기반으로 암묵적 피드백(implicit feedback)과 순차적 이력(time-ordered sequence)을 활용하여 추천 시스템을 개발합니다. 이 문제는 기존의 명시적 피드백(explicit feedback, 평점 기반) 방식과는 달리, 일부 시청 이력이 누락된 현실적인 시나리오를 다루며, 다양한 부가 정보(side-information)를 통합하여 추천 성능을 향상시키는 방법도 탐구합니다.  |
| 목표 | 누락된 시청 이력을 포함한 복잡한 순차적 추천 문제를 해결하고, 사용자가 다음에 시청하거나 좋아할 영화를 효과적으로 예측하며, 아이템 관련 부가 정보를 활용한 정교한 추천 모델을 설계합니다. |
| 평가 지표 | **Recall@10**  |
| 개발 환경 | `GPU` : Tesla V100 Server 4대, `IDE` : VSCode, Jupyter Notebook, Google Colab |
| 협업 환경 | `Notion`(진행 상황 공유), `Github`(코드 및 데이터 공유), `Slack` , `카카오톡`(실시간 소통), `WandB`(모델링 상황 공유) |


### 데이터셋 구성
>- `train_ratings.tsv` : 사용자의 영화 시청 정보

| 컬럼명 | 설명 |
| --- | --- |
|`user`|사용자 ID|
|`item`|영화 ID|
|`time`|영화를 시청한 timestamp|


>- `genres.tsv` : 영화 및 장르 정보

| 컬럼명 | 설명 |
| --- | --- |
|`item`|영화 ID|
|`genre`|영화 장르|

>- `titles.tsv` : 영화 제목 정보

| 컬럼명 | 설명 |
| --- | --- |
|`item`|영화 ID|
|`title`|영화 제목|




----
## 🕹️ 프로젝트 실행
### 디렉토리 구조

```
📦level2-recsys-movierecommendation-recsys-02-lv3
 ┣ 📂configs
 ┃ ┗ 📜config.yaml
 ┣ 📂src
 ┃ ┣ 📂BERT4Rec
 ┃ ┃ ┣ 📜BERT4Rec.py
 ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┣ 📜config.yaml
 ┃ ┃ ┣ 📜dataloader.py
 ┃ ┃ ┣ 📜dataset.py
 ┃ ┃ ┣ 📜inference.py
 ┃ ┃ ┣ 📜main.py
 ┃ ┃ ┣ 📜train.py
 ┃ ┃ ┗ 📜utils.py
 ┃ ┣ 📂BERT4Rec_with_side_info
 ┃ ┃ ┣ 📜config.yaml
 ┃ ┃ ┣ 📜dataloader.py
 ┃ ┃ ┣ 📜dataset.py
 ┃ ┃ ┣ 📜inference.py
 ┃ ┃ ┣ 📜main.py
 ┃ ┃ ┣ 📜metrics.py
 ┃ ┃ ┣ 📜model.py
 ┃ ┃ ┗ 📜train.py
 ┃ ┣ 📂DeepFM
 ┃ ┃ ┣ 📜DeepFM.py
 ┃ ┃ ┣ 📜config.yaml
 ┃ ┃ ┣ 📜dataloader.py
 ┃ ┃ ┣ 📜dataset.py
 ┃ ┃ ┣ 📜inference.py
 ┃ ┃ ┣ 📜loss.py
 ┃ ┃ ┣ 📜main.py
 ┃ ┃ ┗ 📜train.py
 ┃ ┣ 📂EASE
 ┃ ┃ ┣ 📜EASE.py
 ┃ ┃ ┣ 📜config.yaml
 ┃ ┃ ┣ 📜dataloader.py
 ┃ ┃ ┣ 📜dataset.py
 ┃ ┃ ┣ 📜inference.py
 ┃ ┃ ┣ 📜loss.py
 ┃ ┃ ┗ 📜main.py
 ┃ ┣ 📂EASER
 ┃ ┃ ┣ 📜EASER.py
 ┃ ┃ ┣ 📜config.yaml
 ┃ ┃ ┣ 📜dataloader.py
 ┃ ┃ ┣ 📜dataset.py
 ┃ ┃ ┗ 📜main.py
 ┃ ┣ 📂MF
 ┃ ┃ ┣ 📜MF.py
 ┃ ┃ ┣ 📜config.yaml
 ┃ ┃ ┣ 📜main.py
 ┃ ┃ ┗ 📜util.py
 ┃ ┣ 📂MultiVAE
 ┃ ┃ ┣ 📜MultiVAE.py
 ┃ ┃ ┣ 📜config.yaml
 ┃ ┃ ┣ 📜config_wandb.yaml
 ┃ ┃ ┣ 📜dataloader.py
 ┃ ┃ ┣ 📜dataset.py
 ┃ ┃ ┣ 📜inference.py
 ┃ ┃ ┣ 📜loss_fn.py
 ┃ ┃ ┣ 📜main.py
 ┃ ┃ ┗ 📜train.py
 ┃ ┣ 📂RecBole
 ┃ ┃ ┣ 📜README.md
 ┃ ┃ ┣ 📜dataloader.py
 ┃ ┃ ┣ 📜inference.py
 ┃ ┃ ┣ 📜recbole_model.yaml
 ┃ ┃ ┣ 📜recbole_setting.yaml
 ┃ ┃ ┗ 📜train.py
 ┃ ┣ 📂RecVAE
 ┃ ┃ ┣ 📜RecVAE.py
 ┃ ┃ ┣ 📜config.yaml
 ┃ ┃ ┣ 📜dataloader.py
 ┃ ┃ ┣ 📜dataset.py
 ┃ ┃ ┣ 📜inference.py
 ┃ ┃ ┣ 📜main.py
 ┃ ┃ ┗ 📜train.py
 ┃ ┗ 📂utils
 ┃ ┃ ┣ 📜metrics.py
 ┃ ┃ ┗ 📜util.py
 ┣ 📜README.md
 ┣ 📜ensemble.py
 ┣ 📜hard_ensemble.py
 ┣ 📜main.py
 ┣ 📜requirements.txt
```

### Installation with pip
1. `pip install -r requirements.txt` 실행
2. Unzip train, dev, test csv files at /data directory
3. Upload sample_submission.csv at /data directory
```bash
# Single Model
# [BERT4Rec, BERT4Rec_with_side_info, DeepFM, EASE, EASER, MF, MultiVAE, RecBole, RecVAE]
$ python main.py -m [모델명] -c configs/config.yaml

# Ensemble
$ python main.py -m MultiVAE EASE -sm min_max -ew 1 3
```
