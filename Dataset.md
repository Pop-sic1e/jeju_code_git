## AI-Hub Data Download Instructions

The datasets used in this study were obtained from **Domestic Travel Log Data (Jeju and Island Regions, 2023)**.
All data information can be accessed through **AI-Hub ([www.aihub.or.kr](http://www.aihub.or.kr))**.

Please note that exporting the original data outside Korea requires prior consent from the **National Information Society Agency (NIA)**.
For this reason, and because of the large data size, this GitHub repository only includes a **1/10 sample dataset** for reproducibility testing.

Researchers who have access to AI-Hub and obtain an API key from **AI-Hub ([www.aihub.or.kr](http://www.aihub.or.kr))** can download the original data used in this study using the following commands.

```bash
# 1. aihubshell download
curl -o "aihubshell" https://api.aihub.or.kr/api/aihubshell.do
chmod +x aihubshell

# 2. find dataset key
./aihubshell -mode l | grep "국내 여행로그 데이터"

# 3. find file key
./aihubshell -mode l -datasetkey 71780

# 4. download dataset
./aihubshell -mode d -datasetkey 71780 -filekey 541665,541667,541668,541670,541671,549769 -aihubapikey 'API KEY'
```

After the download is completed, `aihubshell` automatically merges split archive files, extracts compressed files, and removes temporary archive files. Therefore, users should prepare sufficient storage space before downloading the full dataset.
