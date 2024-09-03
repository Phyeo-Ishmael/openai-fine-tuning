# 📝适用于GPT的模型微调   ——在低耗费情况下实现适应性文本模型 
[![Open In Colab](https://colab.research.google.com/drive/1vvQq6-BksAkRun_GeMYqM8D8UREG2y9o?usp=sharing)
 
 愿你得到自己的罗盘，找到自己的航向。
 
 本教程旨在能够方便快捷的引导部署openai模型的微调,理论上携带有训练数据集与可使用的apikey就可以通过指令训练出适合的对话式模型。

这篇笔记使用gpt仅起到推荐作用，深入研究请广泛寻找合适的平台与模型。如有需要，可以参考[官方文档](https://platform.openai.com/docs/guides/fine-tuning)

具体流程见下。

## 配置🛠️
**安装依赖**
```python
pip install pandas
pip install openai==0.28
```
**配置Api key**
```python
import os
os.environ['OPENAI_API_KEY'] = '******' # 输入key
import openai
openai.api_key = os.environ['OPENAI_API_KEY']
```
这样就好，现在，需要的配置和api都导入了。如有必要，我们可以检查一下相应的配置:
```python
echo $OPENAI_API_KEY
```
>以上完成基本准备工作，OpenAI的Api Key需要在官网处自行申请。理论可使用colab的代理也可以直接登录官网界面，但是这里也给出大陆接口网站:[F2API](https://f2api.com/?r=8270)

## 准备数据📑
那么，下面就是数据准备的工作。

针对不同的模型，训练数据也应当由个人所好来设置。
准备相应的格式数据。考虑到openai的支持范围，文件必须是.JSONL类型， 
它的样式可以是这样:

{
  "messages": [
    { "role": "system", "content": "<放入系統訊息>" },
    { "role": "user", "content": "<放入使用者的問題>" },
    { "role": "assistant", "content": "<放入理想的回答>." }
  ]
}   


+ 然而,对于数据庞大的文本数据，我们必然不可能逐行地写入信息。

  这种情况下，我们就需要通过csv文件将数据集转换为我们需要的格式。
>准备csv文件


| system_content | user_content |  assistant_content  |
| :------------: | :----------: |:------------------: |
| 对于系统的提示词 |对话者的讲述内容| 得到的回答 | 
| 对于系统的提示词 |对话者的讲述内容| 得到的回答 |
|......  |......   | ......  | 

......  随后，我们可以快速转换为jsonl格式的文件。
<details>
<summary style="font-weight: bold; font-size: larger;">点击查看</summary>

```python
import csv
import json

csv_file = 'sample.csv'  # 替换为你的CSV文件名
jsonl_file = 'sample.jsonl'  # 输出的JSONL文件名

# 定义列名映射
column_mapping = {
    'system_content': ['system_content', 'System Content'],
    'user_content': ['user_content', 'User Content'],
    'assistant_content': ['assistant_content', 'Assistant Content']
}

def get_column_name(header, mapping):
    for key, names in mapping.items():
        if header in names:
            return key
    raise KeyError(f"Column name '{header}' is not recognized")

# 去掉BOM的函数
def remove_bom(text):
    if text.startswith('\ufeff'):
        return text[1:]
    return text

with open(csv_file, 'r', newline='', encoding='utf-8-sig') as csvfile, open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
    reader = csv.DictReader(csvfile)
    # 去掉列名中的BOM
    headers = [remove_bom(header) for header in reader.fieldnames]
    # 获取实际列名并转换为标准列名
    mapped_headers = {header: get_column_name(header, column_mapping) for header in headers}

    for row in reader:
        messages = [
            {"role": "system", "content": row.get(mapped_headers.get('system_content', ''), '')},
            {"role": "user", "content": row.get(mapped_headers.get('user_content', ''), '')},
            {"role": "assistant", "content": row.get(mapped_headers.get('assistant_content', ''), '')}
        ]
        jsonlfile.write(json.dumps({"messages": messages}) + '\n')

print("CSV file has been converted to JSONL format.")
```

</details>

当然，你也可以使用openai预置的CLI数据准备工具，不过会有一点点麻烦:

```python
openai tools fine_tunes.prepare_data -f sample.csv

```

## 完成准备工作，准备开始微调✏

现在，我们的准备工作已经完成，接下来使用系统的指令训练。

```python
openai api fine_tunes.create -t "sample_prepared.jsonl" 
```
出现了报错，请不要担心，这也是正常的情况。

这时候我们需要单独使用完成的文件对脚本进行微调，考虑使用id切换文件:
```python
openai api files.create -f "sample_prepared.jsonl" -p "fine-tune"
```
> + 得到的输出会包含文件的id，以便下一步使用。

我们先查看可使用的模型id:
```python
models = openai.Model.list()
print(models)
```
选择好需要的模型后，就可以开始微调了:
```python
response = openai.FineTuningJob.create(
    training_file="******",  # 上传后的文件 ID
    model="gpt-3.5-turbo"  # 使用支持微调的模型名称，例如 "gpt-3.5-turbo"
)

print(response)
```
提交任务过后，可以通过以下方式检查你的作业进程:
```python
job_id = '******'  # 替换为你的任务 ID

while True:
    response = openai.FineTuningJob.retrieve(id='******')
    status = response['status']
    print(f"Current status: {status}")

    if status in ['succeeded', 'failed']:
        break

    time.sleep(60)  # 每分钟检查一次

```

好，到这里应该得到了一个已经训练好的模型，在你的openai playground当中会有显示，可供使用。

最后，还请不要忘记[估算你的token数与相关的成本。](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset)
            
## 备注
+ 条件允许，建议要预估成本，否则可能会出现意想不到的开销。
+ 数据集文件csv请务必采用UTF-8格式的，否则会无法解析并报错。


我是以实玛利，期待我们下次再见。

