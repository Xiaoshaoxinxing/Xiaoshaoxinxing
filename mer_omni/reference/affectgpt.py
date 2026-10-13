# Function bodies copied verbatim from local AffectGPT (Apache-2.0).
# See provenance.json and LICENSE-AffectGPT. Unused API imports omitted.
import os
import glob
import re
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
config = SimpleNamespace(EMOTION_WHEEL_ROOT=str(Path(__file__).parent / 'emotion_wheel'))

def read_wheel_to_map(xlsx_path):
    store_map = {}
    level1, level2, level3 = "", "", ""

    df = pd.read_excel(xlsx_path)
    for _, row in df.iterrows():
        row_level1 = row['level1']
        row_level2 = row['level2']
        row_level3 = row['level3']

        # update level1, level2, level3
        if not pd.isna(row_level1):
           level1 = row_level1
        if not pd.isna(row_level2):
           level2 = row_level2
        if not pd.isna(row_level3):
           level3 = row_level3
        
        # store into store_map [存储前需要经过预处理]
        level1 = level1.lower().strip() 
        level2 = level2.lower().strip()
        level3 = level3.lower().strip()
        if level1 not in store_map:
           store_map[level1] = {}
        if level2 not in store_map[level1]:
           store_map[level1][level2] = []
        store_map[level1][level2].append(level3)
    return store_map

def convert_all_wheels_to_candidate_labels():
    candidate_labels = []
    for xlsx_path in glob.glob(config.EMOTION_WHEEL_ROOT + '/wheel*.xlsx'):
        # print (xlsx_path)
        store_map = read_wheel_to_map(xlsx_path)
        # print (store_map)
        for level1 in store_map:
           for level2 in store_map[level1]:
              level3 = store_map[level1][level2]
              candidate_labels.append(level1)
              candidate_labels.append(level2)
              candidate_labels.extend(level3)
    candidate_labels = list(set(candidate_labels))
    return candidate_labels

def func_merge_map(map1, map2):
    all_items = list(map1.keys()) + list(map2.keys())

    merge_map = {}
    for item in all_items:
        if item in map1 and item in map2:
            value = list(set(map1[item] + map2[item]))
        elif item not in map1 and item in map2:
            value = map2[item]
        elif item in map1 and item not in map2:
            value = map1[item]
        merge_map[item] = value
    return merge_map

def read_candidate_synonym_onerun(runname='run1'):

    ## read candidate labels
    wheel_labels = convert_all_wheels_to_candidate_labels()

    ## gain mapping
    label2wheel = {}
    synonym_path = os.path.join(config.EMOTION_WHEEL_ROOT, 'synonym.xlsx')
    df = pd.read_excel(synonym_path)
    for _, row in df.iterrows():

        # 建立 self-mapping
        raw = row[f'word_{runname}'].strip().lower()
        assert raw in wheel_labels, f'error in {raw}' # check openai returns
        if raw not in label2wheel:
            label2wheel[raw] = []
        label2wheel[raw].append(raw)

        # 建立 synonyms -> raw 映射
        synonyms = row[f'synonym_{runname}']
        synonyms = string_to_list(synonyms)
        for synonym in synonyms:
            synonym = synonym.strip().lower()
            if synonym not in label2wheel: 
                label2wheel[synonym] = []
            label2wheel[synonym].append(raw)
    return label2wheel

def read_candidate_synonym_merge():
    mapping_run1 = read_candidate_synonym_onerun('run1')
    mapping_run2 = read_candidate_synonym_onerun('run2')
    mapping_run3 = read_candidate_synonym_onerun('run3')
    mapping_run4 = read_candidate_synonym_onerun('run4')
    mapping_run5 = read_candidate_synonym_onerun('run5') 
    mapping_run6 = read_candidate_synonym_onerun('run6') 
    mapping_run7 = read_candidate_synonym_onerun('run7') 
    mapping_run8 = read_candidate_synonym_onerun('run8') 
    mapping_merge = func_merge_map(mapping_run1,  mapping_run2)
    mapping_merge = func_merge_map(mapping_merge, mapping_run3)
    mapping_merge = func_merge_map(mapping_merge, mapping_run4)
    mapping_merge = func_merge_map(mapping_merge, mapping_run5)
    mapping_merge = func_merge_map(mapping_merge, mapping_run6)
    mapping_merge = func_merge_map(mapping_merge, mapping_run7)
    mapping_merge = func_merge_map(mapping_merge, mapping_run8)
    print (f'label number: {len(mapping_merge)}') 
    return mapping_merge

def read_format2raws():
    format2raws = {}

    format_path = os.path.join(config.EMOTION_WHEEL_ROOT, 'format.csv')
    raws = func_read_key_from_csv(format_path, 'name')
    formats = func_read_key_from_csv(format_path, 'format')
    for raw, format in zip(raws, formats):

        # 1. 建立 format 与 raw 之间的映射
        format = string_to_list(format)
        for format_item in format:
            if format_item not in format2raws:
                format2raws[format_item] = []
            format2raws[format_item].append(raw)
        
        # 2. 建立 raw 与 raw 之间的映射
        if raw not in format2raws:
            format2raws[raw] = []
        format2raws[raw].append(raw)
    print (len(format2raws))
    return format2raws

def func_get_name2reason(reason_root):
    name2reason = {}
    for reason_npy in glob.glob(reason_root + '/*.npy'):
        name = os.path.basename(reason_npy)[:-4]
        reason = np.load(reason_npy).tolist()
        name2reason[name] = reason
    return name2reason

def func_backward_case1(label, format_mapping, raw_mapping=None, wheel_map=None):
    if label not in format_mapping:
        return ""
    
    stage1_labels = format_mapping[label]
    assert isinstance(stage1_labels, list)
    stage1_unique = sorted(stage1_labels)[0]
    
    return stage1_unique

def func_backward_case2(label, format_mapping, raw_mapping, wheel_map=None):
    if label not in format_mapping:
        return ""
    
    stage1_labels = format_mapping[label]
    assert isinstance(stage1_labels, list)
    stage1_unique = sorted(stage1_labels)[0]
    
    stage2_labels = raw_mapping[stage1_unique]
    assert isinstance(stage2_labels, list)
    stage2_unique = sorted(stage2_labels)[0]
    return stage2_unique

def func_get_wheel_cluster(wheel='wheel1', level='level1'):
    # print (f'process on wheel: {wheel} level: {level}')
    xlsx_path = os.path.join(config.EMOTION_WHEEL_ROOT, f'{wheel}.xlsx')
    emotion_wheel = read_wheel_to_map(xlsx_path)
    # print (emotion_wheel.keys())
    wheel_map = {}

    # 1. 所有聚类到level1
    if level == 'level1':
        for level1 in emotion_wheel:
            wheel_map[level1] = level1
            for level2 in emotion_wheel[level1]:
                wheel_map[level2] = level1
                for level3 in emotion_wheel[level1][level2]:
                    wheel_map[level3] = level1

    # 2. 全部聚类到level2
    elif level == 'level2':
        for level1 in emotion_wheel:
            wheel_map[level1] = sorted(emotion_wheel[level1])[0] # level1 映射到一个固定的 level2 上
            for level2 in emotion_wheel[level1]:
                wheel_map[level2] = level2
                for level3 in emotion_wheel[level1][level2]:
                    wheel_map[level3] = level2
    return wheel_map

def func_backward_case3(label, format_mapping, raw_mapping, wheel_map):
    if label not in format_mapping:
        return ""
    
    level1_whole = []
    for format in format_mapping[label]:
        for raw in raw_mapping[format]:
            level1_whole.append(raw)
    
    for level1 in sorted(level1_whole): # 保证了结果唯一性
        if level1 in wheel_map:
            return wheel_map[level1]
    return ""

def func_map_label_to_synonym(mlist, format_mapping, raw_mapping, wheel_map, metric='case1'):
    new_mlist = []
    for label in mlist:
        if metric.startswith('case1'): label = func_backward_case1(label, format_mapping)
        if metric.startswith('case2'): label = func_backward_case2(label, format_mapping, raw_mapping)
        if metric.startswith('case3'): label = func_backward_case3(label, format_mapping, raw_mapping, wheel_map)
        if label == '': continue # 如果找不到 backward 的词，就把他剔除
        new_mlist.append(label)
    return new_mlist

def calculate_openset_overlap_rate(gt_root=None, gt_csv=None, name2gt=None, 
                                   openset_root=None, openset_npz=None, name2pred=None, 
                                   process_names=None, 
                                   metric='case1', format_mapping=None, raw_mapping=None,
                                   inter_print=True):

    # read name2gt
    if name2gt is None:
        if gt_root is not None:
            name2gt = func_get_name2reason(gt_root)
        elif gt_csv is not None:
            name2gt = {}
            names = func_read_key_from_csv(gt_csv, 'name')
            gts   = func_read_key_from_csv(gt_csv, 'openset')
            for (name, gt) in zip(names, gts):
                name2gt[name] = gt

    # read name2pred
    if name2pred is None:
        if openset_root is not None:
            name2pred = func_get_name2reason(openset_root)
        elif openset_npz is not None:
            names = np.load(openset_npz)['filenames']
            items = np.load(openset_npz)['fileitems']
            name2pred = {}
            for (name, item) in zip(names, items):
                name2pred[name] = item

    # process_names => (whole) / (subset)
    if process_names is None:
        process_names = [name for name in name2gt]

    # read all mapping 
    if format_mapping is None:
        format_mapping = read_format2raws()          # level3 -> level2
    if raw_mapping is None:
        raw_mapping = read_candidate_synonym_merge() # level2 -> level1
    if metric.startswith('case3'): # level1 -> cluster center
        _, wheelname, levelname = metric.split('_')
        wheel_map = func_get_wheel_cluster(wheelname, levelname)
    else:
        wheel_map = None

    # calculate (accuracy, recall) two values
    accuracy, recall = [], []
    for name in process_names:      
       
        # 删除 gt and pred 中的同义词
        gt = string_to_list(name2gt[name])
        gt = [item.lower().strip() for item in gt]
        gt = set(func_map_label_to_synonym(gt, format_mapping, raw_mapping, wheel_map, metric))

        pred = string_to_list(name2pred[name])
        pred = [item.lower().strip() for item in pred]
        pred = set(func_map_label_to_synonym(pred, format_mapping, raw_mapping, wheel_map, metric))

        if len(gt) == 0: continue
        if len(pred) == 0:
            accuracy.append(0)
            recall.append(0)
        else:
            accuracy.append(len(gt & pred)/len(pred))
            recall.append(len(gt & pred)/len(gt))
    if inter_print: print ('process number (after filter): ', len(accuracy))
    avg_accuracy, avg_recall = np.mean(accuracy), np.mean(recall)
    if inter_print: print (f'avg acc: {avg_accuracy} avg recall: {avg_recall}')
    return  avg_accuracy, avg_recall

def calculate_openset_onehot_hitrate(name2gt=None, 
                                     openset_root=None, openset_npz=None, name2pred=None, 
                                     metric='case1', format_mapping=None, raw_mapping=None,
                                     inter_print=True):

    # read name2gt
    candidata_labels = list(set([name2gt[name] for name in name2gt]))
    if inter_print: print (f'candidata_labels: {candidata_labels}')
    
    # read name2pred
    if name2pred is None:
        if openset_root is not None:
            name2pred = func_get_name2reason(openset_root)
        elif openset_npz is not None:
            names = np.load(openset_npz)['filenames']
            items = np.load(openset_npz)['fileitems']
            name2pred = {}
            for (name, item) in zip(names, items):
                name2pred[name] = item

    # read all mapping 
    if format_mapping is None:
        format_mapping = read_format2raws()          # level3 -> level2
    if raw_mapping is None:
        raw_mapping = read_candidate_synonym_merge() # level2 -> level1
    if metric.startswith('case3'): # level1 -> cluster center
        _, wheelname, levelname = metric.split('_')
        wheel_map = func_get_wheel_cluster(wheelname, levelname)
    else:
        wheel_map = None

    # score calculation
    hitrates, mscores = [], []
    for name in name2gt:
       
        # (gt) process
        gt = string_to_list(name2gt[name])
        gt = [item.lower().strip() for item in gt]
        gt = list(set(func_map_label_to_synonym(gt, format_mapping, raw_mapping, wheel_map, metric)))
        if len(gt) == 0: continue # we only calculate non-neutral label hit rate

        # (pred) process
        pred = string_to_list(name2pred[name])
        pred = [item.lower().strip() for item in pred]
        pred = list(set(func_map_label_to_synonym(pred, format_mapping, raw_mapping, wheel_map, metric)))

        # (candidata_labels) process
        candidates = list(set(func_map_label_to_synonym(candidata_labels, format_mapping, raw_mapping, wheel_map, metric)))

        # metric calculation
        hitrates.append(len(set(pred) & set(gt)))
        if len(set(pred) & set(candidates)) == 0:
            mscores.append(0)
        else:
            score = len(set(pred) & set(gt)) / len(set(pred) & set(candidates))
            mscores.append(score)
    if inter_print: print (f'after filter sample number: {len(hitrates)}')
    avg_hitrate, avg_mscore = np.mean(hitrates), np.mean(mscores)
    if inter_print: print (f'avg_hitrate: {avg_hitrate}; avg_mscore: {avg_mscore}')
    return avg_hitrate, avg_mscore

def string_to_list(str):

    if type(str) == np.ndarray:
        str = str.tolist()

    # 如果本身就是list了，那么就不需要其他操作了
    if isinstance(str, list):
        return str
    
    if str == '':
        str = []
    elif pd.isna(str):
        str = []
    else:
        if str[0] == '[':  str = str[1:]
        if str[-1] == ']': str = str[:-1]
        str = [item.strip() for item in re.split('[\'\",]', str) if item.strip() not in ['', ',']]
    return str

def func_read_key_from_csv(csv_path, key):
    values = []
    df = pd.read_csv(csv_path)
    # for _, row in df.iterrows():
    for _, row in df.iterrows():
        if key not in row:
            values.append("")
        else:
            value = row[key]
            if pd.isna(value): value=""
            values.append(value)
    return values

def func_postprocess_qwen(response):
    response = response.strip()
    if response.startswith("输入"):   response = response[len("输入"):]
    if response.startswith("输出"):   response = response[len("输出"):]
    if response.startswith("翻译"):   response = response[len("翻译"):]
    if response.startswith("让我们来翻译一下："): response = response[len("让我们来翻译一下："):]
    if response.startswith("output"): response = response[len("output"):]
    if response.startswith("Output"): response = response[len("Output"):]
    if response.startswith("input"): response = response[len("input"):]
    if response.startswith("Input"): response = response[len("Input"):]
    response = response.strip()
    if response.startswith(":"):  response = response[len(":"):]
    if response.startswith("："): response = response[len("："):]
    response = response.strip()
    response = response.replace('\n', '') # remove \n
    response = response.strip()
    return response

def get_completion_qwen(model, tokenizer, prompt):
    
    assert isinstance(prompt, str)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = func_postprocess_qwen(response)
    print(f"Prompt: {prompt} \n Response: {response}")
    return response

def get_completion_qwen_bacth(llm, sampling_params, tokenizer, prompt_list):
    
    assert isinstance(prompt_list, list)

    message_batch = []
    for prompt in prompt_list:
        message_batch.append([{"role": "user", "content": prompt}])

    text_batch = tokenizer.apply_chat_template(
        message_batch,
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = llm.generate(text_batch, sampling_params)
    
    # => batch_responses
    batch_responses = []
    for output in outputs:
        prompt = output.prompt
        response = output.outputs[0].text
        response = func_postprocess_qwen(response)
        batch_responses.append(response)
        print(f"Prompt: {prompt} \n Response: {response}")
    return batch_responses

def reason_to_openset_qwen(model=None, tokenizer=None, llm=None, sampling_params=None, reason=None, batch_reasons=None):
    
    def func_prompt_template(reason):
        prompt = f"""Please assume the role of an expert in the field of emotions. \
We provide clues that may be related to the emotions of the characters. Based on the provided clues, please identify the emotional states of the main character. \
The main character is the one with the most detailed clues. \
Please separate different emotional categories with commas and output only the clearly identifiable emotional categories in a list format. \
If none are identified, please output an empty list. \
Input: We cannot recognize his emotional state; Output: [] \
Input: His emotional state is happy, sad, and angry; Output: [happy, sad, angry] \
Input: {reason}; Output: """
        return prompt

    ## process for reason
    if reason is not None:
        prompt = func_prompt_template(reason)
        response = get_completion_qwen(model, tokenizer, prompt)
        return response

    ## process for reason_list
    if batch_reasons is not None:
        prompt_list = []
        for reason in batch_reasons:
            prompt = func_prompt_template(reason)
            prompt_list.append(prompt)
        response_list = get_completion_qwen_bacth(llm, sampling_params, tokenizer, prompt_list)
        return response_list

def openset_to_sentiment_qwen(model=None, tokenizer=None, llm=None, sampling_params=None, reason=None, batch_reasons=None):
    def func_prompt_template(reason):
        prompt = f"""Please act as an expert in the field of emotions. \
            We provide a few words to describe the emotions of a character. \
            Please choose the most likely sentiment from the given candidates: [positive, negative, neutral] \
            Please direct output answer without analyzing process. \
            Input: [joyful]; Output: positive \
            Input: []; Output: neutral \
            Input: {reason}; Output: """
        return prompt

    ## process for reason
    if reason is not None:
        prompt = func_prompt_template(reason)
        response = get_completion_qwen(model, tokenizer, prompt)
        return response

    ## process for reason_list
    if batch_reasons is not None:
        prompt_list = []
        for reason in batch_reasons:
            prompt = func_prompt_template(reason)
            prompt_list.append(prompt)
        response_list = get_completion_qwen_bacth(llm, sampling_params, tokenizer, prompt_list)
        return response_list
