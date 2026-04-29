import argparse #用于解析命令行参数，定义API密钥，配置文件路径
import glob  #用于查找符合特定模式的文件路径，遍历pddl，动态加载测试数据
import json   #用于处理json格式数据，加载保存配置文件，解析API返回的json数据
import os  #提供与操作系统交互的功能，操作文件和目录（如创建文件夹、检查路径是否存在）。
import random #用于生成随机数
import sys  #提供与python解释器交互的功能，处理命令行参数（如 sys.argv）。终止程序（如 sys.exit()）。
import time  #用于时间相关的操作，记录规划求解时间，添加延迟
import backoff # 用于实现指数退避策略，处理 API 调用失败的情况。

import openai #OpenAI 提供的官方 Python SDK，用于调用其 API（如 GPT-4）。
from openai import OpenAIError


#指定搜索算法，lama，即Landmark-based heuristic （基于关键点的启发式）
#           seq-opt-fdss-1Sequence Optimization （序列优化）策略，
#           结合特定的启发式函数（如 ff 或 fdss）来搜索最优计划。
#           lama速度更快，但并不保证找到最优解，seq-opt更耗时，但结果更优
FAST_DOWNWARD_ALIAS = "lama"
# FAST_DOWNWARD_ALIAS = "astar(blind())"
# FAST_DOWNWARD_ALIAS = "seq-opt-fdss-1"


#  strip移除字符串开头和结尾的所有空白字符,后处理，
def postprocess(x):
    return x.strip()


# 获取cost函数
def get_cost(x):
    splitted = x.split() #提取每个单词，获得一个列表
    counter = 0
    found = False
    cost = 1e5
    for i, xx in enumerate(splitted): #i为xx的索引，遍历列表splitted寻找cost
        if xx == "cost":
            counter = i
            found = True
            break
    if found:
        cost = float(splitted[counter+2])   #有解决方案即有cost，无就是一个很大的值
    return cost


###############################################################################
#
# Define different problem domains
#
###############################################################################

DOMAINS = [
    "barman", #调酒师
    "blocksworld", #积木世界
    "floortile",   #地板瓷砖
    "grippers",   #机械手
    "storage",   #仓储
    "termes",   #建筑
    "tyreworld",  #更换轮胎
    "manipulation" #操作问题
]


class Domain:
    def __init__(self):

        self.context = ("p_example.nl", "p_example.pddl", "p_example.sol")
        self.tasks = [] # should be list of tuples like (descritpion, ground_truth_pddl)
                        #（每个任务是一个自然语言文件和对应的 PDDL 文件的元组）。
        self.grab_tasks() #获取任务的方法，新建一个类自动获取
        # every domain should contain the context as in "in-context learning" (ICL)
        # which are the example problem in natural language.
        # For instance, in our case, context is:
        # 1. p_example.nl  (a language description of the problem)
        # 2. p_example.pddl (the ground-truth problem pddl for the problem)
        # 3. p_example.sol  (the ground-truth solution in natural language to the problem)
#从 ./domains/{domain_name} 目录中自动收集所有任务文件。
    def grab_tasks(self):
        path = f"./domains/{self.name}"  #对应领域文件夹路径
        nls = []
        for fn in glob.glob(f"{path}\*.nl"): #使用glob模块查找path目录下以.nl结尾的文件（自然语言描述文件）路径
            fn_ = fn.split("\\")[-1]          #将fn字符串按/分割成一个列表，并负索引取得文件名
            if "domain" not in fn_ and "p_example" not in fn_:  #排除领域文件（domain）和示例文件（example）
                if os.path.exists(fn.replace("nl", "pddl")): #确保.nl文件有对应的pddl文件
                    nls.append(fn_)                       #若有，将.nl文件增加到自然语言描述的列表
        sorted_nls = sorted(nls)            #对文件进行排序
        self.tasks = [(nl, nl.replace("nl", "pddl")) for nl in sorted_nls]
        #生成包含任务元组的列表，.nl 文件名与对应的 .pddl 文件名配对，

    def __len__(self):
        return len(self.tasks)   #返回任务的数量

    def get_task_suffix(self, i):  #生成一个唯一标识符 ，用于表示某个具体任务的来源和文件名。
        nl, pddl = self.tasks[i]
        return f"{self.name}/{pddl}"

    def get_task_file(self, i):     #获得.nl文件和.pddl文件的路径
        nl, pddl = self.tasks[i]
        print(nl,pddl)
        return f"./domains/{self.name}/{nl}", f"./domains/{self.name}/{pddl}"

    def get_task(self, i):      #读取nl文件和pddl文件，并用postprocess清理空白字符
        nl_f, pddl_f = self.get_task_file(i)
        with open(nl_f, 'r') as f:
            nl = f.read()
        with open(pddl_f, 'r') as f:
            pddl = f.read()
        return postprocess(nl), postprocess(pddl)

    def get_context(self):      #获取示例的自然语言描述和对应的PDDL以及解决方案，并打开，postprocess清理空白字符
        nl_f   = f"./domains/{self.name}/{self.context[0]}"
        pddl_f = f"./domains/{self.name}/{self.context[1]}"
        sol_f  = f"./domains/{self.name}/{self.context[2]}"
        with open(nl_f, 'r') as f:
            nl   = f.read()
        with open(pddl_f, 'r') as f:
            pddl = f.read()
        with open(sol_f, 'r') as f:
            sol  = f.read()
        return postprocess(nl), postprocess(pddl), postprocess(sol)

    def get_domain_pddl(self):   #打开领域描述文件，并postprocess清除空白字符
        domain_pddl_f = self.get_domain_pddl_file()
        with open(domain_pddl_f, 'r') as f:
            domain_pddl = f.read()
        return postprocess(domain_pddl)

    def get_domain_pddl_file(self):   #获取领域文件路径
        domain_pddl_f = f"./domains/{self.name}/domain.pddl"
        return domain_pddl_f

    def get_domain_nl(self):     #打开领域文件，并postprocess清楚空白字符返回
        domain_nl_f = self.get_domain_nl_file()
        try:
            with open(domain_nl_f, 'r') as f:
                domain_nl = f.read()
        except:
            domain_nl = "Nothing"
        return postprocess(domain_nl)

    def get_domain_nl_file(self):   #获取领域文件的自然语言描述
        domain_nl_f = f"./domains/{self.name}/domain.nl"
        return domain_nl_f


class Barman(Domain):
    name = "barman" # this should match the directory name

class Floortile(Domain):
    name = "floortile" # this should match the directory name

class Termes(Domain):
    name = "termes" # this should match the directory name

class Tyreworld(Domain):
    name = "tyreworld" # this should match the directory name

class Grippers(Domain):
    name = "grippers" # this should match the directory name

class Storage(Domain):
    name = "storage" # this should match the directory name

class Blocksworld(Domain):
    name = "blocksworld" # this should match the directory name

class Manipulation(Domain):
    name = "manipulation" # this should match the directory name

###############################################################################
#
# The agent that leverages classical planner to help LLMs to plan
#
###############################################################################


class Planner:
    def __init__(self):
        self.openai_api_keys = self.load_openai_keys()  #加载openai api密钥
        self.use_chatgpt = True                         #设置是否使用chatgpt

    def load_openai_keys(self,):    #从文件中加载api密钥
        openai_keys_file = os.path.join(os.getcwd(), "keys/openai_keys.txt")
        with open(openai_keys_file, "r") as f:
            context = f.read()
        context_lines = context.strip().split('\n')
        print(context_lines)
        return context_lines

    def create_llm_prompt(self, task_nl, domain_nl):    #创造直接生成计划的prompt提示词
        # Baseline 1 (LLM-as-P): directly ask the LLM for plan
        prompt = f"{domain_nl} \n" + \
                 f"Now consider a planning problem. " + \
                 f"The problem description is: \n {task_nl} \n" + \
                 f"Can you provide an optimal plan, in the way of a " + \
                 f"sequence of behaviors, to solve the problem?"
        return prompt

    def create_llm_stepbystep_prompt(self, task_nl, domain_nl):   #step by step 生成计划的prompt
        # Baseline 1 (LLM-as-P): directly ask the LLM for plan
        prompt = f"{domain_nl} \n" + \
                 f"Now consider a planning problem. " + \
                 f"The problem description is: \n {task_nl} \n" + \
                 f"Can you provide an optimal plan, in the way of a " + \
                 f"sequence of behaviors, to solve the problem? \n" + \
                 f"Please think step by step."
        return prompt

    def create_llm_tot_ic_prompt(self, task_nl, domain_nl, context, plan):  #TOT，上下文学习的LLM-AS-PLANNER
        context_nl, context_pddl, context_sol = context
        prompt = f"Given the current state, provide the set of feasible actions and their corresponding next states, using the format 'action -> state'. \n" + \
                 f"Keep the list short. Think carefully about the requirements of the actions you select and make sure they are met in the current state. \n" + \
                 f"Start with actions that are most likely to make progress towards the goal. \n" + \
                 f"Only output one action per line. Do not return anything else. " + \
                 f"Here are the rules. \n {domain_nl} \n\n" + \
                 f"An example planning problem is: \n {context_nl} \n" + \
                 f"A plan for the example problem is: \n {context_sol} \n" + \
                 f"Now I have a new planning problem and its description is: \n {task_nl} \n" + \
                 f"You have taken the following actions: \n {plan} \n"
        # print(prompt)
        return prompt

    def create_llm_tot_ic_value_prompt(self, task_nl, domain_nl, context, plan):  #tot上下文学习的评估prompt
        context_nl, context_pddl, context_sol = context
        context_sure_1 = context_sol.split('\n')[0]                     #提取解决方案的第一行（第一个动作）
        context_sure_2 = context_sol.split('\n')[0] + context_sol.split('\n')[1]  #提取解决方案的前两行（前两个动作）
        context_impossible_1 = '\n'.join(context_sol.split('\n')[1:])   #提取解决方案第二行及之后的所有行
        context_impossible_2 = context_sol.split('\n')[-1]             #提取解决方案的最后一行
        '''
        prompt = f"Evaluate if a given plan reaches the goal or is an optimal partial plan towards the goal (reached/sure/maybe/impossible). \n" + \
                 f"Only answer 'reached' if the goal conditions are reached by the exact plan in the prompt. \n" + \
                 f"Only answer 'sure' if you are sure that preconditions are satisfied for all actions in the plan, and the plan makes fast progress towards the goal. \n" + \
                 f"Answer 'impossible' if one of the actions has unmet preconditions. \n" + \
                 f"Here are the rules. \n {domain_nl} \n\n" + \
                 f"Here are some example evaluations for the planning problem: \n {context_nl} \n\n " + \
                 f"Plan: {context_sure_1} \n" + \
                 f"Answer: Sure. \n\n" + \
                 f"Plan: {context_sure_2} \n" + \
                 f"Answer: Sure. \n\n" + \
                 f"Plan: {context_sol} \n" + \
                 f"Answer: Reached. \n\n" + \
                 f"Plan: {context_impossible_1} \n" + \
                 f"Answer: Impossible. \n\n" + \
                 f"Plan: {context_impossible_2} \n" + \
                 f"Answer: Impossible. \n\n" + \
                 f"Now I have a new planning problem and its description is: \n {task_nl} \n" + \
                 f"Evaluate the following partial plan as reached/sure/maybe/impossible. DO NOT RETURN ANYTHING ELSE. DO NOT TRY TO COMPLETE THE PLAN. \n" + \
                 f"Plan: {plan} \n"
        '''
        prompt = f"Determine if a given plan reaches the goal or give your confidence score that it is an optimal partial plan towards the goal (reached/impossible/0-1). \n" + \
                 f"Only answer 'reached' if the goal conditions are reached by the exact plan in the prompt. \n" + \
                 f"Answer 'impossible' if one of the actions has unmet preconditions. \n" + \
                 f"Otherwise,give a number between 0 and 1 as your evaluation of the partial plan's progress towards the goal. \n" + \
                 f"Here are the rules. \n {domain_nl} \n\n" + \
                 f"Here are some example evaluations for the planning problem: \n {context_nl} \n\n " + \
                 f"Plan: {context_sure_1} \n" + \
                 f"Answer: 0.8. \n\n" + \
                 f"Plan: {context_sure_2} \n" + \
                 f"Answer: 0.9. \n\n" + \
                 f"Plan: {context_sol} \n" + \
                 f"Answer: Reached. \n\n" + \
                 f"Plan: {context_impossible_1} \n" + \
                 f"Answer: Impossible. \n\n" + \
                 f"Plan: {context_impossible_2} \n" + \
                 f"Answer: Impossible. \n\n" + \
                 f"Now I have a new planning problem and its description is: \n {task_nl} \n" + \
                 f"Evaluate the following partial plan as reached/impossible/0-1. DO NOT RETURN ANYTHING ELSE. DO NOT TRY TO COMPLETE THE PLAN. \n" + \
                 f"Plan: {plan} \n"

        return prompt


    def tot_bfs(self, task_nl, domain_nl, context, time_left=200, max_depth=2):  #使用广度优先搜索（BFS）结合 LLM 逐步扩展计划。
        from queue import PriorityQueue
        start_time = time.time()
        plan_queue = PriorityQueue()
        plan_queue.put((0, ""))
        while time.time() - start_time < time_left and not plan_queue.empty():
            priority, plan = plan_queue.get()
            # print (priority, plan)
            steps = plan.split('\n')
            if len(steps) > max_depth:
                return ""
            candidates_prompt = self.create_llm_tot_ic_prompt(task_nl, domain_nl, context, plan)
            candidates = self.query(candidates_prompt).strip()
            print (candidates)
            lines = candidates.split('\n')
            for line in lines:
                if time.time() - start_time > time_left:
                    break
                if len(line) > 0 and '->' in line:
                    new_plan = plan + "\n" + line
                    value_prompt = self.create_llm_tot_ic_value_prompt(task_nl, domain_nl, context, new_plan)
                    answer = self.query(value_prompt).strip().lower()
                    print(new_plan)
                    print("Response \n" + answer)

                    if "reached" in answer:
                        return new_plan

                    if "impossible" in answer:
                        continue

                    if "answer: " in answer:
                        answer = answer.split("answer: ")[1]

                    try:
                        score = float(answer)
                    except ValueError:
                        continue

                    if score > 0:
                        new_priority = priority + 1 / score
                        plan_queue.put((new_priority, new_plan))

        return ""

    def create_llm_ic_prompt(self, task_nl, domain_nl, context):   #上下文学习的LLM-AS-PLANNER
        # Baseline 2 (LLM-as-P with context): directly ask the LLM for plan
        context_nl, context_pddl, context_sol = context
        prompt = f"{domain_nl} \n" + \
                 f"An example planning problem is: \n {context_nl} \n" + \
                 f"A plan for the example problem is: \n {context_sol} \n" + \
                 f"Now I have a new planning problem and its description is: \n {task_nl} \n" + \
                 f"Can you provide an optimal plan, in the way of a " + \
                 f"sequence of behaviors, to solve the problem?"
        return prompt

    def create_llm_pddl_prompt(self, task_nl, domain_nl):    #LLM+P,无上下文学习
        # Baseline 3 (LM+P w/o context), no context, create the problem PDDL
        prompt = f"{domain_nl} \n" + \
                 f"Now consider a planning problem. " + \
                 f"The problem description is: \n {task_nl} \n" + \
                 f"Provide me with the problem PDDL file that describes " + \
                 f"the planning problem directly without further explanations?" +\
                 f"Keep the domain name consistent in the problem PDDL. Only return the PDDL file. Do not return anything else."
        return prompt

    def create_llm_ic_pddl_prompt(self, task_nl, domain_pddl, context): #LLM+P,有上下文学习
        # our method (LM+P), create the problem PDDL given the context
        context_nl, context_pddl, context_sol = context
        prompt = f"I want you to solve planning problems. " + \
                 f"An example planning problem is: \n {context_nl} \n" + \
                 f"The problem PDDL file to this problem is: \n {context_pddl} \n" + \
                 f"Now I have a new planning problem and its description is: \n {task_nl} \n" + \
                 f"Provide me with the problem PDDL file that describes " + \
                 f"the new planning problem directly without further explanations."+ \
                 f"Please check the corresponding relationship of the parentheses carefully and do not omit any of them.Only return the PDDL file. Do not return anything else，such Punctuation marks‘’‘and lisp.Only return the PDDL file. Do not return anything else，Only return the PDDL file. Do not return anything else，"

        prompt = f"我想要你解决一个机器人领域的任务规划问题. " + \
                 f"以下有一个示例的规划问题的自然语言描述: \n {context_nl} \n" + \
                 f"这个规划问题对应的PDDL（规划领域定义语言）文件如下: \n {context_pddl} \n" + \
                 f"现在我有一个新的规划问题，我给你提供该问题的自然语言描述如下: \n {task_nl} \n" + \
                 f"请给我提供该问题描述对应的PDDL形式（参考上述示例）" + \
                 f"请直接解决新的规划问题而不作进一步解释。只返回PDDL文件。不要返回任何其他东西"
        return prompt

    def query(self, prompt_text):  # 向 LLM 发送提示并获取响应。
        base_url = "https://api.deepseek.com"
        openai.base_url = base_url
        server_flag = 0
        server_cnt = 0
        result_text = ""
        while server_cnt < 10:
            try:
                self.update_key()
                if self.use_chatgpt:  # currently, we will always use chatgpt
                    @backoff.on_exception(backoff.expo, openai.RateLimitError)
                    def completions_with_backoff(**kwargs):
                        client = openai.OpenAI(api_key="",
                                               base_url="https://api.deepseek.com")
                        return client.chat.completions.create(**kwargs)
                        # return openai.ChatCompletion.create(**kwargs)

                    # response = openai.ChatCompletion.create(
                    response = completions_with_backoff(
                        model="deepseek-reasoner",
                        temperature=0.1,
                        top_p=0.5,
                        max_tokens=1000,
                        frequency_penalty=0,
                        presence_penalty=0,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt_text},
                        ],
                    )
                    result_text = response.choices[0].message.content
                    # result_text = response['choices'][0]['message']['content']
                else:
                    response = openai.Completion.create(
                        model="text-davinci-003",
                        prompt=prompt_text,
                        temperature=0.0,
                        max_tokens=1024,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    result_text = response['choices'][0]['text']
                server_flag = 1
                if server_flag:
                    break
            except Exception as e:
                server_cnt += 1
                print(e)
        return result_text

    def update_key(self):   #更新api密钥
        curr_key = self.openai_api_keys[0]
        openai.api_key = curr_key
        self.openai_api_keys.remove(curr_key)
        self.openai_api_keys.append(curr_key)

    def parse_result(self, pddl_string):   #解析结果？
        # remove extra texts
        #try:
        #    beg = pddl_string.find("```") + 3
        #    pddl_string = pddl_string[beg: beg + pddl_string[beg:].find("```")]
        #except:
        #    raise Exception("[error] cannot find ```pddl-file``` in the pddl string")

        # remove comments, they can cause error
        #t0 = time.time()
        #while pddl_string.find(";") >= 0:
        #    start = pddl_string.find(";")
        #    i = 0
        #    while pddl_string[start+i] != ")" and pddl_string[start+i] != "\n":
        #        i += 1
        #    pddl_string = pddl_string[:start] + pddl_string[start+i:]
        #pddl_string = pddl_string.strip() + "\n"
        #t1 = time.time()
        #print(f"[info] remove comments takes {t1-t0} sec")
        return pddl_string

    def plan_to_language(self, plan, task_nl, domain_nl, domain_pddl):   #将pddl计划翻译回自然语言
        domain_pddl_ = " ".join(domain_pddl.split())
        task_nl_ = " ".join(task_nl.split())
        prompt = f"A planning problem is described as: \n {task_nl} \n" + \
                 f"The corresponding domain PDDL file is: \n {domain_pddl_} \n" + \
                 f"The optimal PDDL plan is: \n {plan} \n" + \
                 f"Transform the PDDL plan into a sequence of behaviors without further explanation."
        res = self.query(prompt).strip() + "\n"
        return res


def llm_ic_pddl_planner(args, planner, domain):
    """
    Our method:
        context: (task natural language, task problem PDDL)
        Condition on the context (task description -> task problem PDDL),
        LLM will be asked to provide the problem PDDL of a new task description.
        Then, we use a planner to find the near optimal solution, and translate
        that back to natural language.
    """
    context          = domain.get_context()  #获取上下文学习的示例文件
    domain_pddl      = domain.get_domain_pddl() #获取领域描述文件
    domain_pddl_file = domain.get_domain_pddl_file()  #获取领域描述文件路径
    domain_nl        = domain.get_domain_nl()   #获取自然语言描述的领域描述文件
    domain_nl_file   = domain.get_domain_nl_file()  #获取自然语言描述的领域描述文件的路径

    # create the tmp / result folders
    #problem_folder 存储生成的pddl问题文件
    #plan_folder 存储经典规划器生成的计划文件
    #result_folder 存储最终翻译回自然语言的结果文件
    problem_folder = f"./experiments/run{args.run}/problems/llm_ic_pddl/{domain.name}"
    plan_folder    = f"./experiments/run{args.run}/plans/llm_ic_pddl/{domain.name}"
    result_folder  = f"./experiments/run{args.run}/results/llm_ic_pddl/{domain.name}"
    #检查文件夹是否存在，不存在就创建
    if not os.path.exists(problem_folder):
        os.system(f"mkdir -p {problem_folder}")
    if not os.path.exists(plan_folder):
        os.system(f"mkdir -p {plan_folder}")
    if not os.path.exists(result_folder):
        os.system(f"mkdir -p {result_folder}")

    #从args中提取任务编号（参数）
    task = args.task

    #记录开始时间
    start_time = time.time()

    # A. generate problem pddl file
    #task_suffix获取当前任务的suffix
    #task_nl, task_pddl 获取任务的自然语言描述和标准 PDDL 文件
    #prompt，构造提示词
    #raw_result 查询 LLM 并获取原始结果
    #task_pddl 对raw_result进行处理，去除多余的内容
    task_suffix        = domain.get_task_suffix(task)
    task_nl, task_pddl = domain.get_task(task) 
    prompt             = planner.create_llm_ic_pddl_prompt(task_nl, domain_pddl, context)
    raw_result         = planner.query(prompt)
    task_pddl_         = planner.parse_result(raw_result)

    # B. write the problem file into the problem folder
    #task_pddl_file_name定义文件路径
    #with open保存生成的问题文件
    #添加延迟，确保能完成写入操作
    task_pddl_file_name = f"./experiments/run{args.run}/problems/llm_ic_pddl/{task_suffix}"
    with open(task_pddl_file_name, "w") as f:
        f.write(task_pddl_)
    time.sleep(1)

    ## C. run fastforward to plan
    #plan_file_name 构造计划文件的路径
    #sas_file_name 定义中间文件的路径（sas是fast downward规划器生成的中间文件，包含状态空间和搜索信息）
    #os.system(command)：执行系统命令。
    # 命令的具体参数如下：
    # python ./downward/fast-downward.py ：
    # 调用 Fast Downward 的主脚本。
    # ./downward/ 是 Fast Downward 的安装目录。
    # --alias {FAST_DOWNWARD_ALIAS} ：
    # 指定规划器的搜索策略。
    # FAST_DOWNWARD_ALIAS 是一个变量，表示预定义的搜索策略别名。例如：
    # seq-opt-fdss-1：保证找到最优计划。
    # lama：不保证最优，但速度更快。
    # --search-time-limit {args.time_limit} ：
    # 设置最大搜索时间（以秒为单位）。
    # args.time_limit 是用户指定的时间限制。
    # --plan-file {plan_file_name} ：
    # 指定输出计划文件的路径。
    # 规划器会将生成的计划保存到该路径。
    # --sas-file {sas_file_name} ：
    # 指定中间文件（SAS 文件）的路径。
    # 该文件记录了规划器在搜索过程中的状态空间和转换信息。
    # {domain_pddl_file} {task_pddl_file_name} ：
    # 输入文件：
    # domain_pddl_file：领域的 PDDL 文件，定义动作的前提条件和效果。
    # task_pddl_file_name：问题的 PDDL 文件，定义初始状态和目标条件。
    # --search-time-limit {args.time_limit}
    plan_file_name = f"./experiments/run{args.run}/plans/llm_ic_pddl/{task_suffix}"
    sas_file_name  = f"./experiments/run{args.run}/plans/llm_ic_pddl/{task_suffix}.sas"
    os.system(f"python ./downward/fast-downward.py --alias {FAST_DOWNWARD_ALIAS} " + \
              f"--plan-file {plan_file_name} " + \
              f"--sas-file {sas_file_name} " + \
              f"{domain_pddl_file} {task_pddl_file_name}")

    # D. collect the least cost plan
    best_cost = 1e10
    best_plan = None
    for fn in glob.glob(f"{plan_file_name}.*"):
        with open(fn, "r") as f:
            plans = f.readlines()
            cost = get_cost(plans[-1])
            if cost < best_cost:
                best_cost = cost
                best_plan = "\n".join([p.strip() for p in plans[:-1]])

    # E. translate the plan back to natural language, and write it to result
    # commented out due to exceeding token limit of gpt-4
            '''
    if best_plan:
        plans_nl = planner.plan_to_language(best_plan, task_nl, domain_nl, domain_pddl)
        plan_nl_file_name = f"./experiments/run{args.run}/results/llm_ic_pddl/{task_suffix}"
        with open(plan_nl_file_name, "w") as f:
            f.write(plans_nl)
            '''

    end_time = time.time()
    if best_plan:
        print(f"[info] task {task} takes {end_time - start_time} sec, found a plan with cost {best_cost}")
    else:
        print(f"[info] task {task} takes {end_time - start_time} sec, no solution found")


def llm_pddl_planner(args, planner, domain):
    """
    Baseline method:
        Same as ours, except that no context is given. In other words, the LLM
        will be asked to directly give a problem PDDL file without any context.
    """
    context          = domain.get_context()
    domain_pddl      = domain.get_domain_pddl()
    domain_pddl_file = domain.get_domain_pddl_file()
    domain_nl        = domain.get_domain_nl()
    domain_nl_file   = domain.get_domain_nl_file()

    # create the tmp / result folders
    problem_folder = f"./experiments/run{args.run}/problems/llm_pddl/{domain.name}"
    plan_folder    = f"./experiments/run{args.run}/plans/llm_pddl/{domain.name}"
    result_folder  = f"./experiments/run{args.run}/results/llm_pddl/{domain.name}"

    if not os.path.exists(problem_folder):
        os.system(f"mkdir -p {problem_folder}")
    if not os.path.exists(plan_folder):
        os.system(f"mkdir -p {plan_folder}")
    if not os.path.exists(result_folder):
        os.system(f"mkdir -p {result_folder}")

    task = args.task

    start_time = time.time()

    # A. generate problem pddl file
    task_suffix        = domain.get_task_suffix(task)
    task_nl, task_pddl = domain.get_task(task) 
    prompt             = planner.create_llm_pddl_prompt(task_nl, domain_nl)
    raw_result         = planner.query(prompt)
    task_pddl_         = planner.parse_result(raw_result)

    # B. write the problem file into the problem folder
    task_pddl_file_name = f"./experiments/run{args.run}/problems/llm_pddl/{task_suffix}"
    with open(task_pddl_file_name, "w") as f:
        f.write(task_pddl_)
    time.sleep(1)

    # C. run fastforward to plan
    plan_file_name = f"./experiments/run{args.run}/plans/llm_pddl/{task_suffix}"
    sas_file_name  = f"./experiments/run{args.run}/plans/llm_pddl/{task_suffix}.sas"
    os.system(f"python ./downward/fast-downward.py --alias {FAST_DOWNWARD_ALIAS} " + \
              f"--search-time-limit {args.time_limit} --plan-file {plan_file_name} " + \
              f"--sas-file {sas_file_name} " + \
              f"{domain_pddl_file} {task_pddl_file_name}")

    # D. collect the least cost plan
    best_cost = 1e10
    best_plan = None
    for fn in glob.glob(f"{plan_file_name}.*"):
        with open(fn, "r") as f:
            try:
                plans = f.readlines()
                cost = get_cost(plans[-1])
                if cost < best_cost:
                    best_cost = cost
                    best_plan = "\n".join([p.strip() for p in plans[:-1]])
            except:
                continue

    # E. translate the plan back to natural language, and write it to result
    # commented out due to exceeding token limit of gpt-4
    '''
    if best_plan:
        plans_nl = planner.plan_to_language(best_plan, task_nl, domain_nl, domain_pddl)
        plan_nl_file_name = f"./experiments/run{args.run}/results/llm_pddl/{task_suffix}"
        with open(plan_nl_file_name, "w") as f:
            f.write(plans_nl)
    '''
    end_time = time.time()
    if best_plan:
        print(f"[info] task {task} takes {end_time - start_time} sec, found a plan with cost {best_cost}")
    else:
        print(f"[info] task {task} takes {end_time - start_time} sec, no solution found")


def llm_planner(args, planner, domain):
    """
    Baseline method:
        The LLM will be asked to directly give a plan based on the task description.
    """
    context          = domain.get_context()
    domain_pddl      = domain.get_domain_pddl()
    domain_pddl_file = domain.get_domain_pddl_file()
    domain_nl        = domain.get_domain_nl()
    domain_nl_file   = domain.get_domain_nl_file()

    # create the tmp / result folders
    problem_folder = f"./experiments/run{args.run}/problems/llm/{domain.name}"
    plan_folder    = f"./experiments/run{args.run}/plans/llm/{domain.name}"
    result_folder  = f"./experiments/run{args.run}/results/llm/{domain.name}"

    if not os.path.exists(problem_folder):
        os.system(f"mkdir -p {problem_folder}")
    if not os.path.exists(plan_folder):
        os.system(f"mkdir -p {plan_folder}")
    if not os.path.exists(result_folder):
        os.system(f"mkdir -p {result_folder}")

    task = args.task

    start_time = time.time()

    # A. generate problem pddl file
    task_suffix        = domain.get_task_suffix(task)
    task_nl, task_pddl = domain.get_task(task) 
    prompt             = planner.create_llm_prompt(task_nl, domain_nl)
    text_plan          = planner.query(prompt)

    # B. write the problem file into the problem folder
    text_plan_file_name = f"./experiments/run{args.run}/results/llm/{task_suffix}"
    with open(text_plan_file_name, "w") as f:
        f.write(text_plan)
    end_time = time.time()
    print(f"[info] task {task} takes {end_time - start_time} sec")


def llm_stepbystep_planner(args, planner, domain):
    """
    Baseline method:
        The LLM will be asked to directly give a plan based on the task description.
    """
    context          = domain.get_context()
    domain_pddl      = domain.get_domain_pddl()
    domain_pddl_file = domain.get_domain_pddl_file()
    domain_nl        = domain.get_domain_nl()
    domain_nl_file   = domain.get_domain_nl_file()

    # create the tmp / result folders
    problem_folder = f"./experiments/run{args.run}/problems/llm_step/{domain.name}"
    plan_folder    = f"./experiments/run{args.run}/plans/llm_step/{domain.name}"
    result_folder  = f"./experiments/run{args.run}/results/llm_step/{domain.name}"

    if not os.path.exists(problem_folder):
        os.system(f"mkdir -p {problem_folder}")
    if not os.path.exists(plan_folder):
        os.system(f"mkdir -p {plan_folder}")
    if not os.path.exists(result_folder):
        os.system(f"mkdir -p {result_folder}")

    task = args.task

    start_time = time.time()

    # A. generate problem pddl file
    task_suffix        = domain.get_task_suffix(task)
    task_nl, task_pddl = domain.get_task(task) 
    prompt             = planner.create_llm_stepbystep_prompt(task_nl, domain_nl)
    text_plan          = planner.query(prompt)

    # B. write the problem file into the problem folder
    text_plan_file_name = f"./experiments/run{args.run}/results/llm_step/{task_suffix}"
    with open(text_plan_file_name, "w") as f:
        f.write(text_plan)
    end_time = time.time()
    print(f"[info] task {task} takes {end_time - start_time} sec")


def llm_tot_ic_planner(args, planner, domain):
    """
    Tree of Thoughts planner
    """
    context          = domain.get_context()
    domain_pddl      = domain.get_domain_pddl()
    domain_pddl_file = domain.get_domain_pddl_file()
    domain_nl        = domain.get_domain_nl()
    domain_nl_file   = domain.get_domain_nl_file()

    # create the tmp / result folders
    problem_folder = f"./experiments/run{args.run}/problems/llm_tot_ic/{domain.name}"
    plan_folder    = f"./experiments/run{args.run}/plans/llm_tot_ic/{domain.name}"
    result_folder  = f"./experiments/run{args.run}/results/llm_tot_ic/{domain.name}"

    if not os.path.exists(problem_folder):
        os.system(f"mkdir -p {problem_folder}")
    if not os.path.exists(plan_folder):
        os.system(f"mkdir -p {plan_folder}")
    if not os.path.exists(result_folder):
        os.system(f"mkdir -p {result_folder}")

    task = args.task

    start_time = time.time()

    # A. generate problem pddl file
    task_suffix        = domain.get_task_suffix(task)
    task_nl, task_pddl = domain.get_task(task)
    text_plan = planner.tot_bfs(task_nl, domain_nl, context, time_left=200, max_depth=10)

    # B. write the problem file into the problem folder
    text_plan_file_name = f"./experiments/run{args.run}/results/llm_tot_ic/{task_suffix}"
    with open(text_plan_file_name, "w") as f:
        f.write(text_plan)
    end_time = time.time()
    print(f"[info] task {task} takes {end_time - start_time} sec")


def llm_ic_planner(args, planner, domain):
    """
    Baseline method:
        The LLM will be asked to directly give a plan based on the task description.
    """
    context          = domain.get_context()
    domain_pddl      = domain.get_domain_pddl()
    domain_pddl_file = domain.get_domain_pddl_file()
    domain_nl        = domain.get_domain_nl()
    domain_nl_file   = domain.get_domain_nl_file()

    # create the tmp / result folders
    problem_folder = f"./experiments/run{args.run}/problems/llm_ic/{domain.name}"
    plan_folder    = f"./experiments/run{args.run}/plans/llm_ic/{domain.name}"
    result_folder  = f"./experiments/run{args.run}/results/llm_ic/{domain.name}"

    if not os.path.exists(problem_folder):
        os.system(f"mkdir -p {problem_folder}")
    if not os.path.exists(plan_folder):
        os.system(f"mkdir -p {plan_folder}")
    if not os.path.exists(result_folder):
        os.system(f"mkdir -p {result_folder}")

    task = args.task

    start_time = time.time()

    # A. generate problem pddl file
    task_suffix        = domain.get_task_suffix(task)
    task_nl, task_pddl = domain.get_task(task) 
    prompt             = planner.create_llm_ic_prompt(task_nl, domain_nl, context)
    text_plan          = planner.query(prompt)

    # B. write the problem file into the problem folder
    text_plan_file_name = f"./experiments/run{args.run}/results/llm_ic/{task_suffix}"
    with open(text_plan_file_name, "w") as f:
        f.write(text_plan)
    end_time = time.time()
    print(f"[info] task {task} takes {end_time - start_time} sec")


def print_all_prompts(planner):
    for domain_name in DOMAINS:
        domain = eval(domain_name.capitalize())()
        context = domain.get_context()
        domain_pddl = domain.get_domain_pddl()
        domain_pddl_file = domain.get_domain_pddl_file()
        domain_nl = domain.get_domain_nl()
        
        for folder_name in [
            f"./prompts/llm/{domain.name}",
            f"./prompts/llm_step/{domain.name}",
            f"./prompts/llm_ic/{domain.name}",
            f"./prompts/llm_pddl/{domain.name}",
            f"./prompts/llm_ic_pddl/{domain.name}"]:
            if not os.path.exists(folder_name):
                os.system(f"mkdir -p {folder_name}")

        for task in range(len(domain)):
            task_nl_file, task_pddl_file = domain.get_task_file(task) 
            task_nl, task_pddl = domain.get_task(task) 
            task_suffix = domain.get_task_suffix(task)

            llm_prompt = planner.create_llm_prompt(task_nl, domain_nl)
            llm_stepbystep_prompt = planner.create_llm_stepbystep_prompt(task_nl, domain_nl)
            llm_ic_prompt = planner.create_llm_ic_prompt(task_nl, domain_nl, context)
            llm_pddl_prompt = planner.create_llm_pddl_prompt(task_nl, domain_nl)
            llm_ic_pddl_prompt = planner.create_llm_ic_pddl_prompt(task_nl, domain_pddl, context)
            with open(f"./prompts/llm/{task_suffix}.prompt", "w") as f:
                f.write(llm_prompt)
            with open(f"./prompts/llm_step/{task_suffix}.prompt", "w") as f:
                f.write(llm_stepbystep_prompt)
            with open(f"./prompts/llm_ic/{task_suffix}.prompt", "w") as f:
                f.write(llm_ic_prompt)
            with open(f"./prompts/llm_pddl/{task_suffix}.prompt", "w") as f:
                f.write(llm_pddl_prompt)
            with open(f"./prompts/llm_ic_pddl/{task_suffix}.prompt", "w") as f:
                f.write(llm_ic_pddl_prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-Planner")  #命令行参数解析器
    parser.add_argument('--domain', type=str, choices=DOMAINS, default="barman")  #添加--domain命令行参数
    parser.add_argument('--method', type=str, choices=["llm_ic_pddl_planner",
                                                       "llm_pddl_planner",
                                                       "llm_planner",
                                                       "llm_stepbystep_planner",
                                                       "llm_ic_planner",
                                                       "llm_tot_ic_planner"],
                                              default="llm_ic_pddl_planner")   #添加--method命令行参数
    parser.add_argument('--time-limit', type=int, default=200)  #添加--time-limit命令行参数，默认200s
    parser.add_argument('--task', type=int, default=0)   #添加--task命令行参数，必须是int
    parser.add_argument('--run', type=int, default=0)  #添加--run命令好参数，int，默认为0
    parser.add_argument('--print-prompts', action='store_true')  #action='store_true'：如果用户在命令行中指定了该参数，则其值为 True；否则为 False
    args = parser.parse_args() #解析命令行参数

    # 1. initialize problem domain
    domain = eval(args.domain.capitalize())()

    # 2. initialize the planner
    planner = Planner()

    # 3. execute the llm planner
    method = {
        "llm_ic_pddl_planner"   : llm_ic_pddl_planner,
        "llm_pddl_planner"      : llm_pddl_planner,
        "llm_planner"           : llm_planner,
        "llm_stepbystep_planner": llm_stepbystep_planner,
        "llm_ic_planner"        : llm_ic_planner,
        "llm_tot_ic_planner"       : llm_tot_ic_planner,
    }[args.method]  #根据用户选择的method从字典中获取对应的method

    if args.print_prompts:    #判断是否打印prompt
        print_all_prompts(planner)
    else:
        method(args, planner, domain)
